import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F


@triton.jit
def _fwd_kernel_apply_penalty(
    Logits, presence_penalty, frequency_penalty,
    p_token_ids, p_token_counts, p_cumsum_seq_len,
    stride_logit_b, stride_logit_s,
    BLOCK_P: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_frequency = tl.load(frequency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)

    cur_batch_id_offset = cur_batch_start_index + tl.arange(0, BLOCK_P)
    batch_ids = tl.load(p_token_ids + cur_batch_id_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0)
    batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0)

    row_start_ptr = Logits + cur_batch * stride_logit_b
    cur_offset = row_start_ptr + batch_ids
    cur_logits = tl.load(cur_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
    freq_logits = cur_logits - batch_ids_count * cur_frequency
    pre_logits = freq_logits - cur_presence
    output_ptr = Logits + cur_batch * stride_logit_b + batch_ids
    tl.store(output_ptr, pre_logits, mask=cur_batch_id_offset < cur_batch_end_index)

    return


@torch.no_grad()
def apply_penalty(Logits, presence_penalty, frequency_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch):
    assert Logits.is_contiguous()
    BLOCK = triton.next_power_of_2(p_max_len_in_batch)
    if BLOCK <= 512:
        BLOCK = 512
    elif BLOCK <= 1024:
        BLOCK = 1024
    num_warps = 8
    _fwd_kernel_apply_penalty[(Logits.shape[0], )](
        Logits, presence_penalty, frequency_penalty,
        p_token_ids, p_token_counts, p_cumsum_seq_len,
        Logits.stride(0), Logits.stride(1),
        num_warps=num_warps,
        BLOCK_P=BLOCK
    )
    return


@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K, Dest_loc,
    Out,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    head_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Dest_loc + cur_index)

    k_ptrs = K + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    k = tl.load(k_ptrs, mask=offs_h[:, None] < head_num, other=0.0)
    tl.store(o_ptrs, k, mask=offs_h[:, None] < head_num)
    return


@torch.no_grad()
def destindex_copy_kv(K, DestLoc, Out):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_kv[grid](
        K, DestLoc, Out,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        head_num,
        BLOCK_DMODEL=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_token_atttention(
    Q, K, sm_scale, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
    Att_Out,
    stride_b_loc_b, stride_b_loc_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    att_stride_h, att_stride_bs,

    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    cur_batch_start_index = max_input_len - cur_batch_seq_len
    cur_batch_end_index = max_input_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(B_Loc + stride_b_loc_b * cur_batch + stride_b_loc_s * offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
    return


@torch.no_grad()
def token_attention_fwd(q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)

    batch, head_num = B_Loc.shape[0], q.shape[1]

    grid = (batch, head_num, triton.cdiv(max_input_len, BLOCK))

    num_warps = 4

    _fwd_kernel_token_atttention[grid](
        q, k, sm_scale, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
        att_out,
        B_Loc.stride(0), B_Loc.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        att_out.stride(0), att_out.stride(1),
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_token_softmax_reducev(
    Logics, V, Out,
    B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
    stride_logic_h, stride_logic_bs,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_b_loc_b, stride_b_loc_s,
    other_kv_index,  # 避免读取到nan的数据
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_v = cur_head * stride_vh + offs_d[None, :] * stride_vd
    off_b_loc = cur_batch * stride_b_loc_b + (max_input_len - cur_batch_seq_len) * stride_b_loc_s

    v_ptrs = V + off_v

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(B_Loc + off_b_loc + (start_n + offs_n) * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=other_kv_index)

        qk = tl.load(Logics + cur_head * stride_logic_h + (cur_batch_start_loc + start_n + offs_n) * stride_logic_bs,
                     mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v = tl.load(v_ptrs + v_index[:, None] * stride_vbs)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max

    acc = acc / e_sum
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return


@torch.no_grad()
def token_softmax_reducev_fwd(logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index):
    BLOCK = 64
    batch, head = b_seq_len.shape[0], logics.shape[0]
    grid = (batch, head)
    num_warps = 1
    _fwd_kernel_token_softmax_reducev[grid](
        logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len,
        logics.stride(0), logics.stride(1),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        b_loc.stride(0), b_loc.stride(1),
        other_kv_index,
        BLOCK_DMODEL=v.shape[-1],
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=3
    )
    return


if triton.__version__ >= "2.1.0":
    @triton.jit
    def _fwd_kernel_context_attention(
        Q, K, V, sm_scale, B_Start_Loc, B_Seqlen,  # B_LOC 内部记录每个batch 输入的真实位置， B_SEQ_len 记录当前输入的真实长度
        Out,
        stride_qbs, stride_qh, stride_qd,
        stride_kbs, stride_kh, stride_kd,
        stride_vbs, stride_vh, stride_vd,
        stride_obs, stride_oh, stride_od,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
        off_k = offs_n[None, :] * stride_kbs + cur_head * stride_kh + offs_d[:, None] * stride_kd
        off_v = offs_n[:, None] * stride_vbs + cur_head * stride_vh + offs_d[None, :] * stride_vd

        q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

        k_ptrs = K + off_k
        v_ptrs = V + off_v

        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=(start_n + offs_n[None, :]) < cur_batch_seq_len, other=0.0)
            # mask = tl.load(mask_ptrs + start_n, mask=start_n + offs_n < cur_batch_end_loc, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)

            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        # initialize pointers to output
        off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
        out_ptrs = Out + off_o
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
        return

    @torch.no_grad()
    def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}

        sm_scale = 1.0 / (Lq**0.5)  # 计算scale系数
        batch, head = b_seq_len.shape[0], q.shape[1]

        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,

        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel_context_attention[grid](
            q, k, v, sm_scale, b_start_loc, b_seq_len,
            o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        return

elif triton.__version__ == "2.0.0":
    @triton.jit
    def _fwd_kernel_context_attention(
        Q, K, V, sm_scale, B_Start_Loc, B_Seqlen,
        TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
        Out,
        stride_qbs, stride_qh, stride_qd,
        stride_kbs, stride_kh, stride_kd,
        stride_vbs, stride_vh, stride_vd,
        stride_obs, stride_oh, stride_od,
        stride_tmp_b, stride_tmp_h, stride_tmp_s,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
        off_k = offs_n[None, :] * stride_kbs + cur_head * stride_kh + offs_d[:, None] * stride_kd
        off_v = offs_n[:, None] * stride_vbs + cur_head * stride_vh + offs_d[None, :] * stride_vd
        q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

        k_ptrs = K + off_k
        v_ptrs = V + off_v

        t_ptrs = TMP + cur_batch * stride_tmp_b + cur_head * stride_tmp_h + offs_m * stride_tmp_s
        # t_ptrs = TMP + offs_m
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=(start_n + offs_n[None, :]) < cur_batch_seq_len, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            tl.store(t_ptrs, acc_scale)
            acc_scale = tl.load(t_ptrs)  # BUG: have to store and immediately load
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)

            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        # initialize pointers to output
        off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
        out_ptrs = Out + off_o
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)

        return

    @torch.no_grad()
    def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}

        sm_scale = 1.0 / (Lq**0.5)
        batch, head = b_seq_len.shape[0], q.shape[1]

        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))

        tmp = torch.empty((batch, head, max_input_len + 256), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        # num_warps = 4
        _fwd_kernel_context_attention[grid](
            q, k, v, sm_scale, b_start_loc, b_seq_len,
            tmp,
            o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            tmp.stride(0), tmp.stride(1), tmp.stride(2),
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        return


else:
    raise Exception("error triton version!")


def torch_att(xq, xk, xv, bs, seqlen, num_head, head_dim):
    xq = xq.view(bs, seqlen, num_head, head_dim)
    xk = xk.view(bs, seqlen, num_head, head_dim)
    xv = xv.view(bs, seqlen, num_head, head_dim)
    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0).cuda()
    mask[mask == 0.] = -100000000.0
    mask = mask.repeat(bs, num_head, 1, 1)
    keys = xk
    values = xv
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    scores = F.softmax(scores.float() + mask, dim=-1).type_as(xq)
    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, head_dim)
    return output


def context_attention(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
    batch, head, dim = b_start_loc.shape[0], q.shape[1], q.shape[2]
    for i in range(batch):
        start = b_start_loc[i]
        end = start + b_seq_len[i]
        o[start:end, :] = torch_att(q[start:end], k[start:end], v[start:end], 1, int(b_seq_len[i]), head, dim)
    return
