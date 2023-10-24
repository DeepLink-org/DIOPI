import torch
import triton
import triton.language as tl

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
    batch_ids = tl.load(p_token_ids + cur_batch_id_offset, mask=cur_batch_id_offset<cur_batch_end_index, other=0)
    batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset, mask=cur_batch_id_offset<cur_batch_end_index, other=0)
    
    row_start_ptr = Logits + cur_batch * stride_logit_b
    cur_offset = row_start_ptr + batch_ids
    cur_logits = tl.load(cur_offset, mask=cur_batch_id_offset<cur_batch_end_index, other=0.0)
    freq_logits = cur_logits - batch_ids_count * cur_frequency
    pre_logits = freq_logits - cur_presence
    output_ptr = Logits + cur_batch * stride_logit_b + batch_ids
    tl.store(output_ptr, pre_logits, mask=cur_batch_id_offset<cur_batch_end_index)

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
