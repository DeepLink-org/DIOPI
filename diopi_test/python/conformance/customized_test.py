import torch
import numpy as np
import math
from einops import rearrange
import torch.nn.functional as F


def _torch_context_attention(xq, xk, xv, bs, seqlen, num_head, head_dim):
    xq = xq.view(bs, seqlen, num_head, head_dim)
    xk = xk.view(bs, seqlen, num_head, head_dim)
    xv = xv.view(bs, seqlen, num_head, head_dim)
    mask = (
        torch.tril(torch.ones(seqlen, seqlen), diagonal=0)
        .unsqueeze(0)
        .unsqueeze(0)
        .cuda()
    )
    mask[mask == 0.0] = -100000000.0
    mask = mask.repeat(bs, num_head, 1, 1)
    keys = xk
    values = xv
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    scores = F.softmax(scores.float() + mask, dim=-1).type_as(xq)
    output = (
        torch.matmul(scores, values)
        .transpose(1, 2)
        .contiguous()
        .reshape(-1, num_head, head_dim)
    )
    return output


def multi_head_attention_inside(
    q, k, v, softmax_scale, causal=None, key_padding_mask=None
):
    # using for multiheadattention & varlen multiheadattention test
    batch_size, seqlen = q.shape[0], q.shape[1]
    causal = causal if causal is None else causal
    softmax_scale = softmax_scale or 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    if key_padding_mask is not None:
        padding_mask = torch.full(
            (batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device
        )
        padding_mask.masked_fill_(key_padding_mask, 0.0)
        scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
    if causal:
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    return output


class CustomizedTest(object):
    def cast_dtype(input, out):
        out = input.to(out.dtype, copy=True)
        return out

    def meshgrid(tensors, shape=None):
        return torch.meshgrid(tensors)

    def slice_op(input, dim, index):
        sizeI = input.size()
        slice_args = []
        for i in range(len(sizeI)):
            slice_args.append(slice(0, sizeI[i], 1))
        slice_args[dim] = index
        return torch.Tensor.__getitem__(input, slice_args)

    def index(input, **kwargs):
        new_args = []
        for ele in kwargs.values():
            if ele is None:
                hasEllipsis = True
                if hasEllipsis and Ellipsis not in new_args:
                    new_args.append(...)
            else:
                new_args.append(ele)
        return torch.Tensor.__getitem__(input, new_args)

    def sgd(
        param,
        param_grad,
        lr,
        buf=None,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        param.requires_grad = True
        param.grad = param_grad
        optimizer = torch.optim.SGD(
            [
                param,
            ],
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
        )
        optimizer.state[param]["momentum_buffer"] = buf
        optimizer.step()
        return param, buf

    def adam(
        param,
        param_grad,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step,
        amsgrad,
    ):
        params_with_grad = [param]
        grads = [param_grad]
        exp_avgs = [exp_avg]
        exp_avg_sqs = [exp_avg_sq]
        max_exp_avg_sqs = [max_exp_avg_sq]
        state_steps = [torch.tensor(float(step))]

        torch.optim._functional.adam(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=False,
        )
        return param, exp_avg, exp_avg_sq, max_exp_avg_sq

    def adamw(
        param,
        param_grad,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        lr,
        beta1,
        beta2,
        eps,
        step,
        weight_decay,
        amsgrad,
    ):
        params_with_grad = [param]
        grads = [param_grad]
        exp_avgs = [exp_avg]
        exp_avg_sqs = [exp_avg_sq]
        max_exp_avg_sqs = [max_exp_avg_sq]
        state_steps = [torch.tensor(float(step))]

        torch.optim._functional.adamw(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=False,
        )
        return param, exp_avg, exp_avg_sq, max_exp_avg_sq

    def adadelta(param, param_grad, square_avg, acc_delta, lr, rho, eps, weight_decay):
        params_with_grad = [param]
        grads = [param_grad]
        square_avgs = [square_avg]
        acc_deltas = [acc_delta]

        torch.optim._functional.adadelta(
            params_with_grad,
            grads,
            square_avgs,
            acc_deltas,
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            maximize=False,
        )
        return param, square_avg, acc_delta

    def rmsprop(
        param,
        param_grad,
        square_avg,
        grad_avg,
        momentum_buffer,
        lr,
        alpha,
        eps,
        weight_decay,
        momentum,
        centered,
    ):
        params = [param]
        grads = [param_grad]
        square_avgs = [square_avg]
        grad_avgs = [grad_avg]
        momentum_buffer_list = [momentum_buffer]

        torch.optim._functional.rmsprop(
            params,
            grads,
            square_avgs,
            grad_avgs,
            momentum_buffer_list,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        return param, square_avg, grad_avg, momentum_buffer

    def index_put(
        input, values, indices1, indices2=None, indices3=None, accumulate=False
    ):
        indices = [indices1]
        if indices2 is not None:
            indices.append(indices2)
        if indices3 is not None:
            indices.append(indices3)
        return torch.index_put(input, indices, values, accumulate)

    def im2col(input, kernel_size, dilation=1, padding=0, stride=1):
        return torch.nn.Unfold(kernel_size, dilation, padding, stride)(input)

    def col2im(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
        return torch.nn.Fold(output_size, kernel_size, dilation, padding, stride)(input)

    def clip_grad_norm_(tensors, max_norm, norm_type=2.0, error_if_nonfinite=False):
        parameters = []
        if torch.is_tensor(tensors):
            tensors = [tensors]
        for grad in tensors:
            tensor = torch.empty_like(grad)
            tensor.grad = grad
            parameters.append(tensor)
        return torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, norm_type, error_if_nonfinite
        )

    def ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=False,
    ):
        log_probs_ = log_probs.log_softmax(2)
        loss = torch.nn.functional.ctc_loss(
            log_probs_,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )
        return loss

    def linalgqr(input, mode):
        q, r = torch.linalg.qr(input, mode)
        out = [q, r]
        return out

    def batch_norm_stats(input, eps):
        mean, invstd = torch.batch_norm_stats(input, eps)
        out = (mean, invstd)
        return out

    def batch_norm_gather_stats_with_counts(
        input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all
    ):
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all,
        )
        out = (mean, invstd)
        return out

    def batch_norm_backward_reduce(
        grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g
    ):
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g
        )
        if input_g:
            out = (sum_dy, sum_dy_xmu, grad_weight, grad_bias)
        else:
            out = (None, None, grad_weight, grad_bias)
        return out

    def batch_norm_backward_elemt(
        grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count
    ):
        grad_input = torch.batch_norm_backward_elemt(
            grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count
        )
        out = grad_input
        return out

    def batch_norm_elemt(input, weight, bias, mean, invstd, eps):
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        return out

    def rotary_emb(input, cos, sin, conj, interleaved):
        x1, x2 = input.chunk(2, dim=-1)
        data_type = input.dtype
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
        if not conj:
            out1 = x1 * cos - x2 * sin
            out2 = x1 * sin + x2 * cos
        else:
            out1 = x1 * cos + x2 * sin
            out2 = -x1 * sin + x2 * cos
        out1 = out1.to(data_type)
        out2 = out2.to(data_type)
        out = torch.cat((out1, out2), dim=-1)
        return out

    def rms_norm(input, normalized_shape, weight, bias, eps):
        if normalized_shape is not None:
            dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
        else:
            dims = -1
        ori_dtype = input.dtype
        variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
        inv_rms = torch.rsqrt(variance + eps)
        input = input * inv_rms
        out = weight * input + bias if bias is not None else weight * input
        out = out.to(ori_dtype)
        return out

    def sort(input, dim, descending, stable=False):
        # Skip compare while stable==False
        sizeI = input.size()
        sorted, indices = torch.sort(
            input, dim=dim, descending=descending, stable=stable
        )
        if len(sizeI) > 0 and not stable:
            return sorted
        else:
            return sorted, indices

    def multihead_attention(q, k, v, dropout_p, is_causal, return_debug_mask, scale):
        # In order to compare the accuracy with the baseline value, dropout is not used during testing.
        output = multi_head_attention_inside(q, k, v, scale, is_causal)
        return output

    def multihead_attention_varlen(
        q, k, v, cu_seqlens, max_seqlen, dropout_p, is_causal, return_debug_mask, scale
    ):
        # In order to compare the accuracy with the baseline value, dropout is not used during testing.
        batch_size = len(cu_seqlens) - 1
        _, head_num, head_dim = q.size()
        device = q.device

        padded_shape = (batch_size, max_seqlen, head_num, head_dim)
        q_padded = torch.zeros(padded_shape, dtype=q.dtype, device=device)
        k_padded = torch.zeros(padded_shape, dtype=k.dtype, device=device)
        v_padded = torch.zeros(padded_shape, dtype=v.dtype, device=device)

        # Initialize the key_padding_mask as a Boolean mask with False values
        key_padding_mask = torch.zeros(
            (batch_size, max_seqlen), dtype=torch.bool, device=device
        )
        # Fill the key_padding_mask with True values at positions with actual data (cu_seqlens)
        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            actual_seq_len = end_idx - start_idx
            key_padding_mask[i, :actual_seq_len] = True
            q_padded[i, :actual_seq_len, :, :] = q[start_idx:end_idx, :, :]
            k_padded[i, :actual_seq_len, :, :] = k[start_idx:end_idx, :, :]
            v_padded[i, :actual_seq_len, :, :] = v[start_idx:end_idx, :, :]

        qkv_padded_result = multi_head_attention_inside(
            q_padded, k_padded, v_padded, scale, is_causal, key_padding_mask
        )
        output = torch.zeros(q.shape, dtype=q.dtype, device=device)

        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            actual_seq_len = end_idx - start_idx
            output[start_idx:end_idx, :, :] = qkv_padded_result[
                i, :actual_seq_len, :, :
            ]
        return output

    def flash_attention_v1(
        q, k, v, p_dropout, softmax_scale, is_causal, head_num, input_layout
    ):
        # In order to compare the accuracy with the baseline value, dropout is not used during testing.
        # For calculation convenience, convert to BSND.
        if input_layout == "SBH":
            q, k, v = [
                rearrange(x, "s b (n d) -> b s n d", n=head_num) for x in [q, k, v]
            ]
        elif input_layout == "BSH":
            q, k, v = [
                rearrange(x, "b s (n d) -> b s n d", n=head_num) for x in [q, k, v]
            ]
        elif input_layout == "BNSD":
            q, k, v = [rearrange(x, "b n s d-> b s n d") for x in [q, k, v]]
        seqlen = q.shape[1]
        softmax_scale = (
            1.0 / math.sqrt(q.shape[-1]) if not softmax_scale else softmax_scale
        )
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if is_causal:
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), float("-inf"), device=scores.device), 1
            )
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        output = torch.einsum("bhts,bshd->bthd", attention, v)
        if input_layout == "SBH":
            output = rearrange(output, "b s n d -> s b (n d)")
        elif input_layout == "BSH":
            output = rearrange(output, "b s n d -> b s (n d)")
        elif input_layout == "BNSD":
            output = rearrange(output, "b s n d -> b n s d")
        return output

    def flash_attention_v3(q, k, v, p_dropout, softmax_scale, is_causal):
        # In order to compare the accuracy with the baseline value, dropout is not used during testing.
        seqlen = q.shape[1]
        softmax_scale = (
            1.0 / math.sqrt(q.shape[-1]) if not softmax_scale else softmax_scale
        )
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if is_causal:
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), float("-inf"), device=scores.device), 1
            )
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        output = torch.einsum("bhts,bshd->bthd", attention, v)
        return output

    def flash_attention_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        p_dropout,
        softmax_scale,
        is_causal,
    ):
        # Currently, only equality between cu_seqlens_q and cu_seqlens_kv is supported here
        cu_seqlens = cu_seqlens_q
        max_seqlen = max_seqlen_q
        # In order to compare the accuracy with the baseline value, dropout is not used during testing.
        batch_size = len(cu_seqlens) - 1
        _, head_num, head_dim = q.size()
        device = q.device

        padded_shape = (batch_size, max_seqlen, head_num, head_dim)
        q_padded = torch.zeros(padded_shape, dtype=q.dtype, device=device)
        k_padded = torch.zeros(padded_shape, dtype=k.dtype, device=device)
        v_padded = torch.zeros(padded_shape, dtype=v.dtype, device=device)

        # Initialize the key_padding_mask as a Boolean mask with False values
        key_padding_mask = torch.zeros(
            (batch_size, max_seqlen), dtype=torch.bool, device=device
        )
        # Fill the key_padding_mask with True values at positions with actual data (cu_seqlens)
        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            actual_seq_len = end_idx - start_idx
            key_padding_mask[i, :actual_seq_len] = True
            q_padded[i, :actual_seq_len, :, :] = q[start_idx:end_idx, :, :]
            k_padded[i, :actual_seq_len, :, :] = k[start_idx:end_idx, :, :]
            v_padded[i, :actual_seq_len, :, :] = v[start_idx:end_idx, :, :]

        qkv_padded_result = multi_head_attention_inside(
            q_padded, k_padded, v_padded, softmax_scale, is_causal, key_padding_mask
        )
        output = torch.zeros(q.shape, dtype=q.dtype, device=device)

        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            actual_seq_len = end_idx - start_idx
            output[start_idx:end_idx, :, :] = qkv_padded_result[
                i, :actual_seq_len, :, :
            ]
        return output

    def scaled_masked_softmax(input, mask, scale, fixed_triu_mask):
        if fixed_triu_mask:
            mask_tri = torch.triu(
                torch.ones(mask.shape, device=input.device), diagonal=1
            ).bool()
            mask_data = (input * scale).masked_fill(mask_tri, value=-1e4)
        else:
            mask_data = (input * scale).masked_fill(mask, value=-1e4)
        output = torch.nn.functional.softmax(mask_data, dim=-1)
        return output

    def apply_penalty(
        logits,
        presence_penalty,
        frequency_penalty,
        p_token_ids,
        p_token_counts,
        p_cumsum_seq_len,
        p_max_len_in_batch,
    ):
        batch = logits.shape[0]
        for i in range(batch):
            cur_batch_start_index = p_cumsum_seq_len[i]
            cur_batch_end_index = p_cumsum_seq_len[i + 1]
            cur_logits = logits[
                i, p_token_ids[cur_batch_start_index:cur_batch_end_index]
            ]
            cur_logits = (
                cur_logits
                - p_token_counts[cur_batch_start_index:cur_batch_end_index]
                * frequency_penalty[i]
                - presence_penalty[i]
            )
            logits[i, p_token_ids[cur_batch_start_index:cur_batch_end_index]] = (
                cur_logits
            )
        return logits

    def destindex_copy_kv(k, dest_loc, out):
        out[dest_loc] = k
        return out

    def token_attention(q, k, out, b_loc, b_start_loc, b_seq_len, max_input_len):
        batch, head, dim = b_loc.shape[0], q.shape[1], q.shape[2]
        q_device = q.device
        xq = q.view(batch, 1, head, dim).transpose(1, 2)
        for i in range(batch):
            k_loc = b_loc[i][
                max_input_len
                - b_seq_len[i]
                + torch.arange(0, b_seq_len[i], device=q_device)
            ]
            key = k[k_loc, :].view(1, b_seq_len[i], head, dim).transpose(1, 2)
            out_loc = b_start_loc[i] + torch.arange(0, b_seq_len[i], device=q_device)
            out[:, out_loc] = (
                torch.matmul(xq[i, :], key.transpose(2, 3)) / math.sqrt(dim)
            ).reshape(head, b_seq_len[i])
        return out

    def token_softmax_reducev(
        logics, v, out, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index
    ):
        batch, head, dim = b_loc.shape[0], v.shape[1], v.shape[2]
        for i in range(batch):
            v_loc = b_loc[i][
                max_input_len
                - b_seq_len[i]
                + torch.arange(0, b_seq_len[i], device=logics.device)
            ]
            P = (
                logics[:, b_start_loc[i] : b_start_loc[i] + b_seq_len[i]]
                .softmax(-1)
                .reshape(head, 1, 1, b_seq_len[i])
                .transpose(0, 1)
            )
            V = v[v_loc, :].view(1, b_seq_len[i], head, dim).transpose(1, 2)
            out[i, :] = torch.matmul(P, V).view(1, head, dim)
        return out

    def context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len):
        batch, head, dim = b_start_loc.shape[0], q.shape[1], q.shape[2]
        for i in range(batch):
            start = b_start_loc[i]
            end = start + b_seq_len[i]
            out[start:end, :] = _torch_context_attention(
                q[start:end],
                k[start:end],
                v[start:end],
                1,
                int(b_seq_len[i]),
                head,
                dim,
            )
        return out

    def attention(
        query,
        key,
        value,
        attn_mask=None,
        attn_bias=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ):
        query = query.permute(0, 2, 1, 3)  # BSND -> BNSD
        key = key.permute(0, 2, 1, 3)  # BSND -> BNSD
        value = value.permute(0, 2, 1, 3)  # BSND -> BNSD
        half_use_float = query.is_cpu and (
            query.dtype == torch.float16 or query.dtype == torch.bfloat16
        )
        raw_dtype = query.dtype
        device = query.device
        if half_use_float:
            query = query.float()
            key = key.float()
            value = value.float()
        batch_size, seq_len_q, seq_len_kv = query.size(0), query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias_temp = torch.zeros(
            seq_len_q, seq_len_kv, dtype=query.dtype, device=device
        )
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(
                seq_len_q, seq_len_kv, dtype=torch.bool, device=device
            ).tril(diagonal=0)
            attn_bias_temp.masked_fill_(temp_mask.logical_not(), -10000.0)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias_temp.masked_fill_(attn_mask.logical_not(), -10000.0)
            else:
                assert False, "atten_mask dtype is not bool"

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias_temp
        if attn_bias is not None:
            attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        out = attn_weight @ value
        if half_use_float:
            out = out.to(raw_dtype)
        out = out.permute(0, 2, 1, 3)  # BNSD -> BSND
        return out

    def attention_varlen(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        attn_mask=None,
        attn_bias=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ):

        batch_size = cu_seqlens_q.numel() - 1
        _, head_num, head_dim = query.size()
        device = query.device
        padded_q_shape = (batch_size, max_seqlen_q, head_num, head_dim)  # BSND
        padded_kv_shape = (batch_size, max_seqlen_kv, head_num, head_dim)  # BSND

        query_padded = torch.zeros(padded_q_shape, dtype=query.dtype, device=device)
        key_padded = torch.zeros(padded_kv_shape, dtype=key.dtype, device=device)
        value_padded = torch.zeros(padded_kv_shape, dtype=value.dtype, device=device)

        padded_attn_bias = torch.zeros(
            batch_size, 1, max_seqlen_q, max_seqlen_kv, dtype=query.dtype, device=device
        )  # BNSS
        # Fill the key_padding_mask with True values at positions with actual data (cu_seqlens)
        for i in range(batch_size):
            actual_q_seq_len = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
            actual_kv_seq_len = cu_seqlens_kv[i + 1] - cu_seqlens_kv[i]
            padded_attn_bias[i, :, actual_q_seq_len:, :] = -10000.0
            padded_attn_bias[i, :, :, actual_kv_seq_len:] = -10000.0
            query_padded[i, :actual_q_seq_len, :, :] = query[
                cu_seqlens_q[i] : cu_seqlens_q[i + 1], :, :
            ]  # BSND
            key_padded[i, :actual_kv_seq_len, :, :] = key[
                cu_seqlens_kv[i] : cu_seqlens_kv[i + 1], :, :
            ]  # BSND
            value_padded[i, :actual_kv_seq_len, :, :] = value[
                cu_seqlens_kv[i] : cu_seqlens_kv[i + 1], :, :
            ]

            actual_q_seq_len = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
        if attn_bias is None:
            attn_bias = padded_attn_bias
        else:
            attn_bias += padded_attn_bias
        out_paded = CustomizedTest.attention(
            query=query_padded,
            key=key_padded,
            value=value_padded,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            attn_bias=attn_bias,
        )  # BSND

        out = torch.zeros(query.shape, dtype=query.dtype, device=device)
        for i in range(batch_size):
            start_idx = cu_seqlens_q[i]
            end_idx = cu_seqlens_q[i + 1]
            actual_seq_len = end_idx - start_idx
            out[start_idx:end_idx, :, :] = out_paded[i, :actual_seq_len, :, :] # BSND->TND
        return out
