import pickle
import numpy as np
import os
import sys
import torch
import torchvision
from . import triton_kernels

from gen_input import GenPolicy
from conformance.utils import logger, get_data_from_file
from conformance.db_operation import db_conn


def multihead_attention_inside(q, k, v, softmax_scale, causal=None, key_padding_mask=None):
    from einops import rearrange
    import math
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

    def sgd(param, param_grad, lr, buf=None, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        param.requires_grad = True
        param.grad = param_grad
        optimizer = torch.optim.SGD([param, ], lr, momentum, dampening, weight_decay, nesterov)
        optimizer.state[param]['momentum_buffer'] = buf
        optimizer.step()
        return param, buf

    def adam(param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad):
        params_with_grad = [param]
        grads = [param_grad]
        exp_avgs = [exp_avg]
        exp_avg_sqs = [exp_avg_sq]
        max_exp_avg_sqs = [max_exp_avg_sq]
        state_steps = [torch.tensor(float(step))]

        torch.optim._functional.adam(params_with_grad,
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
                                     maximize=False)
        return param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq

    def adamw(param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, step, weight_decay, amsgrad):
        params_with_grad = [param]
        grads = [param_grad]
        exp_avgs = [exp_avg]
        exp_avg_sqs = [exp_avg_sq]
        max_exp_avg_sqs = [max_exp_avg_sq]
        state_steps = [torch.tensor(float(step))]

        torch.optim._functional.adamw(params_with_grad,
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
                                      maximize=False)
        return param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq

    def adadelta(param, param_grad, square_avg, acc_delta, lr, rho, eps, weight_decay):
        params_with_grad = [param]
        grads = [param_grad]
        square_avgs = [square_avg]
        acc_deltas = [acc_delta]

        torch.optim._functional.adadelta(params_with_grad,
                                         grads,
                                         square_avgs,
                                         acc_deltas,
                                         lr=lr,
                                         rho=rho,
                                         eps=eps,
                                         weight_decay=weight_decay,
                                         maximize=False)
        return param, param_grad, square_avg, acc_delta

    def rmsprop(param, param_grad, square_avg, grad_avg, momentum_buffer, lr, alpha, eps, weight_decay, momentum, centered):
        params = [param]
        grads = [param_grad]
        square_avgs = [square_avg]
        grad_avgs = [grad_avg]
        momentum_buffer_list = [momentum_buffer]

        torch.optim._functional.rmsprop(params,
                                        grads,
                                        square_avgs,
                                        grad_avgs,
                                        momentum_buffer_list,
                                        lr=lr,
                                        alpha=alpha,
                                        eps=eps,
                                        weight_decay=weight_decay,
                                        momentum=momentum,
                                        centered=centered)
        return param, param_grad, square_avg, grad_avg, momentum_buffer

    def index_put(input, values, indices1, indices2=None, indices3=None, accumulate=False):
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
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite)

    def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
        log_probs_ = log_probs.log_softmax(2)
        loss = torch.nn.functional.ctc_loss(log_probs_, targets, input_lengths, target_lengths, blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        return loss

    def linalgqr(input, mode):
        q, r = torch.linalg.qr(input, mode)
        out = [q, r]
        return out

    def batch_norm_stats(input, eps):
        mean, invstd = torch.batch_norm_stats(input, eps)
        out = (mean, invstd)
        return out

    def batch_norm_gather_stats_with_counts(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all):
        mean, invstd = torch.batch_norm_gather_stats_with_counts(input, mean_all, invstd_all, running_mean, running_var, momentum, eps, count_all)
        out = (mean, invstd)
        return out

    def batch_norm_backward_reduce(grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g):
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g)
        out = (sum_dy, sum_dy_xmu, grad_weight, grad_bias)
        return out

    def batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count):
        grad_input = torch.batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count)
        out = grad_input
        return out

    def batch_norm_elemt(input, weight, bias, mean, invstd, eps):
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        return out

    def rotary_emb(input, cos, sin, conj):
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
        variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + eps)
        out = weight * input
        return out

    def multihead_attention_forward(q, k, v, dropout_p, is_causal, return_debug_mask, scale):
        # 为了保证精度，因此在test的时候不使用dropout
        output = multihead_attention_inside(q, k, v, scale, is_causal)
        return output

    def varlen_multihead_attention_forward(q, k, v, cu_seqlens, max_seqlen, dropout_p, is_causal, return_debug_mask, scale):
        # 为了保证精度，因此在test的时候不使用dropout
        from einops import rearrange
        import math
        batch_size = len(cu_seqlens)-1
        seq_len = max_seqlen
        _, num_heads, feature_size = q.size()
        # Initialize the key_padding_mask as a Boolean mask with False values
        key_padding_mask = torch.zeros((batch_size, max_seqlen), dtype=torch.bool, device="cuda")

        # Fill the key_padding_mask with True values at positions with actual data (cu_seqlens)
        for i in range(batch_size):
            seq_len_in = cu_seqlens[i+1]-cu_seqlens[i]
            key_padding_mask[i, :seq_len_in] = True
        padded_q_shape = (batch_size, seq_len, num_heads, feature_size)
        q_padded = torch.zeros(padded_q_shape, dtype=torch.float16, device="cuda")
        k_padded = torch.zeros(padded_q_shape, dtype=torch.float16, device="cuda")
        v_padded = torch.zeros(padded_q_shape, dtype=torch.float16, device="cuda")
        for i in range(batch_size):
            seq_len = cu_seqlens[i+1] - cu_seqlens[i]
            q_padded[i, :seq_len, :, :] = q[cu_seqlens[i]:cu_seqlens[i + 1], :, :]
            k_padded[i, :seq_len, :, :] = k[cu_seqlens[i]:cu_seqlens[i + 1], :, :]
            v_padded[i, :seq_len, :, :] = v[cu_seqlens[i]:cu_seqlens[i + 1], :, :]
        qkv_result = multihead_attention_inside(q_padded, k_padded, v_padded, scale, is_causal, key_padding_mask)
        output = torch.zeros(q.shape, dtype=torch.float16).cuda()
        for i in range(1, len(cu_seqlens)):
            start_idx = cu_seqlens[i - 1]
            end_idx = cu_seqlens[i]
            output[start_idx:end_idx, :, :] = qkv_result[i - 1, :end_idx - start_idx, :, :]
        return output

    def apply_penalty(logits, presence_penalty, frequency_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch):
        triton_kernels.apply_penalty(logits, presence_penalty, frequency_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch)
        return logits

    def destindex_copy_kv(k, dest_loc, out):
        triton_kernels.destindex_copy_kv(k, dest_loc, out)
        return out

    def token_attention(q, k, out, b_loc, b_start_loc, b_seq_len, max_input_len):
        triton_kernels.token_attention_fwd(q, k, out, b_loc, b_start_loc, b_seq_len, max_input_len)
        return out

    def token_softmax_reducev(logics, v, out, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index):
        triton_kernels.token_softmax_reducev_fwd(logics, v, out, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index)
        return out

    def context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len):
        # triton_kernels.context_attention_fwd(q, k, v, out, b_start_loc, b_seq_len, max_input_len)
        triton_kernels.context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len)
        return out


class GenOutputData(object):
    r'''
    Generate output data for all functions by using numpy and input data
    '''
    db_case_items = {}

    @staticmethod
    def run(diopi_item_config_path='diopi_case_items.cfg', input_path='data/inputs/',
            output_path='data/outputs/', fname='all_ops', model_name='diopi'):
        if not os.path.exists(input_path):
            logger.error("Input data is not generated!")
            sys.exit(0)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(diopi_item_config_path, 'rb') as f:
            all_cfg_dict = pickle.load(f)

        # XXX save case number in glob_var
        case_counter = 0
        func_name_list = []  # make the info log once

        for case_name in all_cfg_dict:
            each_cfg_dict = all_cfg_dict[case_name]
            func_name = each_cfg_dict["name"]
            item = {'case_name': case_name, 'model_name': model_name}
            if fname not in [func_name, 'all_ops']:
                continue
            data_path = os.path.join(input_path, case_name)
            input_ = get_data_from_file(data_path, case_name, 'input')
            if "no_output_ref" in each_cfg_dict:
                logger.info(f'diopi_functions.{func_name} [{case_name}] is set to no_output_ref, skip generate output')
                continue

            gen_tensor_obj = GenTensor(case_name, each_cfg_dict)

            try:
                output, saved_grads = gen_tensor_obj.gen_data(input_)
                item['result'] = 'passed'
            except Exception as err_msg:
                logger.error(f'Generate output data for diopi_functions.{func_name} [{case_name}] failed, cause by \n{err_msg}')
                item.update({'result': 'failed', 'err_msg': err_msg})
                continue
            finally:
                GenOutputData.db_case_items[case_name] = item
            if output is not None:
                with open(os.path.join(output_path, case_name), "wb") as f:
                    pickle.dump(GenOutputData.to_numpy(output), f, protocol=4)
                    logger_str = "output"
                    case_counter += 1
                if saved_grads is not None:
                    saved_backward_pth = case_name.split(".pth")[0] + "_backward.pth"
                    with open(os.path.join(output_path, saved_backward_pth), "wb") as f:
                        pickle.dump(GenOutputData.to_numpy(saved_grads), f, protocol=4)
                    logger_str = f"{logger_str} and backward"

                if func_name not in func_name_list:
                    func_signature = f"diopi_functions.{func_name}"
                    logger.info(f"Generate benchmark {logger_str} data for {func_signature}")
                    func_name_list.append(func_name)

        logger.info(f"Generate test cases number for output data: {case_counter}")
        if case_counter == 0:
            logger.info("No benchmark output data is generated")
        else:
            logger.info("Generate benchmark output and backward data done!")

    @staticmethod
    def to_numpy(tensors):
        if isinstance(tensors, torch.Tensor):
            ndarrays = tensors.detach().cpu().numpy()
        elif isinstance(tensors, (list, tuple)):
            ndarrays = []
            for i in range(len(tensors)):
                if isinstance(tensors[i], torch.Tensor):
                    ndarrays.append(tensors[i].detach().cpu().numpy())
                else:
                    ndarrays.append(tensors[i])
        elif isinstance(tensors, dict):
            ndarrays = {}
            for k, v in tensors.items():
                if isinstance(v, torch.Tensor):
                    tmp = {k: v.detach().cpu().numpy()}
                else:
                    tmp = {k: v}
                ndarrays.update(tmp)
        elif isinstance(tensors, (int, float)):
            ndarrays = np.array(tensors)
        else:
            ndarrays = None

        return ndarrays


class GenTensor(object):
    def __init__(self, case_name, case_cfg) -> None:
        self.case_name = case_name
        self.case_cfg = case_cfg
        self.func_name = case_cfg["name"]
        self.module = "torch.nn.functional"
        self.input = None
        self.output = None
        self.if_forward_success = False

    def gen_data(self, input_data):
        output = self.gen_forward_data(input_data)
        saved_grads = self.gen_backward_data(input_data)
        return output, saved_grads

    def gen_forward_data(self, input_data):
        if self.case_cfg['interface']:
            self.module = self.case_cfg["interface"][0]
        function_paras = input_data["function_paras"]
        self.transfer_tensor_to_device(function_paras)
        kwargs = function_paras['kwargs']
        if self.module == "torch.Tensor":
            input = kwargs['input']
            self.input = input
            self.module = "input"
            del kwargs['input']
        if 'dtype' in kwargs.keys():
            kwargs['dtype'] = self.change_np_dtype_to_torch(kwargs['dtype'])
        func_call = f"{self.module}.{self.func_name}(**kwargs)"

        try:
            self.output = eval(func_call)
            self.if_forward_success = True
        except Exception as e:
            logger.error(f"Failed to execute function {func_call}, caused by {e}")
        return self.output

    def gen_backward_data(self, input_data):
        if not self.if_forward_success:
            return None
        function_paras = input_data["function_paras"]
        kwargs = function_paras['kwargs']
        saved_grads = None
        if function_paras["requires_grad"]:
            if self.module == "input":
                kwargs['input'] = self.input
            outputs = self.output
            if not isinstance(self.output, (list, tuple)):
                outputs = [self.output]

            requires_backward = self.case_cfg["requires_backward"]
            outputs_for_backward = outputs if len(requires_backward) == 0 \
                else [outputs[i] for i in requires_backward]

            inputs_name_for_grad, inputs_for_grad = self.get_name_and_data_for_grad(function_paras)
            if len(inputs_for_grad) != 0:
                grad_outputs = [torch.ones_like(i) for i in outputs_for_backward]
                grads = torch.autograd.grad(
                    outputs_for_backward, inputs_for_grad, grad_outputs, allow_unused=True)
                saved_grads = {k: v for k, v in zip(inputs_name_for_grad, grads)}
        return saved_grads

    def transfer_tensor_to_device(self, function_paras: dict):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for para in function_paras["kwargs"].keys():
            if isinstance(function_paras['kwargs'][para], np.ndarray):
                tensor = torch.from_numpy(function_paras['kwargs'][para])
                if function_paras["requires_grad"].get(para, []) == [True]:
                    tensor.requires_grad = True
                function_paras['kwargs'][para] = tensor.to(device=device)

            gen_policy = [i.get('gen_policy', None) for i in self.case_cfg['tensor_para']['args'] if i['ins'] == para]
            if_gen_list = len(gen_policy) > 0 and gen_policy[0] in GenPolicy.gen_list_policy
            if if_gen_list:
                if isinstance(function_paras['kwargs'][para], (list, tuple)):
                    tensors = function_paras['kwargs'][para]
                    for idx, ele in enumerate(tensors):
                        tensors[idx] = torch.from_numpy(ele).to(device=device)
                        if function_paras["requires_grad"].get(para, []) == [True]:
                            tensors[idx].requires_grad = True
                    function_paras['kwargs'][para] = tensors

    def get_name_and_data_for_grad(self, function_paras):
        inputs_for_grad_value = []
        inputs_for_grad_key = []
        for k, v in function_paras["kwargs"].items():
            if function_paras["requires_grad"].get(k, []) == [True]:
                inputs_for_grad_key.append(k)
                if isinstance(v, (list, tuple)):
                    inputs_for_grad_value.extend(v)
                else:
                    inputs_for_grad_value.append(v)
        return inputs_for_grad_key, inputs_for_grad_value

    def change_np_dtype_to_torch(self, dtype):
        if dtype == np.bool_:
            return torch.bool
        return eval(str(dtype).replace("<class 'numpy.", "torch.").replace("'>", ""))


if __name__ == '__main__':
    GenOutputData.run(os.path.join(os.path.dirname(__file__), '../cache/diopi_case_items.cfg'),
                      os.path.join(os.path.dirname(__file__), '../cache/data/inputs/'),
                      os.path.join(os.path.dirname(__file__), '../cache/data/outputs/'))
