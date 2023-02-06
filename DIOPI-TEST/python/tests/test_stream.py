import numpy as np
import time
from threading import Thread
from conformance.diopi_runtime import Context, Tensor, Sizes
from conformance.utils import check_function, logger
from ctypes import c_int32


class TestStream(object):
    # To do stream tests, the following workflow is used:
    # begin = time.time()
    # for i in range(nums):
    #   y = mat1 @ mat2
    #   mat1 = y
    # Using Tensor.numpy() to sync stream, 'sum' is helpful to reduce the cost of memcpy
    # res = sum(mat1)
    # res_ndarray = Tensor.numpy(res)
    # end = time.time()
    context = Context()
    context1 = Context()
    stream = context.get_handle()
    stream1 = context1.get_handle()
    nums = 10
    bmm_func = check_function("diopiMatmul")
    sum_func = check_function("diopiSum")

    @classmethod
    def setup_class(self):
        # generate numpy data
        mat1_shape = (2, 32, 1024)
        mat2_shape = (2, 1024, 1024)
        self.mat1_ndarray = np.random.randn(*mat1_shape).astype(np.float32)
        self.mat2_ndarray = np.random.randn(*mat2_shape).astype(np.float32)

        out_ndarray = np.copy(self.mat1_ndarray)
        for i in range(self.nums):
            out_ndarray = np.matmul(out_ndarray, self.mat2_ndarray)
        self.out_ref_ndarry = np.sum(out_ndarray)

    def gen_device_data(self, stream):
        # from_numpy call cudaMalloc which can not be concurrent with other missions on stream
        mat1_tensor = Tensor.from_numpy(self.mat1_ndarray, context_handle=stream)
        mat2_tensor = Tensor.from_numpy(self.mat2_ndarray, context_handle=stream)
        out_tensor = Tensor.raw_like(mat1_tensor)
        res_tensor = Tensor([], mat1_tensor.get_dtype(), context_handle=stream)
        return mat1_tensor, mat2_tensor, out_tensor, res_tensor

    def call_func(self, stream):
        mat1, mat2, out, res = self.gen_device_data(stream)
        # Allocate all the device memory in advance,
        # so we can assure that stream will not be interrupted by device api like xxxmalloc()
        begin = time.time()
        for i in range(self.nums):
            self.bmm_func(stream, out.tensor_handle, mat1.tensor_handle, mat2.tensor_handle)
            tmp = out
            out = mat1
            mat1 = tmp

        dim = Sizes((0, 1, 2))
        dtype = res.get_dtype()
        self.sum_func(stream, res.tensor_handle, mat1.tensor_handle, dim, c_int32(dtype.value))
        out_ndarray = Tensor.numpy(res)
        end = time.time()

        assert np.allclose(out_ndarray, self.out_ref_ndarry, 1e-2, 1e-1, True)
        return end - begin

    def test_stream(self):
        # warm up
        cost = self.call_func(self.stream)
        logger.info(f"warming-up costs: {cost}s")

    def test_multi_stream(self):
        mat1, mat2, out, res = self.gen_device_data(self.stream)
        mat1_s1, mat2_s1, out_s1, res_s1 = self.gen_device_data(self.stream1)

        baseline = self.call_func(self.stream)

        begin = time.time()
        for i in range(self.nums):
            self.bmm_func(self.stream, out.tensor_handle, mat1.tensor_handle, mat2.tensor_handle)
            self.bmm_func(self.stream1, out_s1.tensor_handle, mat1_s1.tensor_handle, mat2_s1.tensor_handle)
            tmp = out
            tmp_s1 = out_s1
            out = mat1
            out_s1 = mat1_s1
            mat1 = tmp
            mat1_s1 = tmp_s1

        dim1 = Sizes((0, 1, 2))
        dtype = res.get_dtype()
        self.sum_func(self.stream, res.tensor_handle, mat1.tensor_handle, dim1, c_int32(dtype.value))
        self.sum_func(self.stream1, res_s1.tensor_handle, mat1_s1.tensor_handle, dim1, c_int32(dtype.value))
        out_ndarray = Tensor.numpy(res)
        out_s1_ndarray = Tensor.numpy(res_s1)
        end = time.time()

        logger.info(f"after warming-up, one stream costs: {baseline}s, two streams costs: {end - begin}s")
        assert (end - begin) < 1.8 * baseline, "don't improve 20% performance by concurrent stream"
        assert np.allclose(out_ndarray, self.out_ref_ndarry, 1e-2, 1e-1, True)
        assert np.allclose(out_s1_ndarray, self.out_ref_ndarry, 1e-2, 1e-1, True)

    def test_multi_thread_multi_stream(self):
        thread_1 = Thread(target=self.call_func, args=(self.stream, ))
        thread_2 = Thread(target=self.call_func, args=(self.stream1, ))
        thread_1.start()
        thread_2.start()
        thread_1.join()
        thread_2.join()
