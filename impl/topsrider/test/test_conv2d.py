import numpy as np
import conformance as cf
from conformance import Device, to_numpy_dtype,Dtype
from conformance import diopi_functions as F
import torch
import unittest
import time

class TestConv2d(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(1,32, 64, 64).astype(np.float32)#np.random.rand(1,1, 3, 3).astype(np.float32)
        self.w = np.random.rand(64,32,3,3).astype(np.float32)#np.random.rand(2,1,1,1).astype(np.float32)


    def run_torch(self):
        x = torch.from_numpy(self.x)
        w = torch.from_numpy(self.w)
        a = torch.nn.functional.conv2d(x,w,padding=1)
        return a.numpy()

    def run_tops(self):
        x = cf.Tensor.from_numpy(self.x)
        w = cf.Tensor.from_numpy(self.w)
        a = F.conv2d(x,w,padding=[1,1])
        return a.numpy()

    def test_api(self):
        a = self.run_torch()
        b = self.run_tops()
        np.testing.assert_allclose(a, b,
                                           rtol=1e-5,
                                           atol=1e-5,
                                           equal_nan=False)


class TestConv2d_bias(unittest.TestCase):
    def setUp(self):

        # self.param = dict(
        #             name=["conv2d"],
        #             atol=1e-3,
        #             rtol=1e-3,
        #             dtype=[Dtype.float32],
        #             para=dict(
        #                 stride=[2, 1, 1, (2, 2)],
        #                 padding=[0, 12, 0, (0, 0)],
        #                 dilation=[1, 12, 1, (1, 1)],
        #                 groups=[1, 2048, 1, 1],
        #             ),
        #             tensor_para=dict(
        #                 args=[
        #                     {
        #                         "ins": ["input"],
        #                         "shape": ((2, 256, 200, 304), (2, 2048, 64, 64), (2, 2048, 1, 1), (2, 256, 200, 304)),
        #                     },
        #                     {
        #                         "ins": ["weight"],
        #                         "shape": ((12, 256, 1, 1), (2048, 1, 3, 3), (512, 2048, 1, 1), (12, 256, 1, 1)),
        #                     },
        #                     {
        #                         "ins": ["bias"],
        #                         "shape": ((12, ), None, None, (12, )),
        #                     },
        #                 ]
        #             ),
        #         )
        self.x = np.random.rand(1, 1232, 64, 64).astype(np.float32)#np.random.rand(1,1, 3, 3).astype(np.float32)
        self.w = np.random.rand(13, 1232, 3, 3).astype(np.float32)#np.random.rand(2,1,1,1).astype(np.float32)
        self.bias = np.random.rand(13).astype(np.float32)

    def run_torch(self):

        x = torch.from_numpy(self.x)
        w = torch.from_numpy(self.w)
        b = torch.from_numpy(self.bias) #,padding=0,bias=None,stride=2,dilation=1,groups=1
        a = torch.nn.functional.conv2d(x,w,bias=b,padding=0,stride=2,dilation=1,groups=1)

        return a.numpy()

    def run_tops(self):
        x = cf.Tensor.from_numpy(self.x)
        w = cf.Tensor.from_numpy(self.w)
        b = cf.Tensor.from_numpy(self.bias)
        a = F.conv2d(x,w,padding=0,bias=b,stride=2,dilation=1,groups=1)

        return a.numpy()

    def test_api(self):
        b = self.run_tops()
        a = self.run_torch()

        np.testing.assert_allclose(a, b,
                                           rtol=1e-5,
                                           atol=1e-5,
                                           equal_nan=False)


if __name__ == "__main__":
    a = TestConv2d_bias()
    a.setUp()
    a.test_api()
    a.test_api()
