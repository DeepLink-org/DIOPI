import numpy as np
import conformance as cf
from conformance import Device, to_numpy_dtype
from conformance import diopi_functions as F
import torch
import unittest

class TestBatchNrom(unittest.TestCase):
    def setUp(self):
        self.C = 32
        self.x = np.random.rand(1,self.C, 64, 64).astype(np.float32)
        self.running_mean = np.random.rand(self.C).astype(np.float32)
        self.running_var = np.random.rand(self.C).astype(np.float32)
        self.weight = np.random.rand(self.C).astype(np.float32)
        self.bias = np.random.rand(self.C).astype(np.float32)
        self.momentum = 1.0
        self.eps = 1e-5
        self.bn_training = False

    def run_torch(self):
        x = torch.from_numpy(self.x)
        running_mean = torch.from_numpy(self.running_mean)
        running_var = torch.from_numpy(self.running_var)
        weight = torch.from_numpy(self.weight)
        bias = torch.from_numpy(self.bias)
        a = torch.nn.functional.batch_norm(x,running_mean,running_var,weight,bias,self.bn_training,self.momentum,self.eps)
        return a.numpy()

    def run_tops(self):

        x = cf.Tensor.from_numpy(self.x)
        running_mean = cf.Tensor.from_numpy(self.running_mean)
        running_var = cf.Tensor.from_numpy(self.running_var)
        weight = cf.Tensor.from_numpy(self.weight)
        bias = cf.Tensor.from_numpy(self.bias)
        a = F.batch_norm(x,running_mean,running_var,weight,bias,self.bn_training,self.momentum,self.eps)
        return a.numpy()

    def test_api(self):
        a = self.run_torch()
        b = self.run_tops()

        np.testing.assert_allclose(a, b,
                                           rtol=1e-5,
                                           atol=1e-5,
                                           equal_nan=False)

if __name__ == "__main__":
    a = TestBatchNrom()
    a.setUp()
    a.test_api()
