
import numpy as np
import conformance as cf
from conformance import Device, to_numpy_dtype
from conformance import diopi_functions as F
import torch
import unittest

class TestAct(unittest.TestCase):
    def setUp(self):
        self.A = np.random.randn(4,1,3,5).astype(np.float32)

    def run_torch(self):
        tensor0 = torch.from_numpy(self.A)

        a = torch.relu_(tensor0)
        return a.numpy()

    def run_tops(self):
        tensor0 = cf.Tensor.from_numpy(self.A)
        a = F.relu(tensor0,inplace=True )
        return a.numpy()

    def test_api(self):
        a = self.run_torch()
        b = self.run_tops()
        np.testing.assert_allclose(a, b,
                                           rtol=1e-5,
                                           atol=1e-5,
                                           equal_nan=False)




class TestLeakyRelu(unittest.TestCase):
    def setUp(self):
        self.A = np.random.randn(1,214,5,1).astype(np.float32)

    def run_torch(self):
        tensor0 = torch.from_numpy(self.A)

        a = torch.nn.functional.leaky_relu(tensor0,0.1,inplace=False)
        return a.numpy()

    def run_tops(self):
        tensor0 = cf.Tensor.from_numpy(self.A)
        a = F.leaky_relu(tensor0,0.1,inplace=True )
        return a.numpy()

    def test_api(self):
        a = self.run_torch()
        b = self.run_tops()
        np.testing.assert_allclose(a, b,
                                           rtol=1e-5,
                                           atol=1e-5,
                                           equal_nan=False)



if __name__ == "__main__":
    a = TestLeakyRelu()
    a.setUp()
    a.test_api()
