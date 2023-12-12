import numpy as np
import conformance as cf
from conformance import Device, to_numpy_dtype
from conformance import diopi_functions as F
import torch
import unittest

class TestCat(unittest.TestCase):
    def setUp(self):
        self.shape = (1,43,25,32)
        self.N = 12
        self.idx = 1
        self.data = []
        for i in range(self.N):
            self.data.append(np.random.rand(self.shape[0],np.random.randint(1,10),  self.shape[2] ,self.shape[3]).astype(np.float32))


    def run_torch(self):
        data = []
        for d in self.data:
            data.append(torch.from_numpy(d))

        a = torch.cat(data,self.idx)
        return a.numpy()

    def run_tops(self):
        data = []
        for d in self.data:
            data.append(cf.Tensor.from_numpy(d))

        a = F.cat(data, self.idx)
        return a.numpy()

    def test_api(self):
        a = self.run_torch()
        b = self.run_tops()

        np.testing.assert_allclose(a, b,
                                           rtol=1e-5,
                                           atol=1e-5,
                                           equal_nan=False)


class TestCat1(TestCat):
    def setUp(self):
        self.shape = (1,43,25,32)
        self.N = 12
        self.idx = 3
        self.data = []
        for i in range(self.N):
            self.data.append(np.random.rand(self.shape[0], self.shape[1] ,self.shape[2],np.random.randint(1,10) ).astype(np.float32))




class TestCat2(TestCat):
    def setUp(self):
        self.shape = (1,233)
        self.N = 12
        self.idx = 0
        self.data = []
        for i in range(self.N):
            self.data.append( np.random.rand(np.random.randint(1,10),self.shape[1]  ).astype(np.float32))


if __name__ == "__main__":
    a = TestCat2()
    a.setUp()
    a.test_api()
