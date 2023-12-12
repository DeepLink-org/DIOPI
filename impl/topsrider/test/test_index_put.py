import numpy as np
import conformance as cf
from conformance import Device, to_numpy_dtype
from conformance import diopi_functions as F
import torch
import unittest

class TestIndexput(unittest.TestCase):
    def setUp(self):
        self.index = []
        self.value = []
        self.N = 16
        self.input = np.random.rand(1,32, 64, 64).astype(np.float32)
        for i in range(len(self.input.shape)):
            ii = []
            for j in range(self.N):
                ii.append(np.random.randint(0,self.input.shape[i]))
            # self.index.append(np.array([
            #     np.random.randint(0,1),
            #     np.random.randint(0,32),
            #     np.random.randint(0,64),
            #     np.random.randint(0,64),
            # ]))
            self.index.append(np.array(ii))
        for j in range(self.N):
            self.value.append(np.random.rand())
        self.value = np.array(self.value).astype(np.float32)



    def run_torch(self):
        index = []
        for i in self.index:
            index.append(torch.from_numpy(i))

        a = torch.index_put(torch.from_numpy(self.input), index,torch.from_numpy(self.value) )
        return a.numpy()

    def run_tops(self):

        index = []
        for i in self.index:
            index.append(cf.Tensor.from_numpy(i))

        a = F.index_put(cf.Tensor.from_numpy(self.input),cf.Tensor.from_numpy(self.value), index)
        return a.numpy()

    def test_api(self):
        a = self.run_torch()
        b = self.run_tops()
        np.testing.assert_allclose(a, b,
                                           rtol=1e-5,
                                           atol=1e-5,
                                           equal_nan=False)





if __name__ == "__main__":
    a = TestIndexput()
    a.setUp()
    a.test_api()


# binary_cross_entropy
