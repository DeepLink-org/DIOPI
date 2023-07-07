import numpy as np
import conformance as cf
from conformance import Device, to_numpy_dtype
from conformance import diopi_functions as F
import torch
import unittest

class TestEmbedding(unittest.TestCase):
    def setUp(self):

        self.index = np.random.randint(0,127, size=[12,146])
        print(self.index)
        self.weight = np.random.rand(128,8 ).astype(np.float32)
        self.reduction = "none"

    def run_torch(self):
        index = torch.from_numpy(self.index)
        weight = torch.from_numpy(self.weight)

        a = torch.nn.functional.embedding(index, weight)
        return a.numpy()

    def run_tops(self):

        index = cf.Tensor.from_numpy(self.index.astype(np.int32))
        weight = cf.Tensor.from_numpy(self.weight)

        a = F.embedding(index,weight)
        return a.numpy()

    def test_api(self):
        a = self.run_torch()
        b = self.run_tops()
        np.testing.assert_allclose(a, b,
                                           rtol=1e-5,
                                           atol=1e-5,
                                           equal_nan=False)





if __name__ == "__main__":
    a = TestEmbedding()
    a.setUp()
    a.test_api()


# binary_cross_entropy
