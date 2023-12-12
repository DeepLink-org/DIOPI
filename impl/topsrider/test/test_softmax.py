import numpy as np
import conformance as cf
from conformance import Device, to_numpy_dtype
from conformance import diopi_functions as F
import torch
import unittest

class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.array = np.ndarray(shape=(5,3, 4, 5), dtype=to_numpy_dtype(cf.float32))
        self.array.fill(10)

    def run_torch(self):
        tensor = torch.from_numpy(self.array)
        a = torch.nn.functional.softmax(tensor,3)
        return a.numpy()

    def run_tops(self):
        tensor = cf.Tensor.from_numpy(self.array)
        a = F.softmax(tensor,3, cf.float32)
        return a.numpy()

    def test_api(self):
        a = self.run_torch()
        b = self.run_tops()
        np.testing.assert_allclose(a, b,
                                           rtol=1e-5,
                                           atol=1e-5,
                                           equal_nan=False)


if __name__ == "__main__":
    a = TestSoftmax()
    a.setUp()
    a.test_api()
