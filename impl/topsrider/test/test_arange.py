import numpy as np
import conformance as cf
from conformance import Device, to_numpy_dtype
from conformance import diopi_functions as F
import torch
import unittest

class TestArange(unittest.TestCase):
    def setUp(self):
        self.start = 0
        self.end = 10
        self.step = 1

    def run_torch(self):

        a = torch.arange(self.start,self.end, self.step)
        return a.numpy()

    def run_tops(self):

        a = F.arange(end=self.end,start=self.start, step=self.step)
        return a.numpy()

    def test_api(self):
        a = self.run_torch()
        b = self.run_tops()
        np.testing.assert_allclose(a, b,
                                           rtol=1e-5,
                                           atol=1e-5,
                                           equal_nan=False)


class TestArange1(TestArange):
    def setUp(self):
        self.start = 0
        self.end = 10
        self.step = 0.5



if __name__ == "__main__":
    a = TestArange1()
    a.setUp()
    a.test_api()
