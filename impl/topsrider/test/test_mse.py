import numpy as np
import conformance as cf
from conformance import Device, to_numpy_dtype
from conformance import diopi_functions as F
import torch
import unittest

class TestMSE(unittest.TestCase):
    def setUp(self):
        self.shape = (1,32)
        self.pred = np.random.rand(*self.shape ).astype(np.float32)
        self.target = np.random.rand(*self.shape ).astype(np.float32)
        self.reduction = "none"

    def run_torch(self):
        pred = torch.from_numpy(self.pred)
        target = torch.from_numpy(self.target)

        a = torch.nn.functional.mse_loss(pred,target,reduction=self.reduction)
        return a.numpy()

    def run_tops(self):

        pred = cf.Tensor.from_numpy(self.pred)
        target = cf.Tensor.from_numpy(self.target)

        a = F.mse_loss(pred,target,reduction=self.reduction)
        return a.numpy()

    def test_api(self):
        a = self.run_torch()
        b = self.run_tops()

        np.testing.assert_allclose(a, b,
                                           rtol=1e-5,
                                           atol=1e-5,
                                           equal_nan=False)



class TestMSE1(TestMSE):
    def setUp(self):
        self.shape = (4,1)
        self.pred = np.ones(self.shape ).astype(np.float32) *0.5
        self.target = np.ones(self.shape ).astype(np.float32) *0.5
        self.reduction = "mean"


class TestMSE2(TestMSE):
    def setUp(self):
        self.shape = (4,1)
        self.pred = np.ones(self.shape ).astype(np.float32) *0.5
        self.target = np.ones(self.shape ).astype(np.float32) *0.5
        self.reduction = "sum"


if __name__ == "__main__":
    a = TestMSE()
    a.setUp()
    a.test_api()
