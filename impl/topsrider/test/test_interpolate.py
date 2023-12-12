import numpy as np
import conformance as cf
from conformance import Device, to_numpy_dtype
from conformance import diopi_functions as F
import torch
import unittest

class TestInterpolate(unittest.TestCase):
    def setUp(self):
        self.shape = (4,1,5,5)
        self.img = np.random.rand(*self.shape ).astype(np.float32)
        self.size = (10,10)
        self.mode = "nearest"

    def run_torch(self):
        img = torch.from_numpy(self.img)


        a = torch.nn.functional.interpolate(img, self.size,mode=self.mode,align_corners=None)
        return a.numpy()

    def run_tops(self):

        img = cf.Tensor.from_numpy(self.img)


        a = F.interpolate(img,size=self.size, scale_factor=None, mode=self.mode, align_corners=None)
        return a.numpy()

    def test_api(self):
        a = self.run_torch()
        b = self.run_tops()

        np.testing.assert_allclose(a, b,
                                           rtol=1e-5,
                                           atol=1e-5,
                                           equal_nan=False)



# class TestInterpolate1(TestInterpolate):
#     def setUp(self):
#         self.shape = (4,256,54,54)
#         self.img = np.random.rand(*self.shape ).astype(np.float32)
#         self.size = (256,256)
#         self.mode = "nearest"


class TestInterpolate2(TestInterpolate):
    def setUp(self):
        self.shape = (4,1,5,5)
        self.img = np.random.rand(*self.shape ).astype(np.float32)
        self.size = (10,10)
        self.mode = "bilinear"

if __name__ == "__main__":
    a = TestInterpolate2()
    a.setUp()
    a.test_api()
