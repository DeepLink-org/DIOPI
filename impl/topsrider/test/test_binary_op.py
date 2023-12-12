import numpy as np
import conformance as cf
from conformance import Device, to_numpy_dtype
from conformance import diopi_functions as F
import torch
import unittest

class TestBinaryOp(unittest.TestCase):
    def setUp(self):
        self.A = np.random.randn(4,2,3,4).astype(np.float32)
        self.B = np.random.randn(4,2,3,4).astype(np.float32)

    def run_torch(self):
        tensor0 = torch.from_numpy(self.A)
        tensor1 = torch.from_numpy(self.B)
        a0 = torch.add(tensor0,tensor1)
        a1 = torch.sub(tensor0,tensor1)
        a2 = torch.mul(tensor0,tensor1)
        a3 = torch.div(tensor0,tensor1)
        return a0.numpy(),a1.numpy(),a2.numpy(),a3.numpy()

    def run_tops(self):
        tensor0 = cf.Tensor.from_numpy(self.A)
        tensor1 = cf.Tensor.from_numpy(self.B)
        a0 = F.add(tensor0,tensor1)
        a1 = F.sub(tensor0,tensor1)
        a2 = F.mul(tensor0,tensor1)
        a3 = F.div(tensor0,tensor1)
        return a0.numpy(),a1.numpy(),a2.numpy(),a3.numpy()

    def test_api(self):
        aa = self.run_torch()
        bb = self.run_tops()
        assert(len(aa) == len(bb))
        for a, b in zip(aa,bb):

            np.testing.assert_allclose(a, b,
                                            rtol=1e-5,
                                            atol=1e-5,
                                            equal_nan=False)



class TestAdd1(TestBinaryOp):
    def setUp(self):
        self.A = np.random.randn(4,1,3,5).astype(np.float32)
        self.B = np.random.randn(1).astype(np.float32)


#todo: we need topsOpConvert to handle the scalar tpyes.
# class TestAdd2(unittest.TestCase):
#     def setUp(self):
#         self.A = np.random.randn(4,1,3,5).astype(np.float32)
#         self.B = np.random.randn(4,1,3,5).astype(np.float32)

#     def run_torch(self):
#         tensor0 = torch.from_numpy(self.A)
#         tensor1 = torch.from_numpy(self.B)
#         a = torch.add(tensor0,1.0)
#         return a.numpy()

#     def run_tops(self):
#         tensor0 = cf.Tensor.from_numpy(self.A)
#         tensor1 = cf.Tensor.from_numpy(self.B)
#         a =F.add(tensor0,float(1.0) )
#         return a.numpy()

#     def test_api(self):
#         a = self.run_torch()
#         b = self.run_tops()
#         np.testing.assert_allclose(a, b,
#                                            rtol=1e-5,
#                                            atol=1e-5,
#                                            equal_nan=False)
#         print(a,b)



if __name__ == "__main__":
    a = TestAdd2()
    a.setUp()
    a.test_api()
