# Copyright (c) 2023, DeepLink.
import numpy as np
import scipy as sp

class Genfunc:
    @staticmethod
    def rand(shape, dtype=np.float32):
        if shape == ():
            return np.array(np.random.rand()).astype(dtype)
        return np.array(np.random.rand(*shape)).astype(dtype)

    @staticmethod
    def randn(shape, dtype=np.float32):
        if shape == ():
            return np.array(np.random.randn()).astype(dtype)
        return np.array(np.random.randn(*shape)).astype(dtype)

    @staticmethod
    def uniform(low=0, high=1, shape=(1,), dtype=np.float32):
        return np.random.uniform(low=low, high=high, size=shape).astype(dtype)

    @staticmethod
    def empty(shape, dtype=np.float32):
        return np.empty(shape).astype(dtype)

    @staticmethod
    def ones(shape, dtype=np.float32):
        return np.ones(shape).astype(dtype)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape=shape).astype(dtype)

    @staticmethod
    def mask(shape, dtype=np.float32):
        return np.random.uniform(low=0, high=2, size=shape).astype(dtype)

    @staticmethod
    def randint(low=0, high=1, shape=(1,), dtype=np.float32):
        return np.random.randint(low=low, high=high, size=shape).astype(dtype)

    @staticmethod
    def positive(shape, dtype=np.float32):
        return np.array(np.abs(np.random.randn(*shape)).astype(dtype))

    @staticmethod
    def sym_mat(shape, dtype=np.float32):
        axis = [i for i in range(len(shape) - 2)] + [-1, -2]
        mat = np.random.randn(*shape).astype(dtype)
        return mat @ mat.transpose(axis) + 1e-3

    # def real(low=0, high=1, shape=(1,), dtype=np.float32):
    @staticmethod
    def randn_int(low=0, high=1, shape=(1,), dtype=np.float32):
        if np.dtype(dtype) in ['i', 'u']:
            return np.random.randint(low=low, high=high, size=shape).astype(dtype)
        else:
            return np.array(np.random.randn(*shape)).astype(dtype)

    @staticmethod
    def randn_complx(shape, dtype=np.float32):
        return np.array(np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
    
    @staticmethod
    def log_softmax(shape, dim, dtype=np.float32):
        val = np.array(np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)
        return sp.special.log_softmax(val, dim)

# only for Genfunc test
if __name__ == '__main__':
    func = Genfunc.rand
    ret = func((3,4), dtype=np.float16)
    print(ret.dtype)
    print(eval(f"Genfunc.randn((4, 4), dtype=np.float16)"))
