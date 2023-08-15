# Copyright (c) 2023, DeepLink.

import numpy as np

class Genfunc:
    # func = ['randn', 'rand']
    # def __init__(self) -> None:
    #     pass

    def rand(shape, dtype=np.float32):
        return np.random.rand(*shape).astype(dtype)
    
    def randn(shape, dtype=np.float32):
        return np.random.randn(*shape).astype(dtype)
    
    def uniform(low=0, high=1, shape=(1,), dtype=np.float32):
        return np.random.uniform(low=low, high=high, size=shape).astype(dtype)
    
    def empty(shape, dtype=np.float32):
        return np.empty(shape).astype(dtype)
    
    def ones(shape, dtype=np.float32):
        return np.ones(shape).astype(dtype)
    
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape=shape).astypde(dtype)
    
    def mask(shape, dtype=np.float32):
        return np.random.uniform(low=0, high=2, size=shape).astype(dtype)
    
    def randint(low=0, high=1, shape=(1,), dtype=np.float32):
        return np.random.randint(low=low, high=high, size=shape).astype(dtype)
    
    def positive(shape, dtype=np.float32):
        return np.abs(np.random.randn(*shape)).astype(dtype)
    
    def sym_mat(shape, dtype=np.float32):
        axis = [i for i in range(len(shape) - 2)] + [-1, -2]
        mat = np.random.randn(*shape).astype(dtype)
        return mat @ mat.traspose(axis) + 1e-3
    
    # def real(low=0, high=1, shape=(1,), dtype=np.float32):
    def rand_int(low=0, high=1, shape=(1,), dtype=np.float32):
        if np.dtype(dtype) in ['i', 'u']:
            return np.random.randint(low=low, high=high, size=shape).astype(dtype)
        else:
            return np.random.randn(*shape).astype(dtype)
        
    def randn_complx(shape, dtype=np.float32):
        return np.array(np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)

# only for Genfunc test
if __name__ == '__main__':
    func = Genfunc.rand
    ret = func((3,4), dtype=np.float16)
    print(ret.dtype)
    print(eval(f"Genfunc.randn((4, 4), dtype=np.float16)"))
