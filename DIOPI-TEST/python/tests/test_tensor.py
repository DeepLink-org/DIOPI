import numpy as np
import conformance as cf
from conformance import to_numpy_dtype


class TestTensor(object):

    def test_construct(self):
        for dtype in [cf.float32, cf.float64, cf.int32, cf.int16, cf.int8, cf.bool]:
            tensor = cf.Tensor(size=(2, 3, 5), dtype=dtype)
            tensor.fill_(10)

            array = np.ndarray(shape=(2, 3, 5), dtype=to_numpy_dtype(dtype))
            array.fill(10)

            assert np.allclose(array, tensor.numpy())
