import numpy as np
import conformance as cf
from conformance import Device, to_numpy_dtype


class TestTensor(object):
    dtypes = [cf.float32, cf.float64, cf.int32, cf.int16, cf.int8, cf.bool]

    def test_construct(self):
        for dtype in self.dtypes:
            tensor = cf.Tensor(size=(2, 3, 5), dtype=dtype)
            assert tensor.numel() == 2*3*5
            assert tensor.size() == (2, 3, 5)
            assert tensor.get_dtype() == dtype
            assert tensor.get_device() == Device.AIChip

        for dtype in self.dtypes:
            array = np.ndarray(shape=(2, 3, 5), dtype=to_numpy_dtype(dtype))
            array.fill(10)
            tensor = cf.Tensor.from_numpy(array)

            array = np.ndarray(shape=(2, 3, 5), dtype=to_numpy_dtype(dtype))
            array.fill(10)

            assert np.allclose(array, tensor.numpy())
