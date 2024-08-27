# Copyright (c) 2023, DeepLink.
import numpy as np
import atexit
import ctypes
from ctypes import c_void_p

from diopilib import (
    diopiTensor,
    diopiSparseCsrTensor,
    diopiSize,
    diopiScalar,
    diopiReduction,
    diopiRoundMode,
    diopiError,
    TensorP,
    Context,
    Device,
    Dtype,
    diopi_tensor_copy_to_buffer,
    get_last_error_string,
    finalize_library,
    diopi_finalize,
    init_library,
    diopiGenerator,
)


def device(dev: str) -> Device:
    if dev == "cpu" or dev == "host":
        return Device.Host
    else:
        return Device.AIChip


all_types = [Dtype.float16, Dtype.float32, Dtype.float64, Dtype.int32, Dtype.int64]
float_types = [Dtype.float16, Dtype.float32, Dtype.float64]
float_no_half_types = [Dtype.float32, Dtype.float64]
int_types = [Dtype.int32, Dtype.int64, Dtype.int16, Dtype.int8, Dtype.uint8]
complex_types = [Dtype.complex64, Dtype.complex128]
default = all_types


def from_dtype_str(dtype_str: str) -> Dtype:
    if dtype_str == "int8":
        return Dtype.int8
    elif dtype_str == "uint8":
        return Dtype.uint8
    elif dtype_str == "int16":
        return Dtype.int16
    elif dtype_str == "uint16":
        return Dtype.uint16
    elif dtype_str == "int32":
        return Dtype.int32
    elif dtype_str == "uint32":
        return Dtype.uint32
    elif dtype_str == "int64":
        return Dtype.int64
    elif dtype_str == "uint64":
        return Dtype.uint64
    elif dtype_str == "float16":
        return Dtype.float16
    elif dtype_str == "float32":
        return Dtype.float32
    elif dtype_str == "float64":
        return Dtype.float64
    elif dtype_str == "bool":
        return Dtype.bool
    else:
        return None


def from_numpy_dtype(dtype: np.dtype) -> Dtype:
    if dtype == np.int8:
        return Dtype.int8
    elif dtype == np.int16:
        return Dtype.int16
    elif dtype == np.int32:
        return Dtype.int32
    elif dtype == np.int64:
        return Dtype.int64
    elif dtype == np.uint8:
        return Dtype.uint8
    elif dtype == np.uint16:
        return Dtype.uint16
    elif dtype == np.uint32:
        return Dtype.uint32
    elif dtype == np.uint64:
        return Dtype.uint64
    elif dtype == np.float16:
        return Dtype.float16
    elif dtype == np.float32:
        return Dtype.float32
    elif dtype == np.float64:
        return Dtype.float64
    elif dtype == np.bool_:
        return Dtype.bool
    elif dtype == np.complex64:
        return Dtype.complex64
    elif dtype == np.complex128:
        return Dtype.complex128
    else:
        return None


def to_numpy_dtype(dtype: Dtype) -> np.dtype:
    if dtype == Dtype.int8:
        return np.int8
    elif dtype == Dtype.int16:
        return np.int16
    elif dtype == Dtype.int32:
        return np.int32
    elif dtype == Dtype.int64:
        return np.int64
    elif dtype == Dtype.uint8:
        return np.uint8
    elif dtype == Dtype.uint16:
        return np.uint16
    elif dtype == Dtype.uint32:
        return np.uint32
    elif dtype == Dtype.uint64:
        return np.uint64
    elif dtype == Dtype.float16:
        return np.float16
    elif dtype == Dtype.float32:
        return np.float32
    elif dtype == Dtype.float64:
        return np.float64
    elif dtype == Dtype.bool:
        return np.bool_
    elif dtype == Dtype.complex64:
        return np.complex64
    elif dtype == Dtype.complex128:
        return np.complex128
    else:
        return None


def is_dtype(dtype) -> bool:
    return isinstance(dtype, Dtype)


def compute_nhwc_stride_2d(sizes, itemsize=1):
    dim = len(sizes)
    strides = [itemsize for i in range(dim)]
    assert dim == 3 or dim == 4, "not supported dim"
    if dim == 3:
        strides[0] = itemsize
        strides[2] = strides[0] * sizes[0]
        strides[1] = strides[2] * sizes[2]
    elif dim == 4:
        strides[1] = itemsize
        strides[3] = strides[0] * sizes[1]
        strides[2] = strides[3] * sizes[3]
        strides[0] = strides[2] * sizes[2]
    return strides


def compute_nhwc_stride_3d(sizes, itemsize=1):
    dim = len(sizes)
    strides = [itemsize for i in range(dim)]
    assert dim == 4 or dim == 5, "not supported dim"
    if dim == 4:
        strides[0] = itemsize
        strides[3] = strides[0] * sizes[0]
        strides[2] = strides[3] * sizes[3]
        strides[1] = strides[2] * sizes[2]
    elif dim == 5:
        strides[1] = itemsize
        strides[4] = strides[0] * sizes[1]
        strides[3] = strides[4] * sizes[4]
        strides[2] = strides[3] * sizes[3]
        strides[0] = strides[2] * sizes[2]
    return strides


def compute_nhwc_stride(size, itemsize=1, name=None):
    if name == "2d":
        return compute_nhwc_stride_2d(size, itemsize)
    if name == "3d":
        return compute_nhwc_stride_3d(size, itemsize)

    dim = len(size)
    if dim < 5:
        return compute_nhwc_stride_2d(size, itemsize)
    else:
        return compute_nhwc_stride_3d(size, itemsize)


def set_nhwc(tensor_nchw, name_2d_3d):
    ndim = tensor_nchw.ndim
    if ndim == 3:
        axis = (1, 2, 0)
    elif ndim == 4 and name_2d_3d == '3d':
        axis = (1, 2, 3, 0)
    elif ndim == 4:
        axis = (0, 2, 3, 1)
    elif ndim == 5:
        axis = (0, 2, 3, 4, 1)
    nhwc_out = np.transpose(tensor_nchw, axis).copy()
    nhwc_out.shape = tensor_nchw.shape
    nhwc_out.strides = compute_nhwc_stride(tensor_nchw.shape, tensor_nchw.itemsize, name_2d_3d)
    return nhwc_out


def diopi_rt_init():
    init_library()


def on_diopi_rt_exit():
    finalize_library()
    diopi_finalize()


atexit.register(on_diopi_rt_exit)


def get_last_error():
    last_error_str = get_last_error_string()
    return last_error_str


default_context = Context()


class Sizes(diopiSize):
    def __init__(self, shape=()):
        super(Sizes, self).__init__(list(shape) if shape else None, len(shape))
        self.shape = self.data


class Scalar(diopiScalar):
    def __init__(self, value, dtype=None):
        from conformance.global_settings import glob_vars

        if dtype is None:
            dtype = (
                from_numpy_dtype(glob_vars.int_type) if isinstance(value, int) else from_numpy_dtype(glob_vars.float_type)
            )
        diopiScalar.__init__(self, dtype, value)


class Tensor(diopiTensor):
    def __init__(
        self,
        size=None,
        dtype=None,
        stride=None,
        context=default_context,
        data_ptr=None,
        device=Device.AIChip,
        requires_grad=False,
    ):
        if size is None:
            return diopiTensor.__init__(self)

        if isinstance(size, (tuple, list)):
            size = Sizes(list(size))

        if isinstance(stride, (tuple, list)):
            stride = Sizes(list(stride))

        if data_ptr is None:
            diopiTensor.__init__(self, size, stride, dtype, device, context, requires_grad=requires_grad)
        else:
            diopiTensor.__init__(self, size, stride, dtype, device, context, data_ptr, requires_grad=requires_grad)

    def __str__(self):
        # array = self.numpy()
        # string = f"{array.__str__()}\n"
        string = ''
        string += f"{self.get_dtype()}, shape:{self.size().data},\
                     stride:{self.get_stride().data}, numel:{self.numel()}\n"
        return string

    def __repr__(self):
        return self.__str__()

    def raw_like(self):
        size = self.size()
        stride = self.get_stride()
        dtype = self.get_dtype()
        return Tensor(
            size=size,
            dtype=dtype,
            stride=stride,
            context=self.context(),
            device=self.get_device(),
            requires_grad=self.requires_grad,
        )

    def size(self):
        return self.shape()

    def dtype(self):
        return self.get_dtype()

    def reset_shape(self, shape):
        assert isinstance(shape, (tuple, list))
        self.reset_shape(Sizes(list(shape)))

    @classmethod
    def from_numpy(
            cls, 
            darray, 
            context=None, 
            device=Device.AIChip, 
            is_sparse=False, 
            sparse_format=None,
            requires_grad=False,
        ):
        if not isinstance(darray, (np.generic, np.ndarray)):
            raise TypeError(f"expected np.ndarray (got {type(darray)})")
        dtype = from_numpy_dtype(darray.dtype)
        stride = [
            int(darray.strides[i] / darray.itemsize) for i in range(len(darray.strides))
        ]

        size = Sizes(list(darray.shape))
        stride = Sizes(list(stride))
        PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
        PyCapsule_New = ctypes.pythonapi.PyCapsule_New
        PyCapsule_New.restype = ctypes.py_object
        PyCapsule_New.argtypes = (
            ctypes.c_void_p,
            ctypes.c_char_p,
            PyCapsule_Destructor,
        )
        capsule = PyCapsule_New(
            c_void_p(darray.ctypes.data), None, PyCapsule_Destructor(0)
        )

        if is_sparse:
            if sparse_format == 'csr':
                return cls.create_sparse_csr_tensor(
                    dense_data=darray, 
                    dtype=dtype, 
                    size=size, 
                    device=device, 
                    context=context if context else default_context
                )
            else:
                raise NotImplementedError(f"Sparse format {sparse_format} is not supported")
        
        return cls(
            size=size,
            dtype=dtype,
            stride=stride,
            data_ptr=capsule,
            context=context if context else default_context,
            device=device,
            requires_grad=requires_grad,
        )

    def numpy(self) -> np.ndarray:
        if all(x == 0 for x in self.size().data) and self.numel() == 0:
            # cases when shape all 0, but not include the scalar tensor
            return np.empty(self.size().data, to_numpy_dtype(self.get_dtype()))
        data = np.empty((1,), to_numpy_dtype(self.get_dtype()))
        element_size = data.itemsize
        stride_scaled = [int(stride * element_size) for stride in self.get_stride().data]
        sum_of_products = sum((s - 1) * st for s, st in zip(self.size().data, stride_scaled))
        sumsize = int(sum_of_products / element_size) + 1
        darray = np.empty(sumsize, to_numpy_dtype(self.get_dtype()))
        PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
        PyCapsule_New = ctypes.pythonapi.PyCapsule_New
        PyCapsule_New.restype = ctypes.py_object
        PyCapsule_New.argtypes = (
            ctypes.c_void_p,
            ctypes.c_char_p,
            PyCapsule_Destructor,
        )
        capsule = PyCapsule_New(
            c_void_p(darray.ctypes.data), None, PyCapsule_Destructor(0)
        )
        diopi_tensor_copy_to_buffer(self.context(), self, capsule)
        strides = [int(stride * darray.itemsize) for stride in self.get_stride().data]
        darray = np.lib.stride_tricks.as_strided(
            darray, shape=list(self.size().data), strides=strides
        )
        return darray

    def to_sparse_csr(self):
        dense_data = self.numpy()

        crow_indices = [0]
        col_indices = []
        values = []

        for i, row in enumerate(dense_data):
            row_nnz = 0
            for j, value in enumerate(row):
                if value != 0:
                    col_indices.append(j)
                    values.append(value)
                    row_nnz += 1
            crow_indices.append(crow_indices[-1] + row_nnz)

        crow_indices = Tensor.from_numpy(np.array(crow_indices, dtype=np.int32))
        col_indices = Tensor.from_numpy(np.array(col_indices, dtype=np.int32))
        values = Tensor.from_numpy(np.array(values, dtype=to_numpy_dtype(self.dtype())))

        sparse_tensor = SparseCsrTensor(
            size=self.size(),
            dtype=self.dtype(),
            context=self.context(),
            device=self.get_device(),
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
        )
        return sparse_tensor

    @classmethod
    def create_sparse_csr_tensor(cls, dense_data, dtype, size, device, context):
        """ 
        Helper method to create a SparseCsrTensor from dense 2D NumPy array.
        
        TODO: The current implementation supports only 2D sparse tensors. 
              If multi-dimensional sparse tensor creation is required, 
              additional implementation will be needed in the future.
        
        The conversion process involves:
         1. Iterate through each row of the dense array:
            - For each non-zero element, record its column index in `col_indices` and its value in `values`.
            - Track the number of non-zero elements per row and append the cumulative count to `crow_indices`.
         2. Convert `crow_indices`, `col_indices`, and `values` lists to tensors.
         3. Create and return a SparseCsrTensor using these tensors.
        """
        crow_indices = [0]
        col_indices = []
        values = []
        
        for row in dense_data:
            row_nnz = 0
            for j, value in enumerate(row):
                if value != 0:
                    col_indices.append(j)
                    values.append(value)
                    row_nnz += 1
            crow_indices.append(crow_indices[-1] + row_nnz)
        
        crow_indices_tensor = Tensor.from_numpy(np.array(crow_indices, dtype=np.int32))
        col_indices_tensor = Tensor.from_numpy(np.array(col_indices, dtype=np.int32))
        values_tensor = Tensor.from_numpy(np.array(values, dtype=dense_data.dtype))
        
        return SparseCsrTensor(
            size=size,
            dtype=dtype,
            device=device,
            crow_indices=crow_indices_tensor,
            col_indices=col_indices_tensor,
            values=values_tensor,
        )


class SparseCsrTensor(diopiSparseCsrTensor):
    def __init__(
        self,
        size=None,
        dtype=None,
        stride=None,
        context=default_context,
        device=Device.AIChip,
        crow_indices=None,
        col_indices=None,
        values=None,
    ):
        if size is None:
            return diopiSparseCsrTensor.__init__(self)

        if isinstance(size, (tuple, list)):
            size = Sizes(list(size))

        if isinstance(stride, (tuple, list)):
            stride = Sizes(list(stride))

        self.col_indices = col_indices
        self.crow_indices = crow_indices
        self.values = values
        
        diopiSparseCsrTensor.__init__(self, size, stride, dtype, device, context, crow_indices, col_indices, values)

    def __str__(self):
        string = f"tensor(crow_indices=tensor({self.crow_indices.numpy()}),\n"
        string += f"       col_indices=tensor({self.col_indices.numpy()}),\n"
        string += f"       values=tensor({self.values.numpy()}),\n"
        string += f"       size={self.size().data}, nnz={self.col_indices.size().data[0]}, layout=sparse_csr)"
        return string

    def __repr__(self):
        return self.__str__()

    def size(self):
        return self.shape()

    def dtype(self):
        return self.get_dtype()

    def reset_shape(self, shape):
        assert isinstance(shape, (tuple, list))
        self.reset_shape(Sizes(list(shape)))

    def numpy(self) -> np.ndarray:
        """
        Converts a sparse tensor in Compressed Sparse Row (CSR) format to a dense NumPy array.
        
        TODO: The current implementation supports only 2D sparse tensors. 
              For multi-dimensional sparse tensor conversion, additional 
              implementation is needed in the future.

        The conversion process involves:
         1. Initialize a dense NumPy array of zeros with the same shape and data type as the sparse tensor.
         2. Extract the CSR components (crow_indices, col_indices, and values) and convert them to NumPy arrays.
         3. Iterate through each row of the sparse tensor:
            - For each row, use crow_indices to find the range of non-zero elements.
            - For each non-zero element in this range, use col_indices to find the column index and values to get the actual value.
            - Assign the value to the corresponding position in the dense array.
        """
        dense_data = np.zeros(self.size().data, dtype=to_numpy_dtype(self.get_dtype()))
        crow_indices = self.crow_indices.numpy()
        col_indices = self.col_indices.numpy()
        values = self.values.numpy()
        for i in range(len(crow_indices) - 1):
            for j in range(crow_indices[i], crow_indices[i + 1]):
                dense_data[i, col_indices[j]] = values[j]
        return dense_data


def raw_like(tensor) -> Tensor:
    return tensor.raw_like()


class Generator(diopiGenerator):
    def __init__(self, state: Tensor):
        diopiGenerator.__init__(self, state)
