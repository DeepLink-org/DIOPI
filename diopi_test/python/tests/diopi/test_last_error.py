from conformance.diopi_functions import check_function
from ctypes import c_char_p, c_void_p, c_char, c_int64, cast, POINTER
from conformance.diopi_runtime import default_context, get_last_error
from conformance.diopi_runtime import diopiTensor


class TestGetLastError(object):
    init_last_error = get_last_error()

    def cat(self):
        func = check_function("diopiCat")
        input_tensors = []
        ret = func(default_context, diopiTensor(),
                   input_tensors, 0, 0)
        assert ret != 0, "there is no error to test"

    def test_device_impl_get_last_error(self):
        func = check_function("diopiGetLastErrorString")
        last_error = func()
        assert isinstance(last_error, str), "no error return"

        self.cat()

        new_last_error = func()
        assert last_error != new_last_error

    def test_diopirt_get_last_error(self):
        assert isinstance(self.init_last_error, str), "no error return"

        # test_device_impl_get_last_error has called before cat() to upate error.
        last_error_str = get_last_error()
        assert self.init_last_error != last_error_str
