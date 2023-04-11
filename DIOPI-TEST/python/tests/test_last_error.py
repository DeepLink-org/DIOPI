from conformance.utils import check_function, check_returncode, get_last_error, logger
from ctypes import c_char_p, c_void_p, c_char, c_int64, cast, POINTER
from conformance.diopi_runtime import default_context


class TestGetLastError(object):
    init_last_error = get_last_error()

    def cat(self):
        func = check_function("diopiCat")
        ret = func(default_context.context_handle, c_void_p(),
                   c_void_p(), c_int64(0), c_int64(0))
        assert ret != 0, "there is no error to test"

    def test_device_impl_get_last_error(self):
        func = check_function("diopiGetLastErrorString")
        func.restype = POINTER(c_char)
        last_error_str = func()
        last_error_str = cast(last_error_str, c_char_p)
        last_error = str(last_error_str.value, encoding="utf-8")
        assert isinstance(last_error, str), "no error return"

        self.cat()

        new_last_error_str = func()
        new_last_error_str = cast(new_last_error_str, c_char_p)
        new_last_error = str(new_last_error_str.value, encoding="utf-8")
        assert last_error != new_last_error

    def test_diopirt_get_last_error(self):
        assert isinstance(self.init_last_error, str), "no error return"

        # test_device_impl_get_last_error has called before cat() to upate error.
        last_error_str = get_last_error()
        assert self.init_last_error != last_error_str
