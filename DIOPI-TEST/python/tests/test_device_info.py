from conformance.utils import check_function, logger
from ctypes import c_char_p, c_char, cast, POINTER


class TestDeviceInfo(object):

    def test_vendor_name(self):
        func = check_function("diopiGetVendorName")
        func.restype = POINTER(c_char)
        vendor_name_str = func()
        vendor_name_str = cast(vendor_name_str, c_char_p)

        assert vendor_name_str.value is not None, "no return value"
        vendor_name = str(vendor_name_str.value, encoding="utf-8")
        assert "Device" in vendor_name

        logger.info(f"Vendor name is {vendor_name}.")

    def test_impl_version(self):
        func = check_function("diopiGetImplVersion")
        diopirt_func = check_function("diopiGetVersion")
        func.restype = POINTER(c_char)
        diopirt_func.restype = POINTER(c_char)
        impl_version_str = func()
        diopi_verison_str = diopirt_func()
        impl_version_str = cast(impl_version_str, c_char_p)
        diopi_version_str = cast(diopi_verison_str, c_char_p)

        assert impl_version_str.value is not None, "no return value"
        assert diopi_version_str.value is not None, "no return value"
        impl_version = str(impl_version_str.value, encoding="utf-8")
        diopi_version = str(diopi_version_str.value, encoding="utf-8")
        assert diopi_version in impl_version

        logger.info(f"Impl_version is {impl_version}.")
