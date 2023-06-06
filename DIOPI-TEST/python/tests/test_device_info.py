from conformance.utils import check_function, logger
from ctypes import c_char_p, c_char, cast, POINTER


class TestDeviceInfo(object):

    def test_vendor_name(self):
        func = check_function("diopiGetVendorName")
        vendor_name = func()

        assert vendor_name is not None, "no return value"
        assert "Device" in vendor_name

        logger.info(f"Vendor name is {vendor_name}.")

    def test_impl_version(self):
        func = check_function("diopiGetImplVersion")
        diopirt_func = check_function("diopiGetVersion")
        impl_version = func()
        diopi_version = diopirt_func()

        assert impl_version is not None, "no return value"
        assert diopi_version is not None, "no return value"
        assert diopi_version in impl_version

        logger.info(f"Impl_version is {impl_version}.")
