from conformance.diopi_functions import check_function, logger
from ctypes import c_char_p, c_char, cast, POINTER


class TestDeviceInfo(object):

    def test_vendor_name(self):
        func = check_function("diopiGetVendorName")
        vendor_name = func()

        assert vendor_name is not None, "no return value"
        assert "Device" in vendor_name

        logger.info(f"Vendor name is {vendor_name}.")
