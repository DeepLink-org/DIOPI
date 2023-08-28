import pytest
import logging
import numpy as np
import sys
import os
import pickle

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conformance'))
from conformance.check_result import CheckResult

# XXX add more case
case_input = {'test check input': [{'input': np.array([1,2,3])}, {'input': np.array([1,2,3])}]}


class TestCheckResult(object):
    @pytest.mark.parametrize('input1,input2', case_input.values(), ids=case_input.keys())
    def test_compare_input(input1, input2):
        logging.info(input1, input2)
