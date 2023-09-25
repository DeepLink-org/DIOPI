import logging
import pytest
import pickle

from conformance.global_settings import glob_vars
from conformance.db_operation import db_conn, TestSummary, FuncList


def pytest_addoption(parser):
    parser.addoption('--impl_folder', type=str, default='', help='folder to find device configs')

def pytest_sessionstart(session):
    db_conn.init_test_flag()
    db_conn.drop_case_table(TestSummary)
    db_conn.drop_case_table(FuncList)

# def pytest_runtest_setup(item):
#     glob_vars.cur_pytest_nodeid = item.nodeid

@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    out = yield
    report = out.get_result()
    item = {'pytest_nodeid': item.nodeid, 'diopi_func_name': glob_vars.cur_test_func}
    if report.when == 'call':
        if report.failed:
            err_msg = f"[message] {report.longrepr.reprcrash.message[:900]}......, [path] {report.longrepr.reprcrash.path}, [lineno] {report.longrepr.reprcrash.lineno}".replace('\'','')
            item['error_msg'] = err_msg
            if 'FunctionNotImplementedError' in report.longrepr.reprcrash.message:
                item['not_implemented_flag'] = 1
        item['result'] = report.outcome
        db_conn.will_update_device_case(item)


def pytest_sessionfinish(session, exitstatus):
    db_conn.update_device_case()
    db_conn.insert_func_list()
    db_conn.insert_test_summary()