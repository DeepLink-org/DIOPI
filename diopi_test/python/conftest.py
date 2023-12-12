import logging
import pytest
import pickle
import re

from conformance.global_settings import glob_vars
from conformance.db_operation import db_conn, TestSummary, FuncList, ExcelOperation
from conformance.diopi_runtime import diopi_rt_init, default_context

init_counter = 0


@pytest.fixture(scope='session', autouse=True)
def init_dev():
    global init_counter
    init_counter += 1
    print(f'[Device Init Times] {init_counter} .........')
    diopi_rt_init()


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
    db_data = {'pytest_nodeid': item.nodeid, 'diopi_func_name': glob_vars.cur_test_func}
    if report.when == 'call':
        if report.failed:
            db_data['error_msg'] = f'{report.longrepr.reprcrash.message}'
        elif report.skipped:
            match = re.search(r'Skipped: (.+)\'\)', report.longreprtext)
            if match:
                skip_message = match.group(1)
                db_data['error_msg'] = skip_message
                if 'FunctionNotImplementedError' in skip_message:
                    db_data['not_implemented_flag'] = 1
        db_data['result'] = report.outcome
        db_conn.will_update_device_case(db_data)
    glob_vars.cur_test_func = ''
    glob_vars._func_status.clear()


def pytest_sessionfinish(session, exitstatus):
    db_conn.update_device_case()
    db_conn.insert_func_list()
    db_conn.insert_test_summary()


def pytest_terminal_summary(terminalreporter):
    ExcelOperation().gen_excel()
