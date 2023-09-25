import pytest
import logging
import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conformance'))
from conformance.global_settings import glob_vars
glob_vars.use_db = True
from conformance.db_operation import DB_Operation, db_conn, Base, BenchMarkCase, DeviceCase, TestSummary, FuncList


cache_path = os.path.join(os.path.dirname(__file__), 'cache')


case_device_case = {
    'test backward case': dict(insert_benchmark_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi', inplace_flag=0, backward_flag=1,
                                                  func_name='batch_norm', case_config={}, result='skipped'),],
                          update_benchmark_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi', result='passed'),],
                          insert_device_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_batch_norm_batch_norm.py', inplace_flag=0, backward_flag=1,
                                            func_name='batch_norm', case_config={}, result='skipped'),],
                          update_device_case=[dict(pytest_nodeid='gencases/diopi_case/test_diopi_batch_norm_batch_norm.py',
                                                   diopi_func_name='diopiBatchNormBackward', result='passed'),],
                          expect_funclist=dict(diopiBatchNorm=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopiBatchNormBackward=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0)),
                          expect_summary=dict(total_case=1, success_case=1, failed_case=0, skip_case=0,
                                              total_func=2, impl_func=2)
                          ),
    'test inplace case': dict(insert_benchmark_case=[dict(case_name='test_add_add_0', model_name='diopi', inplace_flag=1, backward_flag=0,
                                                  func_name='add', case_config={}, result='skipped'),],
                          update_benchmark_case=[dict(case_name='test_add_add_0', model_name='diopi', result='passed'),],
                          insert_device_case=[dict(case_name='test_add_add_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_add_add.py', inplace_flag=1, backward_flag=0,
                                            func_name='add', case_config={}, result='skipped'),],
                          update_device_case=[dict(pytest_nodeid='gencases/diopi_case/test_diopi_add_add.py',
                                                   diopi_func_name='diopiaddInp', result='passed'),],
                          expect_funclist=dict(diopiadd=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopiaddInp=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0)),
                          expect_summary=dict(total_case=1, success_case=1, failed_case=0, skip_case=0,
                                              total_func=2, impl_func=2)
                          ),
    'test inplace backward': dict(insert_benchmark_case=[dict(case_name='test_hardswish_hardswish_0', model_name='diopi', inplace_flag=1, backward_flag=1,
                                                  func_name='hardswish', case_config={}, result='skipped'),],
                          update_benchmark_case=[dict(case_name='test_hardswish_hardswish_0', model_name='diopi', result='passed'),],
                          insert_device_case=[dict(case_name='test_hardswish_hardswish_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_hardswish_hardswish.py', inplace_flag=1, backward_flag=1,
                                            func_name='hardswish', case_config={}, result='skipped'),],
                          update_device_case=[dict(pytest_nodeid='gencases/diopi_case/test_diopi_hardswish_hardswish.py',
                                                   diopi_func_name='diopihardswishBackward', result='passed'),],
                          expect_funclist=dict(diopihardswish=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopihardswishBackward=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopihardswishInp=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0)),
                          expect_summary=dict(total_case=1, success_case=1, failed_case=0, skip_case=0,
                                              total_func=3, impl_func=3)
                          ),
    'test three case some failed': dict(insert_benchmark_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi', inplace_flag=0, backward_flag=1,
                                                  func_name='batch_norm', case_config={}, result='skipped'),
                                                   dict(case_name='test_add_add_0', model_name='diopi', inplace_flag=1, backward_flag=0,
                                                  func_name='add', case_config={}, result='skipped'),
                                                   dict(case_name='test_hardswish_hardswish_0', model_name='diopi', inplace_flag=1, backward_flag=1,
                                                  func_name='hardswish', case_config={}, result='skipped'),],
                          update_benchmark_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi', result='passed'),
                                                 dict(case_name='test_add_add_0', model_name='diopi', result='passed'),
                                                 dict(case_name='test_hardswish_hardswish_0', model_name='diopi', result='passed'),],
                          insert_device_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_batch_norm_batch_norm.py', inplace_flag=0, backward_flag=1,
                                            func_name='batch_norm', case_config={}, result='skipped'),
                                              dict(case_name='test_add_add_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_add_add.py', inplace_flag=1, backward_flag=0,
                                            func_name='add', case_config={}, result='skipped'),
                                              dict(case_name='test_hardswish_hardswish_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_hardswish_hardswish.py', inplace_flag=1, backward_flag=1,
                                            func_name='hardswish', case_config={}, result='skipped'),],
                          update_device_case=[dict(pytest_nodeid='gencases/diopi_case/test_diopi_batch_norm_batch_norm.py',
                                                   diopi_func_name='diopiBatchNormBackward', result='passed'),
                                              dict(pytest_nodeid='gencases/diopi_case/test_diopi_add_add.py',
                                                   diopi_func_name='diopiaddInp', result='skipped'),
                                              dict(pytest_nodeid='gencases/diopi_case/test_diopi_hardswish_hardswish.py',
                                                   diopi_func_name='diopihardswishInp', result='failed'),],
                          expect_funclist=dict(diopiBatchNorm=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopiBatchNormBackward=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopiadd=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=0, failed_case=0, skipped_case=1),
                                               diopiaddInp=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=0, failed_case=0, skipped_case=1),
                                               diopihardswish=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopihardswishBackward=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopihardswishInp=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=0, failed_case=1, skipped_case=0)),
                          expect_summary=dict(total_case=3, success_case=1, failed_case=1, skip_case=1,
                                              total_func=7, impl_func=7)
                          ),
    'test not impl': dict(insert_benchmark_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi', inplace_flag=0, backward_flag=1,
                                                  func_name='batch_norm', case_config={}, result='skipped'),
                                                   dict(case_name='test_add_add_0', model_name='diopi', inplace_flag=1, backward_flag=0,
                                                  func_name='add', case_config={}, result='skipped'),
                                                   dict(case_name='test_hardswish_hardswish_0', model_name='diopi', inplace_flag=1, backward_flag=1,
                                                  func_name='hardswish', case_config={}, result='skipped'),],
                          update_benchmark_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi', result='passed'),
                                                 dict(case_name='test_add_add_0', model_name='diopi', result='passed'),
                                                 dict(case_name='test_hardswish_hardswish_0', model_name='diopi', result='passed'),],
                          insert_device_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_batch_norm_batch_norm.py', inplace_flag=0, backward_flag=1,
                                            func_name='batch_norm', case_config={}, result='skipped'),
                                              dict(case_name='test_add_add_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_add_add.py', inplace_flag=1, backward_flag=0,
                                            func_name='add', case_config={}, result='skipped'),
                                              dict(case_name='test_hardswish_hardswish_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_hardswish_hardswish.py', inplace_flag=1, backward_flag=1,
                                            func_name='hardswish', case_config={}, result='skipped'),],
                          update_device_case=[dict(pytest_nodeid='gencases/diopi_case/test_diopi_batch_norm_batch_norm.py',
                                                   diopi_func_name='diopiBatchNormBackward', result='passed'),
                                              dict(pytest_nodeid='gencases/diopi_case/test_diopi_add_add.py',
                                                   not_implemented_flag=1, diopi_func_name='diopiaddInpScalar', result='failed'),
                                              dict(pytest_nodeid='gencases/diopi_case/test_diopi_hardswish_hardswish.py',
                                                   not_implemented_flag=1, diopi_func_name='diopihardswishBackward', result='failed'),],
                          expect_funclist=dict(diopiBatchNorm=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopiBatchNormBackward=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopiaddScalar=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopiaddInpScalar=dict(not_implemented_flag=1, case_num=1,
                                                                      success_case=0, failed_case=1, skipped_case=0),
                                               diopihardswish=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopihardswishBackward=dict(not_implemented_flag=1, case_num=1,
                                                                      success_case=0, failed_case=1, skipped_case=0),
                                               diopihardswishInp=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=0, failed_case=0, skipped_case=1)),
                          expect_summary=dict(total_case=3, success_case=1, failed_case=2, skip_case=0,
                                              total_func=7, impl_func=5)
                          ),
    'test all failed': dict(insert_benchmark_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi', inplace_flag=0, backward_flag=1,
                                                  func_name='batch_norm', case_config={}, result='skipped'),
                                                   dict(case_name='test_add_add_0', model_name='diopi', inplace_flag=1, backward_flag=0,
                                                  func_name='add', case_config={}, result='skipped'),
                                                   dict(case_name='test_hardswish_hardswish_0', model_name='diopi', inplace_flag=1, backward_flag=1,
                                                  func_name='hardswish', case_config={}, result='skipped'),],
                          update_benchmark_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi', result='passed'),
                                                 dict(case_name='test_add_add_0', model_name='diopi', result='passed'),
                                                 dict(case_name='test_hardswish_hardswish_0', model_name='diopi', result='passed'),],
                          insert_device_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_batch_norm_batch_norm.py', inplace_flag=0, backward_flag=1,
                                            func_name='batch_norm', case_config={}, result='skipped'),
                                              dict(case_name='test_add_add_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_add_add.py', inplace_flag=1, backward_flag=0,
                                            func_name='add', case_config={}, result='skipped'),
                                              dict(case_name='test_hardswish_hardswish_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_hardswish_hardswish.py', inplace_flag=1, backward_flag=1,
                                            func_name='hardswish', case_config={}, result='skipped'),],
                          update_device_case=[dict(pytest_nodeid='gencases/diopi_case/test_diopi_batch_norm_batch_norm.py',
                                                   diopi_func_name='diopiBatchNorm', result='failed'),
                                              dict(pytest_nodeid='gencases/diopi_case/test_diopi_add_add.py',
                                                   diopi_func_name='diopiaddInpScalar', result='failed'),
                                              dict(pytest_nodeid='gencases/diopi_case/test_diopi_hardswish_hardswish.py',
                                                   diopi_func_name='diopihardswishInp', result='failed'),],
                          expect_funclist=dict(diopiBatchNorm=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=0, failed_case=1, skipped_case=0),
                                               diopiBatchNormBackward=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=0, failed_case=0, skipped_case=1),
                                               diopiaddScalar=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopiaddInpScalar=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=0, failed_case=1, skipped_case=0),
                                               diopihardswish=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopihardswishBackward=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=1, failed_case=0, skipped_case=0),
                                               diopihardswishInp=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=0, failed_case=1, skipped_case=0)),
                          expect_summary=dict(total_case=3, success_case=0, failed_case=3, skip_case=0,
                                              total_func=7, impl_func=7)
                          ),
    'test nv failed case': dict(insert_benchmark_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi', inplace_flag=0, backward_flag=1,
                                                  func_name='batch_norm', case_config={}, result='skipped'),],
                          update_benchmark_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi', result='failed'),],
                          insert_device_case=[dict(case_name='test_batch_norm_batch_norm_0', model_name='diopi',
                                            pytest_nodeid='gencases/diopi_case/test_diopi_batch_norm_batch_norm.py', inplace_flag=0, backward_flag=1,
                                            func_name='batch_norm', case_config={}, result='skipped'),],
                          update_device_case=[dict(pytest_nodeid='gencases/diopi_case/test_diopi_batch_norm_batch_norm.py',
                                                   diopi_func_name='diopiBatchNorm', result='failed'),],
                          expect_funclist=dict(diopiBatchNorm=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=0, failed_case=1, skipped_case=0),
                                               diopiBatchNormBackward=dict(not_implemented_flag=0, case_num=1,
                                                                      success_case=0, failed_case=0, skipped_case=1)),
                          expect_summary=dict(total_case=1, success_case=0, failed_case=1, skip_case=0,
                                              total_func=2, impl_func=2)
                          ),
}


class TestDBOperation(object):
    @pytest.fixture(scope='class', name='db_conn')
    def create_and_drop_table(self, request):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        cache_path = os.path.join(cur_path, 'cache')
        db_conn.init_db(f'sqlite:///{cache_path}/testrecord.db')

        yield db_conn

        Base.metadata.drop_all(db_conn.engine)

    @pytest.fixture()
    def drop_device_case(self, request):
        yield
        db_conn.all_case_items = None
        db_conn.func_dict = {}
        db_conn.drop_case_table(BenchMarkCase)
        db_conn.drop_case_table(DeviceCase)
        db_conn.drop_case_table(FuncList)
        db_conn.drop_case_table(TestSummary)

    @pytest.mark.parametrize('case_items', case_device_case.values(), ids=case_device_case.keys())
    @pytest.mark.usefixtures("drop_device_case")
    def test_device_case_funclist_summary(self, case_items, db_conn: DB_Operation):
        db_conn.insert_benchmark_case(case_items['insert_benchmark_case'], {i['case_name']:i for i in case_items['update_benchmark_case']})
        db_data = db_conn.session.query(BenchMarkCase).filter_by(delete_flag=1).all()
        assert len(db_data) == len(case_items['insert_benchmark_case']) == len(case_items['update_benchmark_case'])

        db_conn.insert_device_case(case_items['insert_device_case'])
        db_data = db_conn.session.query(DeviceCase).filter_by(delete_flag=1).all()
        assert len(db_data) == len(case_items['insert_device_case'])

        db_conn.init_test_flag()
        db_data = db_conn.session.query(DeviceCase).filter_by(delete_flag=1, test_flag=0).all()
        assert len(db_data) == len(case_items['insert_device_case'])

        for each_case_item in case_items['update_device_case']:
            db_conn.will_update_device_case(each_case_item)
        db_conn.update_device_case()
        db_data = db_conn.session.query(DeviceCase).filter_by(delete_flag=1).all()
        assert len(db_data) == len(case_items['insert_device_case']) == len(case_items['update_device_case'])
        db_data = db_conn.session.query(DeviceCase).filter_by(delete_flag=1, test_flag=1).all()
        assert len(db_data) == len(case_items['update_device_case'])

        db_conn.insert_func_list()
        db_data = db_conn.session.query(FuncList).filter_by(delete_flag=1).all()
        assert len(db_data) == len(case_items['expect_funclist'])
        for actual_data in db_data:
            expect_data = case_items['expect_funclist'].get(actual_data.diopi_func_name)
            for key, each_expect_data in expect_data.items():
                assert each_expect_data == getattr(actual_data, key)
            assert actual_data.success_rate == expect_data['success_case'] / expect_data['case_num']

        db_conn.insert_test_summary()
        db_data = db_conn.session.query(TestSummary).filter_by(delete_flag=1).all()
        assert len(db_data) == 1
        for key, expect_data in case_items['expect_summary'].items():
            assert expect_data == getattr(db_data[0], key)
        assert db_data[0].success_rate == case_items['expect_summary']['success_case'] / case_items['expect_summary']['total_case']
        assert db_data[0].func_coverage_rate == case_items['expect_summary']['impl_func'] / case_items['expect_summary']['total_func']

        for model in [BenchMarkCase, DeviceCase, FuncList, TestSummary]:
            db_conn.drop_case_table(model)
            db_data = db_conn.session.query(model).all()
            for item in db_data:
                assert item.delete_flag == 0
