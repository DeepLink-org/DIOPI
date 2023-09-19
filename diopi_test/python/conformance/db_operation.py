import threading
import os
import pickle
import time
from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, FLOAT, String, create_engine, event, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine import Engine


from conformance.global_settings import glob_vars
from conformance.utils import logger

Base = declarative_base()


class BenchMarkCase(Base):
    __tablename__ = 'benchmark_case'

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_name = Column(String(100), comment='test case name')
    model_name = Column(String(100), comment='test model name')
    func_name = Column(String(100))
    case_config = Column(String(1000))
    result = Column(String(10))
    error_msg = Column(String(1000))
    delete_flag = Column(Integer)
    inplace_flag = Column(Integer)
    backward_flag = Column(Integer)
    created_time = Column(DateTime)
    updated_time = Column(DateTime)

    def __repr__(self):
        return f'id: {self.id}, case name: {self.case_name}, model name: {self.model_name}, \
            func name: {self.func_name}, result: {self.result}'


class DeviceCase(Base):
    __tablename__ = 'device_case'

    id = Column(Integer, primary_key=True, autoincrement=True)
    pytest_nodeid = Column(String(100))
    case_name = Column(String(100))
    model_name = Column(String(100))
    func_name = Column(String(100))
    diopi_func_name = Column(String(100))
    not_implemented_flag = Column(Integer)
    case_config = Column(String(1000))
    result = Column(String(10))
    error_msg = Column(String(1000))
    delete_flag = Column(Integer)
    test_flag = Column(Integer)
    inplace_flag = Column(Integer)
    backward_flag = Column(Integer)
    created_time = Column(DateTime)
    updated_time = Column(DateTime)

    def __repr__(self):
        return f'id: {self.id}, case name: {self.case_name}, model name: {self.model_name}, \
            func name: {self.func_name}, result: {self.result}'


class FuncList(Base):
    __tablename__ = 'func_list'

    id = Column(Integer, primary_key=True, autoincrement=True)
    diopi_func_name = Column(String(100))
    not_implemented_flag = Column(Integer)
    case_num = Column(Integer)
    success_case_num = Column(Integer)
    failed_case_num = Column(Integer)
    skipped_case_num = Column(Integer)
    success_rate = Column(FLOAT)
    delete_flag = Column(Integer)
    created_time = Column(DateTime)
    updated_time = Column(DateTime)


class TestSummary(Base):
    __tablename__ = 'test_summary'

    id = Column(Integer, primary_key=True, autoincrement=True)
    total_case_banchmark = Column(Integer)
    success_case_device = Column(Integer)
    failed_case_deivce = Column(Integer)
    skip_case_device = Column(Integer)
    total_func_benchmark = Column(Integer)
    impl_func_device = Column(Integer)
    success_rate = Column(FLOAT)
    func_coverage_rate = Column(FLOAT)
    delete_flag = Column(Integer)
    created_time = Column(DateTime)
    updated_time = Column(DateTime)


def use_db(if_use_db):
    def wrapper(func):
        def decorator(*args, **kwargs):
            if if_use_db:
                return func(*args, **kwargs)
        return decorator
    return wrapper


@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, 
                        parameters, context, executemany):
    context._query_start_time = time.time()
    logger.debug("Start Query:\n%s" % statement)
    # logger.debug("Parameters:\n%r" % (parameters,))


@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, 
                        parameters, context, executemany):
    total = time.time() - context._query_start_time
    logger.debug("Query Complete!")

    logger.debug("Total Time: %.02fms" % (total*1000))


class DB_Operation(object):
    _instance_lock = threading.Lock()
    all_case_items = []
    func_dict = {}

    def __init__(self, db_path='sqlite:///./cache/testrecord.db?check_same_thread=False'):
        self.db_path = db_path

    @use_db(glob_vars.use_db)
    def init_db(self):
        self.engine = create_engine(self.db_path, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    @use_db(glob_vars.use_db)
    def drop_case_table(self, case_module):
        self.session.query(case_module).filter_by(delete_flag=1).update({'delete_flag': 0})

    @use_db(glob_vars.use_db)
    def insert_benchmark_case(self, case_items):
        for item in case_items:
            item['case_config'] = pickle.dumps(item['case_config'])
            item['delete_flag'] = 1
            item['created_time'] = datetime.now()
            item['updated_time'] = datetime.now()
        self.session.bulk_insert_mappings(BenchMarkCase, case_items)
        self.session.commit()

    @use_db(glob_vars.use_db)
    def update_benchmark_case(self, case_items):
        model_name = case_items[0]['model_name']
        benchmark_model = self.session.query(BenchMarkCase).filter_by(model_name=model_name, delete_flag=1).all()
        case_id_map = {i.case_name: i.id for i in benchmark_model}
        for item in case_items:
            # case_model = self.session.query(BenchMarkCase).filter_by(case_name=item['case_name'], model_name=item['model_name'], delete_flag=1).first()
            if item.get('case_config'):
                item['case_config'] = pickle.dumps(item['case_config'])
            item['updated_time'] = datetime.now()
            item['id'] = case_id_map[item['case_name']]
        self.session.bulk_update_mappings(BenchMarkCase, case_items)
        self.session.commit()

    @use_db(glob_vars.use_db)
    def will_insert_device_case(self, case_items):
        for item in case_items:
            item['case_config'] = pickle.dumps(item['case_config'])
            item['not_implemented_flag'] = 0
            item['delete_flag'] = 1
            item['test_flag'] = 0
            item['created_time'] = datetime.now()
            item['updated_time'] = datetime.now()
        self.all_case_items.extend(case_items)

    @use_db(glob_vars.use_db)
    def insert_device_case(self):
        self.session.bulk_insert_mappings(DeviceCase, self.all_case_items)
        self.session.commit()

    @use_db(glob_vars.use_db)
    def init_test_flag(self):
        self.session.query(DeviceCase).filter_by(delete_flag=1, test_flag=1).update({'test_flag': 0})

    @use_db(glob_vars.use_db)
    def will_update_device_case(self, case_item):
        case_model = self.session.query(DeviceCase).filter_by(pytest_nodeid=case_item['pytest_nodeid'], delete_flag=1).first()
        inplace_flag = case_model.inplace_flag
        backward_flag = case_model.backward_flag
        if case_item.get('case_config'):
            case_item['case_config'] = pickle.dumps(case_item['case_config'])

        diopi_func_name = case_item['diopi_func_name'].replace('Inp', '').replace('Backward', '')
        diopi_func_name_list = [diopi_func_name]
        if inplace_flag:
            if 'Scalar' in diopi_func_name:
                diopi_func_name_list.append(f'{diopi_func_name.replace("Scalar", "")}InpScalar')
            else:
                diopi_func_name_list.append(f'{diopi_func_name}Inp')
        if backward_flag:
            diopi_func_name_list.append(f'{diopi_func_name}Backward')
        case_item['diopi_func_name'] = ','.join(diopi_func_name_list)
        case_item['updated_time'] = datetime.now()
        case_item['id'] = case_model.id
        case_item['test_flag'] = 1
        self.all_case_items.append(case_item)

        self.expand_func_list(diopi_func_name, case_item.get('not_implemented_flag', case_model.not_implemented_flag),
                              diopi_func_name_list, case_item['result'])

    @use_db(glob_vars.use_db)
    def expand_func_list(self, last_diopi_func_name, not_implemented_flag, func_name_list, result):
        for index, func in enumerate(func_name_list):
            if func not in self.func_dict:
                self.func_dict[func] = dict(diopi_func_name=func,
                                           not_implemented_flag=0,
                                           case_num=0,
                                           success_case_num=0,
                                           failed_case_num=0,
                                           skipped_case_num=0,
                                           delete_flag=1,
                                           created_time=datetime.now(),
                                           updated_time=datetime.now(),
                                    )
            if result == 'passed':
                self.func_dict[func]['success_case_num'] += 1
            elif result == 'skipped':
                self.func_dict[func]['skipped_case_num'] += 1
            else:
                if func == last_diopi_func_name:
                    self.func_dict[func]['failed_case_num'] += 1
                    if not_implemented_flag == 1:
                        self.func_dict[func]['not_implemented_flag'] = 1
                elif index < func_name_list.index(last_diopi_func_name):
                    self.func_dict[func]['success_case_num'] += 1
                else:
                    self.func_dict[func]['skipped_case_num'] += 1
            self.func_dict[func]['case_num'] += 1

    @use_db(glob_vars.use_db)
    def update_device_case(self):
        self.session.bulk_update_mappings(DeviceCase, self.all_case_items)
        self.session.commit()

    @use_db(glob_vars.use_db)
    def insert_func_list(self):
        for func in self.func_dict.values():
            func['success_rate'] = func['success_case_num'] / func['case_num']
        self.session.bulk_insert_mappings(FuncList, self.func_dict.values())
        self.session.commit()

    @use_db(glob_vars.use_db)
    def insert_test_summary(self):
        total_case_banchmark = self.session.query(func.count(BenchMarkCase.func_name)).filter_by(delete_flag=1).one()[0]
        success_case_device = self.session.query(func.count(DeviceCase.func_name)).filter_by(result='passed', delete_flag=1, test_flag=1).one()[0]
        failed_case_deivce = self.session.query(func.count(DeviceCase.func_name)).filter_by(result='failed', delete_flag=1, test_flag=1).one()[0]
        skip_case_device = self.session.query(func.count(DeviceCase.func_name)).filter_by(result='skipped', delete_flag=1, test_flag=1).one()[0]
        total_func_benchmark = sum([1 + i[1] + i[2] for i in self.session\
            .query(BenchMarkCase.func_name, BenchMarkCase.inplace_flag, BenchMarkCase.backward_flag) \
            .filter_by(delete_flag=1).group_by(BenchMarkCase.func_name).all()])
        impl_func_device = self.session.query(func.count(FuncList.diopi_func_name)).filter_by(not_implemented_flag=0, delete_flag=1).one()[0]
        summary_item = dict(
            total_case_banchmark=total_case_banchmark,
            success_case_device=success_case_device,
            failed_case_deivce=failed_case_deivce,
            skip_case_device=skip_case_device,
            total_func_benchmark=total_func_benchmark,
            impl_func_device=impl_func_device,
            func_coverage_rate=impl_func_device / total_func_benchmark,
            success_rate=success_case_device / total_case_banchmark,
            delete_flag=1,
            created_time=datetime.now(),
            updated_time=datetime.now()
        )
        self.session.add(TestSummary(**summary_item))
        self.session.commit()

    def __new__(cls, *args, **kwargs):
        if not hasattr(DB_Operation, "_instance"):
            with DB_Operation._instance_lock:
                if not hasattr(DB_Operation, "_instance"):
                    DB_Operation._instance = object.__new__(cls)
        return DB_Operation._instance


db_conn = DB_Operation()
