import threading
import os
import pickle
import time
from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, Integer, String, create_engine, event, bindparam, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine import Engine


from conformance.global_settings import glob_vars
from conformance.utils import logger

Base = declarative_base()


class BenchMarkCase(Base):
    __tablename__ = 'benchmark_case'

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_name = Column(String(100))
    model_name = Column(String(100))
    func_name = Column(String(100))
    case_config = Column(String(1000))
    result = Column(String(10))
    error_msg = Column(String(1000))
    delete_flag = Column(Integer)
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
    created_time = Column(DateTime)
    updated_time = Column(DateTime)

    def __repr__(self):
        return f'id: {self.id}, case name: {self.case_name}, model name: {self.model_name}, \
            func name: {self.func_name}, result: {self.result}'


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

    def __init__(self):
        self.db_path = 'sqlite:///./cache/testrecord.db?check_same_thread=False'

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
        for item in case_items:
            case_model = self.session.query(BenchMarkCase).filter_by(case_name=item['case_name'], model_name=item['model_name'], delete_flag=1).first()
            if item.get('case_config'):
                item['case_config'] = pickle.dumps(item['case_config'])
            item['updated_time'] = datetime.now()
            item['id'] = case_model.id
        self.session.bulk_update_mappings(BenchMarkCase, case_items)
        self.session.commit()

    @use_db(glob_vars.use_db)
    def will_insert_device_case(self, case_items):
        for item in case_items:
            item['case_config'] = pickle.dumps(item['case_config'])
            item['not_implemented_flag'] = 0
            item['delete_flag'] = 1
            item['created_time'] = datetime.now()
            item['updated_time'] = datetime.now()
        self.all_case_items.extend(case_items)

    @use_db(glob_vars.use_db)
    def insert_device_case(self):
        self.session.bulk_insert_mappings(DeviceCase, self.all_case_items)
        self.session.commit()

    @use_db(glob_vars.use_db)
    def will_update_device_case(self, case_item):
        case_model = self.session.query(DeviceCase).filter_by(pytest_nodeid=case_item['pytest_nodeid'], delete_flag=1).first()
        if case_item.get('case_config'):
            case_item['case_config'] = pickle.dumps(case_item['case_config'])
        case_item['updated_time'] = datetime.now()
        case_item['id'] = case_model.id
        self.all_case_items.append(case_item)

    @use_db(glob_vars.use_db)
    def update_device_case(self):
        self.session.bulk_update_mappings(DeviceCase, self.all_case_items)
        self.session.commit()

    def __new__(cls, *args, **kwargs):
        if not hasattr(DB_Operation, "_instance"):
            with DB_Operation._instance_lock:
                if not hasattr(DB_Operation, "_instance"):
                    DB_Operation._instance = object.__new__(cls)
        return DB_Operation._instance


db_conn = DB_Operation()
