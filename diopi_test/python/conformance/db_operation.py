import threading
import os
import pickle
import time
import pandas as pd
from datetime import datetime
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    FLOAT,
    String,
    create_engine,
    event,
    func,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine import Engine
from sqlalchemy import text


from conformance.global_settings import glob_vars
from conformance.utils import logger

Base = declarative_base()


class BenchMarkCase(Base):
    __tablename__ = "benchmark_case"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_name = Column(String(100))
    model_name = Column(String(100))
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
        return f"id: {self.id}, case name: {self.case_name}, model name: {self.model_name}, \
            func name: {self.func_name}, result: {self.result}, delete_flag: {self.delete_flag}, \
                created_time:{self.created_time}, updated_time: {self.updated_time}"


class DeviceCase(Base):
    __tablename__ = "device_case"

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
        return f"id: {self.id}, case name: {self.case_name}, model name: {self.model_name}, \
            func name: {self.func_name}, result: {self.result}"


class FuncList(Base):
    __tablename__ = "func_list"

    id = Column(Integer, primary_key=True, autoincrement=True)
    diopi_func_name = Column(String(100))
    func_name = Column(String(100))
    not_implemented_flag = Column(Integer)
    case_num = Column(Integer)
    success_case = Column(Integer)
    failed_case = Column(Integer)
    skipped_case = Column(Integer)
    success_rate = Column(FLOAT)
    delete_flag = Column(Integer)
    created_time = Column(DateTime)
    updated_time = Column(DateTime)


class TestSummary(Base):
    __tablename__ = "test_summary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    total_case = Column(Integer)
    success_case = Column(Integer)
    failed_case = Column(Integer)
    skipped_case = Column(Integer)
    total_func = Column(Integer)
    impl_func = Column(Integer)
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
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()
    logger.debug("Start Query:\n%s" % statement)
    # logger.debug("Parameters:\n%r" % (parameters,))


@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - context._query_start_time
    logger.debug("Query Complete!")
    logger.debug("Total Time: %.02fms" % (total * 1000))
    logger.debug("affected row count: %d" % context.rowcount)


class DB_Operation(object):
    _instance_lock = threading.Lock()
    func_dict = {}
    all_case_items = None

    def __init__(self):
        pass

    @use_db(glob_vars.use_db)
    def init_db(self, db_path="sqlite:///./cache/testrecord.db"):
        self.engine = create_engine(f"{db_path}?check_same_thread=False", echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    @use_db(glob_vars.use_db)
    def drop_case_table(self, case_module):
        self.session.query(case_module).filter_by(delete_flag=1).update(
            {"delete_flag": 0}
        )
        self.session.commit()

    @use_db(glob_vars.use_db)
    def insert_benchmark_case(
        self, geninput_case_items: list, genoutput_case_items: dict
    ):
        for item in geninput_case_items:
            output_case_item = genoutput_case_items.get(item["case_name"])
            if output_case_item is not None:
                item.update(output_case_item)
            item["case_config"] = pickle.dumps(item["case_config"])
            item["delete_flag"] = 1
            item["created_time"] = datetime.now()
            item["updated_time"] = datetime.now()
        self.session.bulk_insert_mappings(BenchMarkCase, geninput_case_items)
        self.session.commit()

    @use_db(glob_vars.use_db)
    def insert_device_case(self, case_items):
        for item in case_items:
            item["case_config"] = pickle.dumps(item["case_config"])
            item["not_implemented_flag"] = 0
            item["delete_flag"] = 1
            item["test_flag"] = 0
            item["created_time"] = datetime.now()
            item["updated_time"] = datetime.now()
        self.session.bulk_insert_mappings(DeviceCase, case_items)
        self.session.commit()

    @use_db(glob_vars.use_db)
    def init_test_flag(self):
        self.session.query(DeviceCase).filter_by(delete_flag=1, test_flag=1).update(
            {"test_flag": 0}
        )
        self.session.commit()

    @use_db(glob_vars.use_db)
    def will_update_device_case(self, case_item):
        if self.all_case_items is None:
            self.all_case_items = {
                i.pytest_nodeid: i.__dict__
                for i in self.session.query(DeviceCase).filter_by(delete_flag=1).all()
            }
        case_model = self.all_case_items[case_item["pytest_nodeid"]]
        if case_item.get("case_config"):
            case_item["case_config"] = pickle.dumps(case_item["case_config"])
        diopi_func_name_list = list(glob_vars.func_status.keys())
        diopi_func_name_list = list(filter(lambda x: case_model["func_name"].replace('_', '') in x.lower(), diopi_func_name_list))
        case_item["diopi_func_name"] = ",".join(diopi_func_name_list)
        case_item["updated_time"] = datetime.now()
        case_item["id"] = case_model["id"]
        case_item["test_flag"] = 1
        self.all_case_items[case_item["pytest_nodeid"]].update(case_item)

        self.expand_func_list(
            case_item.get(
                "not_implemented_flag", case_model["not_implemented_flag"]
            ),
            diopi_func_name_list,
            case_model["func_name"],
        )

    @use_db(glob_vars.use_db)
    def expand_func_list(
        self,
        not_implemented_flag,
        func_name_list,
        func_name,
    ):
        tmp_func_status = {i: "skipped" for i in func_name_list}
        tmp_func_status.update(glob_vars.func_status)
        for _, diopi_func in enumerate(func_name_list):
            if diopi_func not in self.func_dict:
                self.func_dict[diopi_func] = dict(
                    diopi_func_name=diopi_func,
                    func_name=func_name,
                    not_implemented_flag=0,
                    case_num=0,
                    success_case=0,
                    failed_case=0,
                    skipped_case=0,
                    delete_flag=1,
                    created_time=datetime.now(),
                    updated_time=datetime.now(),
                )
            if tmp_func_status[diopi_func] == "passed":
                self.func_dict[diopi_func]["success_case"] += 1
            elif tmp_func_status[diopi_func] == "skipped":
                self.func_dict[diopi_func]["skipped_case"] += 1
            else:
                self.func_dict[diopi_func]["failed_case"] += 1
                if not_implemented_flag == 1:
                    self.func_dict[diopi_func]["not_implemented_flag"] = 1
            self.func_dict[diopi_func]["case_num"] += 1

    @use_db(glob_vars.use_db)
    def update_device_case(self):
        if self.all_case_items:
            self.session.bulk_update_mappings(DeviceCase, self.all_case_items.values())
            self.session.commit()

    @use_db(glob_vars.use_db)
    def insert_func_list(self):
        for diopi_func in self.func_dict.values():
            diopi_func["success_rate"] = (
                diopi_func["success_case"] / diopi_func["case_num"]
            )
        self.session.bulk_insert_mappings(FuncList, self.func_dict.values())
        self.session.commit()

    @use_db(glob_vars.use_db)
    def insert_test_summary(self):
        total_case = (
            self.session.query(func.count(BenchMarkCase.func_name))
            .filter_by(delete_flag=1)
            .one()[0]
        )
        success_case = (
            self.session.query(func.count(DeviceCase.func_name))
            .filter_by(result="passed", delete_flag=1, test_flag=1)
            .one()[0]
        )
        failed_case = (
            self.session.query(func.count(DeviceCase.func_name))
            .filter_by(result="failed", delete_flag=1, test_flag=1)
            .one()[0]
        )
        skipped_case = (
            self.session.query(func.count(DeviceCase.func_name))
            .filter_by(result="skipped", delete_flag=1, test_flag=1)
            .one()[0]
        )
        sql = text(
            """
            WITH FuncCounts AS (
                SELECT
                    func_name as func_name_benchmark,
                    1 + COALESCE(inplace_flag, 0) + COALESCE(backward_flag, 0) AS total_func
                FROM
                    benchmark_case
                WHERE
                    delete_flag = 1
                GROUP BY
                    func_name_benchmark
                UNION ALL
                SELECT
                    func_name as func_name_funclist,
                    COUNT(*) as func_count
                FROM func_list
                WHERE delete_flag = 1
                GROUP BY func_name_funclist
            )
            SELECT SUM(MaxCount) AS TotalMaxCount
            FROM (
                SELECT func_name_benchmark, MAX(total_func) AS MaxCount
                FROM FuncCounts
                GROUP BY func_name_benchmark
            ) MaxCounts;
        """
        )
        result = self.session.execute(sql)
        total_func = result.scalar()
        impl_func = (
            self.session.query(func.count(FuncList.diopi_func_name))
            .filter_by(not_implemented_flag=0, delete_flag=1)
            .one()[0]
        )
        summary_item = dict(
            total_case=total_case,
            success_case=success_case,
            failed_case=failed_case,
            skipped_case=skipped_case,
            total_func=total_func,
            impl_func=impl_func,
            func_coverage_rate=impl_func / total_func,
            success_rate=success_case / total_case,
            delete_flag=1,
            created_time=datetime.now(),
            updated_time=datetime.now(),
        )
        self.session.add(TestSummary(**summary_item))
        self.session.commit()

    @use_db(glob_vars.use_db)
    def query_data(self, case_module, **kwargs):
        data = self.session.query(case_module).filter_by(**kwargs).all()
        return data

    def __new__(cls, *args, **kwargs):
        if not hasattr(DB_Operation, "_instance"):
            with DB_Operation._instance_lock:
                if not hasattr(DB_Operation, "_instance"):
                    DB_Operation._instance = object.__new__(cls)
        return DB_Operation._instance


db_conn = DB_Operation()


class ExcelOperation(object):
    def __init__(self) -> None:
        self.excel_writer = pd.ExcelWriter("report.xlsx", engine="xlsxwriter")

    def add_benchmark_case_sheet(self):
        sheet_name = "Benchmark Test Result"
        data_query = db_conn.query_data(BenchMarkCase, delete_flag=1)
        df = pd.DataFrame(
            [data.__dict__ for data in data_query],
            columns=BenchMarkCase.__table__.columns.keys(),
        )
        df["case_config"] = df["case_config"].apply(lambda x: pickle.loads(x))
        columns = [
            "case_name",
            "model_name",
            "func_name",
            "case_config",
            "result",
            "error_msg",
        ]
        df.index = df.index + 1
        df.to_excel(self.excel_writer, sheet_name=sheet_name, columns=columns)
        self.adjust_column(df, sheet_name)

    def add_device_case_sheet(self):
        sheet_name = "Device Test Result"
        data_query = db_conn.query_data(DeviceCase, delete_flag=1, test_flag=1)
        df = pd.DataFrame(
            [data.__dict__ for data in data_query],
            columns=DeviceCase.__table__.columns.keys(),
        )
        df["case_config"] = df["case_config"].apply(lambda x: pickle.loads(x))
        columns = [
            "pytest_nodeid",
            "case_name",
            "model_name",
            "func_name",
            "diopi_func_name",
            "not_implemented_flag",
            "case_config",
            "result",
            "error_msg",
        ]
        df.index = df.index + 1
        df.to_excel(self.excel_writer, sheet_name="Device Test Result", columns=columns)
        self.adjust_column(df, sheet_name)

    def add_func_list_sheet(self):
        sheet_name = "Func List"
        data_query = db_conn.query_data(FuncList, delete_flag=1)
        df = pd.DataFrame(
            [data.__dict__ for data in data_query],
            columns=FuncList.__table__.columns.keys(),
        )
        columns = [
            "diopi_func_name",
            "not_implemented_flag",
            "case_num",
            "success_case",
            "failed_case",
            "skipped_case",
            "success_rate",
        ]
        df.index = df.index + 1
        df.to_excel(self.excel_writer, sheet_name="Func List", columns=columns)
        self.adjust_column(df, sheet_name)

    def add_sumary_sheet(self):
        sheet_name = "Sumary"
        data_query = db_conn.query_data(TestSummary, delete_flag=1)
        df = pd.DataFrame(
            [data.__dict__ for data in data_query],
            columns=TestSummary.__table__.columns.keys(),
        )
        columns = [
            "total_case",
            "success_case",
            "failed_case",
            "skipped_case",
            "total_func",
            "impl_func",
            "success_rate",
            "func_coverage_rate",
        ]
        df.index = df.index + 1
        df.to_excel(self.excel_writer, sheet_name="Sumary", columns=columns)
        self.adjust_column(df, sheet_name)

    def adjust_column(self, data, sheet_name):
        worksheet = self.excel_writer.sheets[sheet_name]
        for i, column_name in enumerate(data.columns):
            format_percent = None
            if column_name in ["success_rate", "func_coverage_rate"]:
                format_percent = self.excel_writer.book.add_format(
                    {"num_format": "0.00%"}
                )
            worksheet.set_column(i, i, len(column_name) + 2, format_percent)

    @use_db(glob_vars.use_db)
    def gen_excel(self):
        self.add_benchmark_case_sheet()
        self.add_device_case_sheet()
        self.add_func_list_sheet()
        self.add_sumary_sheet()
        self.excel_writer.close()
