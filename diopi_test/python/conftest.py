import logging
import pytest
import pickle

from conformance.run_test import RunTest


def pytest_addoption(parser):
    parser.addoption('--impl_folder', type=str, default='', help='folder to find device configs')


def pytest_collect_file(parent, path):
    if path.basename.endswith("diopi_case_items.cfg"):
        return CustomFile.from_parent(parent, fspath=path)


class CustomFile(pytest.File):
    def collect(self):
        with open(self.fspath, 'rb') as f:
            case_cfg = pickle.load(f)
        for case_name, case_dict in case_cfg.items():
            yield CustomItem.from_parent(self, name=case_name.replace('.pth', ''), case_dict=case_dict)


class CustomItem(pytest.Item):
    def __init__(self, *, case_dict, **kwargs):
        super().__init__(**kwargs)
        self.case_dict = case_dict

        # can add mark for case
        # self.add_marker(pytest.mark.__getattr__('P0'))
        # skip case
        # self.add_marker(pytest.__getattr__('skip')('skip reason'))
        # self.add_marker(pytest.__getattr__('skipif')(condition=True, reason=''))

        # self.add_marker(mark.parametrize(Datagen.gen_params(copy.deepcopy(params))))
        
        # self.parent.add_marker(allure.feature(feature)())
        # self.parent.add_marker(allure.story(story)())
        # allure.title(title)(self.obj)
        
        # setattr(self, 'op_name', op)

    def runtest(self):
        print(self.config.getoption('--impl_folder'))
        RunTest.run(self.case_dict)
