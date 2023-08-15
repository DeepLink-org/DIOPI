import pickle
import numpy as np
from generator import Genfunc

# *cfg path, cfg

# .cfg, 

# .dict()

class GenInputData:

    r'''
    '''
    @staticmethod
    def run(diopi_item_config_path='test_diopi_config.cfg', input_path='../config/data/inputs/'):
        pass


class GenTensor:
    def __init__(self, item: dict =None, rules=None) -> None:
        self.item = item
        self.rule = rules
        self.data = None

    def _check_item(self):
        pass

    def gen_data(self):
        pass


    def get_data(self):
        pass



# test for generate functions
if __name__ == '__main__':
    shape = (3, 4)
    dtype = np.float16
    func = f"Genfunc.randn({str(shape)}, np.{np.dtype(dtype).name})"
    t = eval(func)
    print(type(t), '\n', t.dtype) 