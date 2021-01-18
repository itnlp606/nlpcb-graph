import numpy as np
from datetime import datetime
from copy import deepcopy
from math import ceil
import os

class CountSmooth:
    def __init__(self, max_steps):
        self.q = []
        self.max_steps = max_steps

    def get(self):
        return np.mean(self.q)

    def add(self, value):
        if len(self.q) > self.max_steps:
            self.q.pop(0)
        self.q.append(value)


def strftime():
    return datetime.now().strftime('%m-%d_%H-%M-%S')


def print_execute_time(func):
    '''函数执行时间装饰器'''
    from time import time

    def wrapper(*args, **kwargs):
        start = time()
        func_return = func(*args, **kwargs)
        end = time()
        print(f'{func.__name__}() execute time: {end - start}s')
        return func_return

    return wrapper

def print_empty_line(func):
    ''' 空行打印装饰器'''
    def wrapper(*args, **kwargs):
        print()
        func_return = func(*args, **kwargs)
        print()
        return func_return

    return wrapper

def clear_dir(dir):
    files = [os.path.join(dir, f) for f in os.listdir(dir)]
    for f in files:
        os.remove(f)

def divide_dataset(seed, array, num_fold, fold):
    np.random.seed(seed)
    np.random.shuffle(array)
    block_len = int(array.shape[0]/num_fold)

    start_id = num_fold-fold
    start = block_len*start_id
    a = np.vstack([array[0:start], array[start+block_len:]])
    return a, array[start:start+block_len]

def divide_by_type(dic, num_fold, fold):
    # 传入的是数组的数组
    gk = deepcopy(dic)
    block_len = ceil(len(gk)/num_fold)

    start_id = num_fold-fold
    start = start_id*block_len
    valid = stack(gk[start:start+block_len])

    del gk[start:start+block_len]
    train = stack(gk)
    return train, valid

# stack a list of tensor, with different shape
def stack(array_list):
    a = np.array(array_list[0], dtype=object)
    for i in range(1, len(array_list)):
        a = np.vstack((a, array_list[i]))
    return a

if __name__ == '__main__':
    print(strftime())