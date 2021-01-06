import numpy as np
from datetime import datetime
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
    block_len = int(array.shape[0]/10)

    num_blocks = num_fold-fold
    start = block_len*num_blocks
    a = np.vstack([array[0:start], array[start+block_len:]])
    return a, array[start:start+block_len]
        
if __name__ == '__main__':
    print(strftime())