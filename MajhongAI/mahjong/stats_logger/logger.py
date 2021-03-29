from collections import defaultdict
import os
import config as cfg

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(cfg.log_dir)

def logger(data_name):
    def _log(func):
        def record(*args, **kwargs):
            result = func(*args, **kwargs)
            writer.add_scalar(data_name, result)
            return result
        return record
    return _log

if __name__ == '__main__':
    '''
    Example codes to use logger decorator
    !!! Temporarily change the log_dir = '../runs/test_run' to run this test !!!

    '''
    @logger(data_name='loss')
    def calc_loss(a, b):
        return a+b

    @logger(data_name='accuracy')
    def calc_acc(a, b):
        return a*b

    @logger(data_name='score')
    def calc_score(a, b):
        return a/b

    calc_loss(7, 2)
    calc_acc(7, 2)
    calc_score(7, 2)