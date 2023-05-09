#-*- coding:utf-8 -*-

from common.params import args
from common import eda


if __name__ == '__main__':
    print('='*50)
    print('Augmenting train data...' + '\n')
    eda.aug_data(args.path / 'train/') 
    print("Run complete.")
    print('='*50)