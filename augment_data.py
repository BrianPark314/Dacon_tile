#-*- coding:utf-8-sig -*-

from common.params import args
from common import eda
from common.utils import seed_everything

if __name__ == '__main__':
    seed_everything(args.seed)
    print('='*50)
    print('Augmenting train data...' + '\n')
    eda.aug_data(args.path / 'train/') 
    print("Run complete.")
    print('='*50)