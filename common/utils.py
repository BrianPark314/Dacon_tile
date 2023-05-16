#-*- coding:utf-8-sig -*-

import pandas as pd
import numpy as np
from common import eda
from glob import glob
import os
import random
import torch

def check(base_dir): #처리 과정에서 오류가 없었는지 간단하게 확인
    train_df, encoder = eda.get_train()
    processed_train = len(glob(str(base_dir)+'/_processed_train/*'))
    
    if len(train_df) != processed_train: 
        print('Error in processing train data.')
        return None

    processed_test_len = len(glob(str(base_dir)+'/_processed_test/*'))
    test_len = len(glob(str(base_dir)+'/test/*'))

    if processed_test_len != test_len: 
        print('Error in processing test data.')
        return None

    print('No error detected.')
    return None

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def submission(preds, path, model_name):
    tests = pd.read_csv(path / 'test.csv',index_col='id')
    list_names = list(tests.index.values)
    df = pd.DataFrame(list(zip(list_names, preds)), columns=['id','label'])
    df.to_csv(path / f'{model_name}.csv', index=False, encoding='utf-8-sig')
    return None

