#-*- coding:utf-8 -*-

import pandas as pd
from common import eda
from glob import glob
import os

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

def submission(preds, path, model_name):
    tests = pd.read_csv(path / 'test.csv',index_col='id')
    list_names = list(tests.index.values)
    df = pd.DataFrame(list(zip(list_names, preds)), columns=['id','label'])
    df.to_csv(path / f'{model_name}.csv', index=False, encoding='utf-8')
    return None

