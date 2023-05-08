#-*- coding:utf-8 -*-

from glob import glob
import pandas as pd
import numpy as np
from customImageFolder import ImageFolderCustom as ifc
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import eda
import os
import time
import gc
from pathlib import Path
from constants import args

def aug_data(train_path):
    try:
        df,enc = eda.load_train(train_path)
        idx = df['label'].value_counts().index
        lb = df['label'].value_counts().values < 60
        lb = idx[lb].tolist()
        aug_data_custom = ifc(targ_dir=train_path)

        print('='*50 + '\n')
        print('Now augmenting train data...' + '\n')

        for i in tqdm(range(len(aug_data_custom))): #학습 데이터 처리
            im, label = aug_data_custom.load_image(i)
            if label in lb:
                path = Path(os.path.join(train_path, label)) 
                eda.data_flip(path,im,i)
                eda.data_rotate(path,im,i)
        return lb
    
    except Exception as e:
        print(e)
        return "Failed"

def process_train(train_path, imsize, enhanceparam): #train 데이터 준비
    start = time.time()
    print('='*50 + '\n')
    train_data_custom = ifc(targ_dir=train_path) #train 데이터 custom imagefolder 사용해서 로드
    print('Train data loaded' + '\n')
    print('Processing Image...' + '\n')
    
    for i in tqdm(range(len(train_data_custom))): #학습 데이터 처리
        im, label = train_data_custom.load_image(i)
        im = eda.process_image(im, imsize, enhanceparam)
        eda.save_result(im, i, train_data_custom.class_to_idx[label])
        gc.collect()

    end = time.time()
    print(f'Run complete in {int(end-start)} seconds.' + '\n')
    print('='*50)
    return train_data_custom.class_to_idx

def process_test(base_path, test_path, imsize, enhanceparam): #테스트 데이터 준비
    start = time.time()
    print('='*50 + '\n')
    print('Now processing test data...' + '\n')
    total_length = len(glob(str(test_path / '*')))
    for x in tqdm(Path(test_path).iterdir(), total = total_length):
        if str(x).split('/')[-1][0] == '.': continue
        im = Image.open(x)
        im = eda.process_image(im, imsize, enhanceparam)

        x = str(x)
        name, ext = x.split('.')[0][-3:], x.split('.')[1]
        im.save(str(base_path) + '/_processed_test/' + f'{name}' + '.' + f'{ext}')
    end = time.time()
    print(f'Run complete in {int(end-start)} seconds.' + '\n')
    print('='*50)
    return None

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

if __name__ == '__main__':
    start = time.time()
    aug_data(args.path / 'train/')
    class_to_idx = process_train(args.path / 'train/', args.imsize, args.enhanceparam)
    process_test(args.path, args.path / 'test/', args.imsize, args.enhanceparam)
    check(args.path) 
    end = time.time()
    print(f'Total runtime is {int(end-start)} seconds.')
