from glob import glob
import pandas as pd
import numpy as np
from customImageFolder import ImageFolderCustom as ifc
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import random
import cv2
import eda
import augment
import time
import easydict
import gc
import torchvision.transforms as T
from pathlib import Path
import os

args = easydict.EasyDict()
args.base_dir = str(Path('data/'))+'/'
train_dir = Path(args.base_dir +'train/')
args.encoder = {}
args.imsize = 256
args.enhanceparam = 10.0
args.sharpnessfactor = 1.5

train_df, enc = augment.load_train()

def aug_data():
    df, enc = augment.load_train()
    idx = df['label'].value_counts().index
    aug_label = idx[df['label'].value_counts().values < 60].tolist()
    aug_data_custom = ifc(targ_dir=train_dir)
    print('='*50 + '\n')
    print('Now augmenting train data...' + '\n')

    for i in tqdm(range(len(aug_data_custom))): #학습 데이터 처리
        im, label = aug_data_custom.load_image(i)
        idx = aug_data_custom.class_to_idx[label]
        save_dir = Path(os.path.join('data/train/' + label))
        if label in aug_label:
            augment.data_rotate(save_dir,im,i),
            augment.data_flip(save_dir,im,i)
    return None

def prep_data(): #train 데이터 준비
    start = time.time()
    print('='*50 + '\n')
    train_data_custom = ifc(targ_dir=train_dir) #train 데이터 custom imagefolder 사용해서 로드
    print('Train data loaded' + '\n')
    print('Processing Image...' + '\n')
    
    for i in tqdm(range(len(train_data_custom))): #학습 데이터 처리
        im, label = train_data_custom.load_image(i)
        im = eda.process_image(im, args.imsize, args.enhanceparam)
        eda.save_result(im, i, train_data_custom.class_to_idx[label])
        gc.collect()

    end = time.time()
    print(f'Run complete in {int(end-start)} seconds.' + '\n')
    print('='*50)
    return None

def process_test(imsize, enhanceparam): #테스트 데이터 준비
    start = time.time()
    print('='*50 + '\n')
    print('Now processing test data...' + '\n')
    test_path = args.base_dir + 'test/'
    total_length = len(glob(test_path+'*'))
    for x in tqdm(Path(test_path).iterdir(), total = total_length):
        if str(x).split('/')[-1][0] == '.': continue
        im = Image.open(x)
        im = eda.process_image(im, imsize, enhanceparam)

        x = str(x)
        name, ext = x.split('.')[0][-3:], x.split('.')[1]
        im.save(args.base_dir+'_processed_test/'+f'{name}'+'.'+f'{ext}')
    end = time.time()
    print(f'Run complete in {int(end-start)} seconds.' + '\n')
    print('='*50)
    return None

def check(): #처리 과정에서 오류가 없었는지 간단하게 확인
    train_df, encoder = eda.get_train()
    processed_train = len(glob(args.base_dir+'_processed_train/*'))
    
    if len(train_df) != processed_train: 
        print('Error in processing train data.')
        return None

    processed_test_len = len(glob(args.base_dir+'_processed_test/*'))
    test_len = len(glob(args.base_dir+'test/*'))

    if processed_test_len != test_len: 
        print('Error in processing test data.')
        return None

    print('No error detected.')
    return None

if __name__ == '__main__':
    start = time.time()
    aug_data()
    prep_data()
    process_test(args.imsize, args.enhanceparam)
    check()
    end = time.time()
    print(f'Total runtime is {int(end-start)} seconds.')
