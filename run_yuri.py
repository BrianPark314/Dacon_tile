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
import time
from pathlib import Path
import easydict
import gc
from torchvision import transforms


base_dir = './data/'
path = Path(base_dir +'train/')
test_path = Path(base_dir +'test/')

encoder = {}
imsize = 256
enhanceparam = 10.0
sharpnessfactor = 1.5

#def augment_data():

def prep_data():
    start = time.time()
    print('='*50 + '\n')
    train_data_custom = ifc(targ_dir=path) #train 데이터 custom imagefolder 사용해서 로드
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
    return None

def process_test(imsize, enhanceparam): #테스트 데이터 불러오기
    start = time.time()
    print('='*50 + '\n')
    print('Now processing test data...' + '\n')
    test_path = list(test_path.glob('*.png'))
    for x in tqdm(test_path):
        im = Image.open(x)
        im = eda.process_image(im, imsize, enhanceparam)
        name = x.split('/')[-1]
        print(name)
        im.save(args.base_dir+'_processed_test/'+f'{name}')
    end = time.time()
    print(f'Run complete in {int(end-start)} seconds.' + '\n')
    print('='*50)
    return None

def check():
    train_df, encoder = eda.get_train()
    processed_train = len(Path(base_dir+'_processed_train/').glob('*'))
    
    if len(train_df) != processed_train: 
        print('Error in processing train data.')
        return None

    processed_test_len = len(Path(base_dir+'_processed_test/').glob('*'))
    test_len = len(glob(test_path.glob('*')))

    if processed_test_len != test_len: 
        print('Error in processing test data.')
        return None

    print('No error detected.')
    return None

if __name__ == '__main__':
    start = time.time()
    #prep_data()
    process_test(imsize, enhanceparam)
    check()
    end = time.time()
    print(f'Total runtime is {int(end-start)} seconds.')
