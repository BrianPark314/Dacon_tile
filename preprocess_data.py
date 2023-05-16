#-*- coding:utf-8 -*-

from glob import glob
from PIL import Image
from tqdm import tqdm
import common.eda as eda
import time
import gc
from pathlib import Path
from common.params import args
from common import load_data

def process_train(path, imsize, enhanceparam): #train 데이터 준비
    start = time.time()
    print('='*50 + '\n')
    train_data_custom = load_data.CustomImageFolder(path / 'train', mode = 'train') #train 데이터 custom imagefolder 사용해서 로드
    print('Train data loaded' + '\n')
    print('Processing Image...' + '\n')
    
    for i in tqdm(range(len(train_data_custom))): #학습 데이터 처리
        im, label = train_data_custom.__getitem__(i)
        im = eda.process_image(im, imsize, enhanceparam)
        eda.save_processed_result(path, im, i, label)
        gc.collect()

    end = time.time()
    print(f'Run complete in {int(end-start)} seconds.' + '\n')
    print('='*50)
    return None

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

if __name__ == '__main__':
    start = time.time()
    process_train(args.path, args.imsize, args.enhanceparam)
    process_test(args.path, args.path / 'test/', args.imsize, args.enhanceparam)
    end = time.time()
    print(f'Total runtime is {int(end-start)} seconds.')
