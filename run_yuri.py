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

root_dir = './data/'
train_dir = Path(root_dir + 'train/')
test_dir = Path(root_dir + 'test/')

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


