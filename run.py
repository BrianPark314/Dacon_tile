import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm
import random
import cv2
import eda
import time
import easydict
import gc

args = easydict.EasyDict()
args.base_dir = './data/'
args.encoder = {}
args.imsize = 256
args.enhanceparam = 10.0

def __main__():
    start = time.time()
    print('='*50)
    train_df , encoder = eda.get_train()
    print('Data load complete!')
    print('Now processing image...')
    for i in tqdm(range(len(train_df))):
        im = Image.open(train_df.loc[i, 'path'])
        im = eda.process_image(im, args.imsize, args.enhanceparam)
        eda.save_result(im, i, encoder[train_df.loc[i,'label']])
        gc.collect()
    end = time.time()
    print(f'Run complete in {np.round(end-start, 3)} seconds!')
    print('='*50)
    return None

def __check__():

    return None

__main__()