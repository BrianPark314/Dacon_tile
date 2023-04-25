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
import easydict
import gc
from torchvision import transforms


args = easydict.EasyDict()
args.base_dir = './data/'
args.train_dir = args.base_dir +'train/'
args.encoder = {}
args.imsize = 256
args.enhanceparam = 10.0
args.sharpnessfactor = 1.5

def prep_data():
    start = time.time()
    print('='*50)
    train_data_custom = ifc(targ_dir=args.train_dir)
    print('Data load complete!')
    print('Now processing image...')
    
    for i in tqdm(range(len(train_data_custom))):
        im, label = train_data_custom.load_image(i)
        im = eda.process_image(im, args.imsize, args.enhanceparam)
        eda.save_result(im, i, train_data_custom.class_to_idx[label])
        gc.collect()

    end = time.time()
    print(f'Run complete in {np.round(end-start, 3)} seconds!')
    print('='*50)
    return None

def __check__():

    return None

if __name__ == '__main__':
    prep_data()