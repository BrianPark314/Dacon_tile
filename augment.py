import os
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import easydict
from PIL import ImageFilter 
from PIL import ImageEnhance
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import random
import cv2 
import matplotlib.pyplot as plt
import torchvision.transforms as T

args = easydict.EasyDict()
args.base_dir = str(Path('data/'))+'/'
train_dir = Path(args.base_dir +'train/')

def load_train():
    train_path = list(train_dir.glob('*/*'))
    labels_list = [os.path.split(os.path.split(path)[0])[1] for path in train_path]
    
    train_df = pd.DataFrame(train_path,columns=['path'])
    train_df['label'] = labels_list
    
    labels = train_df.label.unique()
    idx = list(range(len(labels)))
    encoder = dict(zip(labels ,idx))

    return train_df, encoder

#path = os.path.join(root_dir,'_augment_')

# def save(path, img, type, n, label):  
#     '''
#     type은 전처리 종류 
#     '''
#     if os.path.isdir(path) != True:
#         os.mkdir(path)

#     saved_name = os.path.join(path,f'{type}{n}_{label}.png')
#     img.save(saved_name)

# def data_flip(saved_dir, img, type, label, n, saving_enable=True):
#     '''
#     type
#     0 - 좌우
#     1 - 상하
#     '''
#     try:
#         if type == 0:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#         elif type == 1:
#             img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
#         if saving_enable == True:
#             save(saved_dir, img, type, n, label)
    
#     except Exception as e:
#         print(e)
#         return "Failed"
    

# def data_rotate(saved_dir, img, degree, label, n, saving_enable=True):
#     img = img.rotate(degree)
#     try:       
#         if saving_enable == True:
#             save(saved_dir, img, degree, n, label)
#         return "Success"
#     except Exception as e:
#         print(e)
#         return "Failed"

def save(path, img, type, n):  
    '''
    type은 전처리 종류 
    '''
    if os.path.isdir(path) != True:
        os.mkdir(path)

    saved_name = os.path.join(path,f'{type}_{n}.png')
    img.save(saved_name)
    
def data_rotate(saved_dir, img, n):
    try:
        for angle in [90,180,270]:
            img = img.rotate(angle)
            #if saving_enable == True:
            save(saved_dir, img, angle, n)
        return "Success"
    except Exception as e:
        print(e)
        return "Failed"

def data_flip(saved_dir, img, n):
    '''
    type
    0 - 좌우
    1 - 상하
    '''
    try:
        for i in range(2):
            if i == 0:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                save(saved_dir, img, i, n)
            elif i == 1:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                save(saved_dir, img, i, n)
        return "Success"
    
    except Exception as e:
        print(e)
        return "Failed"




            