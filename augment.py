from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
import easydict
from PIL import ImageFilter 
from PIL import ImageEnhance
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

args = easydict.EasyDict()
args.base_dir = './data/'
train_dir = Path(args.base_dir + 'train/')

def load_train():
    path_train = list(train_dir.glob('*/*'))
    labels = [os.path.split(os.path.split(name)[0])[1] for name in path_train]
    
    train_df = pd.DataFrame(path_train,columns=['path'])
    train_df['label']= labels
    
    label = train_df['label'].unique()
    number = list(range(len(label)))
    encoder = dict(zip(label, number))

    return train_df, encoder 
                    
def save(keyPath, im, type,n):
    '''
    처리된 이미지 저장
    '''
    saved_name = os.path.join(keyPath , f'{type}_{n}.png')
    im = im.save(saved_name)

def data_flip(saved_dir, im, n):
    try:
        for i in range(2):
            if i == 0:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)  #좌우
                save(saved_dir, im,i,n)
            elif i == 1:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)  #좌우
                save(saved_dir, im,i,n)
        return "Success"
    
    except Exception as e:
        print(e)
        return "Failed"
    
def data_rotate(saved_dir, im, n):
    try:
        for angle in [90,180,270]:
            im = im.rotate(angle)
            save(saved_dir,im,angle,n)
        return "Success"
    except Exception as e:
        print(e)
        return "Failed"
    

