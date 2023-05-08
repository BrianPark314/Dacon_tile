from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
import easydict
from PIL import ImageFilter 
from PIL import ImageEnhance
import time
import os

from PIL import Image
from PIL import ImageOps
from tqdm import tqdm

args = easydict.EasyDict()
args.base_dir = './data/'
args.encoder = {}
args.imsize = 256
args.enhanceparam = 10.0

def get_train(): #train data 불러오기
    train_folder = glob(args.base_dir + 'train/*')
    train_path = []
    for folder in train_folder:
        tmp = glob(folder + '/*')
        train_path += tmp

    train_df = pd.DataFrame(train_path, columns=['path'])
    train_df['label'] = train_df['path'].apply(lambda x: x.split('/')[-2])
    labels = train_df.label.unique()
    number = list(range(len(labels)))
    encoder = dict(zip(labels, number))
    return train_df, encoder #train data의 path와 label이 담긴 dataframe과 인코더 반환
    
def process_image(im, imsize, enhanceparam): #image를 인풋으로 받아 각종 필터 적용 후 이미지 리턴
    im = square_pad(im)
    im = im.resize((imsize, imsize))
    enhancer = ImageEnhance.Sharpness(im)
    im = enhancer.enhance(enhanceparam)
    im = im.filter(ImageFilter.BLUR)
    im = im.filter(ImageFilter.DETAIL)
    im = im.filter(ImageFilter.EDGE_ENHANCE)
    return im

def square_pad(im):
    desired_size = max(im.size)
    delta_w = desired_size - im.size[0]
    delta_h = desired_size - im.size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(im, padding)
    return new_im

def save_result(im, n, label): #각각의 파일 읽어오기
    im = im.save(args.base_dir+'_processed_train/'+f'{n}_{label}.png')
    return None

def load_train(train_path):
    path_train = list(train_path.glob('*/*'))
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

def data_flip(saved_path, im, n):
    try:
        for i in range(2):
            if i == 0:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)  #좌우
                save(saved_path, im,i,n)
            elif i == 1:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)  #좌우
                save(saved_path, im,i,n)
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
    

