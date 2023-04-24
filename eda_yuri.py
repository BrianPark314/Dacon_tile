import os
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import easydict
from PIL import ImageFilter 
from PIL import ImageEnhance

from PIL import Image
from tqdm import tqdm
import random
import cv2

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

def get_test(): #테스트 데이터 불러오기
    test_path = glob(args.base_dir + 'test/')
    print(test_path)
    return None

def process_image(im, imsize, enhanceparam): #image를 인풋으로 받아 각종 필터 적용 후 이미지 리턴
    im = im.resize((imsize, imsize))
    enhancer = ImageEnhance.Sharpness(im)
    im = enhancer.enhance(enhanceparam)
    im = im.filter(ImageFilter.BLUR)
    im = im.filter(ImageFilter.DETAIL)
    im = im.filter(ImageFilter.EDGE_ENHANCE)
    return im

def check_value_counts(df):
    df['label'].value_counts()
    
    
def enhance_data(im):
    im 

    return None

def save_result(im, n, label): #각각의 파일 읽어오기
    im = im.save(args.base_dir+'_processed_train/'+f'{n}_{label}.png')
    return None
