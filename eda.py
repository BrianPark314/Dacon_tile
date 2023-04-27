from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
import easydict
from PIL import ImageFilter 
from PIL import ImageEnhance
import time

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
