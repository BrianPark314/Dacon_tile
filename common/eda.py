#-*- coding:utf-8-sig -*-

import pandas as pd
from PIL import ImageFilter 
from PIL import ImageEnhance
import os
from pathlib import Path
from PIL import Image
from PIL import ImageOps
from tqdm import tqdm
from torchvision import datasets
import torch
from common import load_data
    
def square_pad(im):
    desired_size = max(im.size)
    delta_w = desired_size - im.size[0]
    delta_h = desired_size - im.size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(im, padding)
    return new_im

def process_image(im, imsize, enhanceparam): #image를 인풋으로 받아 각종 필터 적용 후 이미지 리턴
    im = square_pad(im)
    im = im.resize((imsize, imsize))
    enhancer = ImageEnhance.Sharpness(im)
    im = enhancer.enhance(enhanceparam)
    im = im.filter(ImageFilter.BLUR)
    im = im.filter(ImageFilter.DETAIL)
    im = im.filter(ImageFilter.EDGE_ENHANCE)
    return im

def save_processed_result(path:str, im, n, label):
    isExist = os.path.exists(path / f'_processed_train/{label}')
    if not isExist:
        os.makedirs(path / f'_processed_train/{label}')    
    im.save(path / f'_processed_train/{label}/{n}.png')


def load_train(train_path): #이것도 삭제해야 하는데 엄두가 안남 하하하하하
    path_train = list(train_path.glob('*/*'))
    labels = [os.path.split(os.path.split(name)[0])[1] for name in path_train]
    
    train_df = pd.DataFrame(path_train,columns=['path'])
    train_df['label']= labels
    
    label = train_df['label'].unique()
    number = list(range(len(label)))
    encoder = dict(zip(label, number))

    return train_df, encoder 

def data_flip(save_path, im, n):
    try:
        for i in range(2):
            if i == 0:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)  #좌우
                im.save(os.path.join(save_path, f'{i}_{n}.png'))
            elif i == 1:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)  #좌우
                im.save(os.path.join(save_path, f'{i}_{n}.png'))
        return "Success"
    
    except Exception as e:
        print(e)
        return "Failed"
    
def data_rotate(save_path, im, n):
    try:
        for angle in [90,180,270]:
            im = im.rotate(angle)
            im.save(os.path.join(save_path, f'{angle}_{n}.png'))
        return "Success"
    except Exception as e:
        print(e)
        return "Failed"
    
def aug_data(train_path):
    try:
        aug_data_custom = datasets.ImageFolder(train_path)
        class_counts = (torch.unique(torch.tensor(aug_data_custom.targets), return_counts=True))
        print(class_counts)
        lb = [int(class_counts[0][i]) for i in range(len(class_counts[0])) if class_counts[1][i] < 400]
        
        label_dict = aug_data_custom.class_to_idx

        for i in tqdm(range(len(aug_data_custom))): #학습 데이터 처리
            im, label = aug_data_custom.__getitem__(i)
            if label in lb:
                path = Path(os.path.join(train_path, [k for k, v in label_dict.items() if v == label][0])) 
                data_flip(path,im,i)
                data_rotate(path,im,i)

        print(torch.unique(torch.tensor(datasets.ImageFolder(train_path).targets), return_counts=True))
        return "Success"
    
    except Exception as e:
        print(e)
        return "Failed"

