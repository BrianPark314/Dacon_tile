from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import os

from PIL import Image
from PIL import ImageOps
from tqdm import tqdm


def load_train():
    root_dir = './data/'
    path = Path(root_dir)
    path_train = list(path.glob('train/*/*'))

    labels = [os.path.split(os.path.split(name)[0])[1] for name in path_train]

    train_df = pd.DataFrame(path_train,columns=['path'])
    train_df['label']= labels
    return train_df
                    
# def augment_data(df):
#     '''label별로 증강 함수 적용'''
#     label_name=[]
#     label_values = df['label'].value_counts()
#     for idx in label_values.index:
#         label_name.append(idx)   #내림차순으로 레이블명 리스트에 저장
#     label_name = label_name[1:]  #훼손 제외

#     for label in label_name:
#         df.loc[(df.label == label),'path']

#     return None

def save(keyPath, file_name, cv_img, rate, type):
    '''
    처리된 이미지 저장
    '''
    if os.path.isdir(keyPath) != True:
        os.mkdir(keyPath)
    
    saved_name = os.path.join(keyPath,"{}{}.{}".format(file_name.split('.')[0], type, 'png'))
    #print(saved_name)
    cv.imwrite(saved_name, cv_img)

def data_flip(saved_dir, data, img, rate, type, saving_enable=False):
    
    img = cv.flip(img, type)
    try:
        if type == 0:
            type = "_horizen_"
        elif type == 1:
            type = "_vertical_"
        elif type == -1:
            type = "_bothFlip_"
        
        if saving_enable == True:
            save(saved_dir, data, img, rate, type)
    
    except Exception as e:
        print(e)
        return "Failed"
    
def data_rotate(saved_dir, data, img, rate, type, saving_enable=False):
    
    xLength = img.shape[0]
    yLength = img.shape[1]
    
    try:
        rotation_matrix = cv.getRotationMatrix2D((xLength/2 , yLength/2), rate, 1)
        img = cv.warpAffine(img, rotation_matrix, (xLength, yLength))
        #print(img.shape)        
        if saving_enable == True:
            save(saved_dir, data, img, rate, type)
        
        return "Success"
    except Exception as e:
        print(e)
        return "Failed"
    

