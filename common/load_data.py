#-*- coding:utf-8-sig -*-


from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.append(os.getcwd())
from common.params import args
import torch

def make_weights(labels, nclasses):
    labels = np.array(labels)   # where, unique 함수를 사용하기 위해 numpy로 변환한다.
    weight_list = []   # 가중치를 저장하는 배열을 생성한다.
 
    for cls in range(nclasses):
        idx = np.where(labels == cls)[0]
        count = len(idx)    #각 클래스 데이터 수 카운트 
        weight = 1/count    
        weights = [weight] * count    #라벨이 뽑힐 가중치를 1/count로 동일하게 전체 라벨에 할당 
        weight_list += weights
    
    return weight_list
    
    
def get_train_dataloader(BATCH_SIZE, path, transform: transforms):
    train_dir = path / "_processed_train/"
    train_data = datasets.ImageFolder(train_dir, transform)
    targets = train_data.targets
    _, weights = torch.unique(torch.tensor(targets), return_counts=True)
    weights = weights/len(train_data)

    label = train_data.class_to_idx
    train_idx, valid_idx= train_test_split(
        np.arange(len(targets)),
        test_size=0.15,
        shuffle=True,
        stratify=targets,
        random_state=args.seed)
    
    split_train = torch.utils.data.Subset(train_data, train_idx)
    split_valid = torch.utils.data.Subset(train_data, valid_idx)

    tr_label = [train_data.targets[i] for i in train_idx]
    #val_label = [train_data.targets[i] for i in valid_idx]

    tr_weights = torch.DoubleTensor(make_weights(tr_label,len(train_data.classes)))
    #val_weights = torch.DoubleTensor(make_weights(val_label,len(train_data.classes)))

    tr_sampler = torch.utils.data.sampler.WeightedRandomSampler(tr_weights, len(tr_weights))
    #val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_weights, len(val_weights))
    
    print(f"Creating DataLoaders with batch size {BATCH_SIZE}.")

    train_dataloader = DataLoader(split_train, batch_size=BATCH_SIZE, sampler = tr_sampler, num_workers=0)
    valid_dataloader = DataLoader(split_valid, batch_size=BATCH_SIZE, num_workers=0)

    return train_dataloader, valid_dataloader, label, weights

def get_test_dataloader(BATCH_SIZE, path, transform_test: transforms):
    train_dir = path / "_processed_train/"
    label = datasets.ImageFolder(train_dir).class_to_idx
    test_dir = path / "_processed_test/"
    test_data = datasets.ImageFolder(test_dir, transform_test)
    test_dataloader = DataLoader(test_data, batch_size = BATCH_SIZE, num_workers=0)
    
    return test_dataloader, label