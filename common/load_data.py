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
    
def get_train_dataloader(BATCH_SIZE, path, transform: transforms):
    train_dir = path / "_processed_train/"
    train_data = datasets.ImageFolder(train_dir, transform)
    targets = train_data.targets
    _, weights = torch.unique(torch.tensor(targets), return_counts=True)
    weights = weights/len(train_data)

    label = train_data.class_to_idx
    train_idx, valid_idx= train_test_split(
        np.arange(len(targets)),
        test_size=0.1,
        shuffle=True,
        stratify=targets,
        random_state=args.seed)

    tr_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_idx))
    val_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(valid_idx))
    
    print(f"Creating DataLoaders with batch size {BATCH_SIZE}.")

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler = tr_sampler, num_workers=0)
    valid_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler = valid_idx, num_workers=0)

    return train_dataloader, valid_dataloader, label, weights

def get_test_dataloader(BATCH_SIZE, path, transform_test: transforms):
    train_dir = path / "_processed_train/"
    label = datasets.ImageFolder(train_dir).class_to_idx
    test_dir = path / "_processed_test/"
    test_data = datasets.ImageFolder(test_dir, transform_test)
    test_dataloader = DataLoader(test_data, batch_size = BATCH_SIZE, num_workers=0)
    
    return test_dataloader, label