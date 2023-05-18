#-*- coding:utf-8-sig -*-


from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from sklearn.model_selection import train_test_split
from common.params import args
import torch

def get_kfold_dataloader(BATCH_SIZE, path, transform: transforms):

    kfold_dataloader = 0

    return kfold_dataloader

def get_train_dataloader(BATCH_SIZE, path, transform: transforms):
    train_dir = path / "_processed_train/"
    train_data = datasets.ImageFolder(train_dir, transform)
    targets = train_data.targets
    class_counts = torch.unique(torch.tensor(targets), return_counts=True)
    print(class_counts)
    label = train_data.class_to_idx
    train_idx, valid_idx= train_test_split(
        np.arange(len(targets)),
        test_size=0.3,
        shuffle=True,
        stratify=targets,
        random_state=args.seed)
    print(f"Creating DataLoaders with batch size {BATCH_SIZE}.")

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler = train_idx, num_workers=0)
    valid_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler = valid_idx, num_workers=0)

    return train_dataloader, valid_dataloader, label

def get_test_dataloader(BATCH_SIZE, path, transform_test: transforms):
    train_dir = path / "_processed_train/"
    label = datasets.ImageFolder(train_dir).class_to_idx
    test_dir = path / "_processed_test/"
    test_data = datasets.ImageFolder(test_dir, transform_test)
    test_dataloader = DataLoader(test_data, batch_size = BATCH_SIZE, num_workers=0)
    
    return test_dataloader, label