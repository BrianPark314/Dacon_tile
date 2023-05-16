#-*- coding:utf-8-sig -*-


from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from sklearn.model_selection import train_test_split


def get_train_dataloader(BATCH_SIZE, path, transform: transforms):
    train_dir = path / "_processed_train/"
    train_data = datasets.ImageFolder(train_dir, transform)
    targets = train_data.targets
    label = train_data.class_to_idx
    train_idx, valid_idx= train_test_split(
        np.arange(len(targets)),
        test_size=0.3,
        shuffle=True,
        stratify=targets)
    print(f"Creating DataLoaders with batch size {BATCH_SIZE}.")

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler = train_idx, num_workers=0)
    valid_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler = valid_idx, num_workers=0)

    return train_dataloader, valid_dataloader, label

def get_test_dataloader(path, transform_test: transforms):
    test_dir = path / "_processed_test/"
    print(test_dir)
    test_data = datasets.ImageFolder(test_dir, transform_test)
    test_dataloader = DataLoader(test_data, batch_size = 100, num_workers=0)
    
    return test_dataloader