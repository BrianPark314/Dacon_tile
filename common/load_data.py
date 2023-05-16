#-*- coding:utf-8-sig -*-

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pathlib
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
from PIL import Image


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    return classes, class_to_idx

class CustomImageFolder(Dataset):
    def __init__(self, targ_dir: str, mode: str, transform=None) -> None:
        self.mode = mode
        if self.mode == 'train':
            self.paths = list(pathlib.Path(targ_dir).glob('*/*'))
        else:
            self.paths = list(pathlib.Path(targ_dir).glob('*')) 

        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.paths[index]).convert('RGB')
        if self.mode == 'test': return self.transform(img)
        else:
            class_name  = self.paths[index].parent.name 
            if self.transform is not None: return self.transform(img), class_name 
            else: return img, class_name

def get_train_dataloader(BATCH_SIZE, path, mode, transform: transforms, split_size=[0.7, 0.3]):
    train_dir = path / "_processed_train/"
    train_data = CustomImageFolder(train_dir, mode, transform)
    label = train_data.class_to_idx
    new_train_data , validation_data = torch.utils.data.random_split(train_data, split_size)
    print(f"Creating DataLoaders with batch size {BATCH_SIZE}.")
    train_dataloader = DataLoader(new_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    return train_dataloader, valid_dataloader, label

def get_test_dataloader(path, mode, transform_test: transforms):
    test_dir = path / "_processed_test/"
    print(test_dir)
    test_data = CustomImageFolder(test_dir, mode, transform_test)
    test_dataloader = DataLoader(test_data, batch_size = 100, shuffle=False, num_workers=0)
    
    return test_dataloader