#-*- coding:utf-8 -*-

import torch
from torch import nn
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import gc
from sklearn.metrics import f1_score
import pandas as pd
import os
import chardet
import pathlib
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
from PIL import Image


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    classes = [x.encode('utf-8').decode('utf-8') for x in classes]
    # 2. Raise an error if class names not found
        
    # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

# 1. Subclass torch.utils.data.Dataset
class CustomImageFolder(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, mode: str, transform=None) -> None:
        self.mode = mode
        # 3. Create class attributes
        # Get all image paths
        if self.mode == 'train':
            self.paths = list(pathlib.Path(targ_dir).glob('*/*')) # note: you'd have to update this if you've got .png's or .jpeg's
        else:
            self.paths = list(pathlib.Path(targ_dir).glob('*')) # note: you'd have to update this if you've got .png's or .jpeg's

        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Tuple[Image.Image, str]:
        "Opens an image via a path and returns it."
        return Image.open(self.paths[index]).convert('RGB')
        
    def __process__(self, img) -> Image.Image:
        img = self.transform(img)
        return img
    
    def _getlabel__(self, dir: str) -> str:
        self.dir = dir.apply(lambda x: x.split('/')[-2])

        return self.dir
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        if self.mode == 'test': return self.transform(img)
        else:
            class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
            class_idx = self.class_to_idx[class_name]
            if self.transform:
                return self.transform(img), class_idx # return data, label (X, y)
            else:
                return img, class_idx # return data, label (X, y)

def get_train_data(BATCH_SIZE, path, mode, transform: transforms):
    train_dir = path / "_processed_train/"
    train_data = CustomImageFolder(train_dir, mode, transform)
    new_train_data , validation_data = torch.utils.data.random_split(train_data, [0.7, 0.3])
    print(f"Creating DataLoaders with batch size {BATCH_SIZE}.")
    train_dataloader = DataLoader(new_train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, valid_dataloader

def get_test_data(path, mode, transform_test: transforms):
    test_dir = path / "_processed_test/"
    print(test_dir)
    test_data = CustomImageFolder(test_dir, mode, transform_test)
    test_dataloader = DataLoader(test_data, batch_size = 100, shuffle=False)
    
    return test_dataloader