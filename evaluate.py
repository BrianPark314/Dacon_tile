#-*- coding:utf-8 -*-

import torch
from torch import nn
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchmetrics
import gc
import glob
import requests
import zipfile
from pathlib import Path
from sklearn.metrics import f1_score
import pandas as pd
import colab_googlenet as ggt
import colab_engine as eng
import customImageFolder as cif

if __name__ == '__main__':
    print('='*50)
    _, train_data, validation_data, test_data = ggt.prep()
    model = ggt.ClassifierModule()
    model.load_state_dict(torch.load(ggt.args.path / f'models/{ggt.args.model_name}.pth', map_location=torch.device('cpu')))
    model.eval()
    label = cif.ImageFolderCustom(ggt.args.path / 'train').class_to_idx
    print('Generating results...')
    preds = eng.inference(model, test_data, label)
    eng.submission(preds)
    print('='*50)
