#-*- coding:utf-8 -*-

import torch
import torchvision
import time
import engine as eng
from torch import nn
import easydict
import torchinfo
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from pathlib import Path
from sklearn import preprocessing
import pandas as pd
import customImageFolder as cif
import glob
from constants import args
import models as mds
import _train

if __name__ == '__main__':
    print('='*50)
    model, train_data, validation_data, test_data = _train.prep()
    print('Loading model...')
    model = args.model
    model.load_state_dict(torch.load(args.path / f'models/{model.__class__.__name__}.pth', map_location=torch.device('cpu')))
    model.eval()
    print('Model load sucessful.')
    print(torchinfo.summary(model))
    label = cif.ImageFolderCustom(args.path / 'train').class_to_idx
    print('Generating results...')
    preds = eng.inference(model, test_data, label)
    eng.submission(preds, args.path, model.__class__.__name__)
    print('Run complete.')
    print('='*50)