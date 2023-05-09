#-*- coding:utf-8 -*-

import common.engine as eng
from torch import nn
import easydict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from pathlib import Path
import models.model_classes as mds

args = easydict.EasyDict()
args.BATCH_SIZE = 128
args.NUM_EPOCHS = 1
args.desired_score = 0.75
args.imsize = 256
args.enhanceparam = 10.0
args.sharpnessfactor = 1.5
args.encoder = {}

args.colab_path = Path("/content/gdrive/MyDrive/project/Dacon_tile/data/")
args.path = Path("./data")
args.base_path = Path(".")
args.transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

args.transform_test = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

args.model = mds.GoogleNet()