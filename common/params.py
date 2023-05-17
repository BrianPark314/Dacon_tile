#-*- coding:utf-8-sig -*-

import easydict
from torchvision import transforms
from pathlib import Path
import models.model_classes as mds
import os
from torchvision.transforms.autoaugment import AutoAugmentPolicy

args = easydict.EasyDict()
args.BATCH_SIZE = 128
args.NUM_EPOCHS = 100
args.desired_score = 0.85
args.imsize = 256
args.enhanceparam = 10.0
args.sharpnessfactor = 1.5
args.patience = 10
args.delta = 0.005
args.lr = 0.001

if os.path.exists("/content/gdrive/MyDrive/project/Dacon_tile/data/") : args.path = Path("/content/gdrive/MyDrive/project/Dacon_tile/data/")
else: args.path = Path("./data")

args.base_path = Path(".")
args.transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

args.transform_test = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

args.model = mds.EfficientNet()

args.seed = 41