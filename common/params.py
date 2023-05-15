#-*- coding:utf-8 -*-

import easydict
from torchvision import transforms
from pathlib import Path
import models.model_classes as mds
import os
from torchvision.transforms.autoaugment import AutoAugmentPolicy

args = easydict.EasyDict()
args.BATCH_SIZE = 128
args.NUM_EPOCHS = 200
args.desired_score = 0.85
args.imsize = 256
args.enhanceparam = 10.0
args.sharpnessfactor = 1.5
args.encoder = {}

if os.path.exists("/content/gdrive/MyDrive/project/Dacon_tile/data/") : args.path = Path("/content/gdrive/MyDrive/project/Dacon_tile/data/")
else: args.path = Path("./data")

args.base_path = Path(".")
args.transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
        transforms.RandomPerspective(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

args.transform_test = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

args.model = mds.EfficientNet()

args.seed = 41