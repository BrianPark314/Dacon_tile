import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm
import random
import cv2
import eda
import time
import easydict
import gc

args = easydict.EasyDict()
args.base_dir = './data/'
args.encoder = {}
args.imsize = 256
args.enhanceparam = 10.0

eda.get_train()