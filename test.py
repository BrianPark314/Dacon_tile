from PIL import Image
import glob
from pathlib import Path

data_path = Path("data/")
train_dir = data_path / "_processed_train"
l = list(train_dir.glob('*'))
for x in l:
    i = Image.open(x)
    