#-*- coding:utf-8 -*-

import torch
import engine as eng
import torchinfo
import customImageFolder as cif
from constants import args
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
    print(label)
    print('Generating results...')
    preds = eng.inference(model, test_data, label)
    eng.submission(preds, args.path, model.__class__.__name__)
    print('Run complete.')
    print('='*50)