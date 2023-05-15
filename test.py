#-*- coding:utf-8 -*-

import torch
import common.engine as eng
import torchinfo
from common.params import args
from common import load_data, utils
from common.utils import seed_everything

if __name__ == '__main__':
    seed_everything(args.seed)
    print('='*50)
    test_data = load_data.get_test_dataloader(args.path, 'test', args.transform_test)
    print('Loading model...')
    model = args.model
    model.load_state_dict(torch.load(args.base_path / f'models/trained_models/{model.__class__.__name__}.pth', map_location=torch.device('cpu')))
    model.eval()
    print('Model load sucessful.')
    torchinfo.summary(model)
    label = load_data.CustomImageFolder(args.path / 'train', 'train').class_to_idx
    print('Generating results...')
    preds = eng.inference(model, test_data, label)
    utils.submission(preds, args.path, model.__class__.__name__)
    print('Run complete.')
    print('='*50)