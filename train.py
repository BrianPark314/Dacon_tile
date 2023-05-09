#-*- coding:utf-8 -*-

import torch
import common.engine as eng
from common import load_data
from torch import nn
from tqdm.auto import tqdm
from common.params import args
import os

torch.manual_seed(42) #파이토치 시드 고정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'Current device is: {device}')

def prep():
    model = args.model
    print(f'Pytorch {model.__class__.__name__} loaded with pre-trained parameters')

    model.to(device)
    
    train_data, validation_data = load_data.get_train_data(args.BATCH_SIZE,
                                                          args.path, 
                                                          'train',
                                                          args.transform,
                                                          )
    print('Data preperation complete.')

    print('='*50)
    return model, train_data, validation_data

def go(model, train_data, validation_data):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    from timeit import default_timer as timer 
    start_time = timer()

    # Train model
    print("Now training model...")
    model_results = eng.train(model=model, 
                        train_dataloader=train_data,
                        test_dataloader=validation_data,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=args.NUM_EPOCHS, 
                        device=device, 
                        desired_score=args.desired_score)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    return model, model_results

if __name__ == '__main__':
    print('='*50)
    model, train_data, validation_data= prep()
    model, results = go(model, train_data, validation_data)
    print('Saving model...')
    isExist = os.path.exists(args.base_path / f'models/trained_models/{model.__class__.__name__}.pth')
    if not isExist:
        os.makedirs(args.base_path / f'models/trained_models/{model.__class__.__name__}.pth')
    torch.save(model.state_dict(), args.base_path / f'models/trained_models/{model.__class__.__name__}.pth')
    print('Model saved!')
    print('Run complete.')
    print('='*50)