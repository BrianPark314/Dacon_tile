#-*- coding:utf-8-sig -*-

import torch
import common.engine as eng
from common import load_data
from torch import nn
from common.params import args
from common.utils import seed_everything
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'Current device is: {device}')

def go(model, train_data, validation_data, label):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size = 5,
                                                gamma = 0.75)
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
                        patience=args.patience,
                        device=device, 
                        lr_scheduler=lr_scheduler,
                        label=label)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    return model, model_results

if __name__ == '__main__':
    seed_everything(args.seed)
    torch.cuda.empty_cache()

    model = args.model
    print(f'Pytorch {model.__class__.__name__} loaded with pre-trained parameters')

    model.to(device)
    
    train_data, validation_data, label = load_data.get_train_dataloader(args.BATCH_SIZE,
                                                          args.path, 
                                                          args.transform,
                                                          )

    print('Data preperation complete.')
    print('='*50)
    model, results = go(model, train_data, validation_data, label)
    print('Saving model...')

    isExist = os.path.exists(args.base_path / 'models/trained_models')
    if not isExist:
        os.makedirs(args.base_path / f'models/trained_models/{model.__class__.__name__}.pth')
    torch.save(model.state_dict(), args.base_path / f'models/trained_models/{model.__class__.__name__}.pth')
    print('Model saved!')
    print('Run complete.')
    print('='*50)