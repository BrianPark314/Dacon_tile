import torch
import torchvision
import time
import colab_engine as eng
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

torch.manual_seed(42) #파이토치 시드 고정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'Current device is: {device}')
args = easydict.EasyDict()
args.model_name = 'googlenet'
args.BATCH_SIZE = 64
args.NUM_EPOCHS = 100
args.desired_score = 0.75
args.path = Path("/content/gdrive/MyDrive/project/Dacon_tile/data/")
#args.path = Path("/Users/Shark/Projects/Dacon_tile/data")

args.transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class ClassifierModule(nn.Module):
    def __init__(self):
        super(ClassifierModule,self).__init__()
        self.layer1 = nn.Linear(1000,512)
        self.Relu1 = nn.ReLU()
        self.Dropout1 = nn.Dropout(p=0.4)
        self.layer2 = nn.Linear(512, 256)
        self.Relu2 = nn.ReLU()
        self.Dropout2 = nn.Dropout(p=0.4)
        self.layer3 = nn.Linear(256, 19)
        self.net = torchvision.models.googlenet(weights = torchvision.models.GoogLeNet_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad=False

    def forward(self,x):
        return self.layer3(self.Dropout2(self.Relu2(self.layer2(self.Dropout1(self.Relu1(self.layer1(self.net(x))))))))

def prep():
    print('='*50)
    model = ClassifierModule()
    print(f'Pytorch {args.model_name} loaded with pre-trained parameters')

    model.to(device)
    
    train_data, validation_data, test_data = eng.get_data(args.BATCH_SIZE, args.path, args.transform)
    print('Data preperation complete.')

    print('='*50)
    return model, train_data, validation_data, test_data

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
    model, train_data, validation_data, test_data = prep()
    #model, results = go(model, train_data, validation_data)
    #print('Saving model...')
    #torch.save(model.state_dict(), args.path / f'models/{args.model_name}.pth')
    #print('Model saved!')
    model = ClassifierModule()
    model.load_state_dict(torch.load(args.path / f'models/{args.model_name}.pth', map_location=torch.device('cpu')))
    model.eval()
    label = cif.ImageFolderCustom(args.path / 'train').class_to_idx
    print('Generating results...')
    preds = eng.inference(model, test_data, label)
    eng.submission(preds, args.path, args.model_name)
    print('Run complete.')
    print('='*50)


