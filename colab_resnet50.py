import torch
import torchvision
import time
import colab_engine
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
args.BATCH_SIZE = 64
args.NUM_EPOCHS = 30
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
        self.layer2 = nn.Linear(512, 19)
        self.Dropout1 = nn.Dropout(p=0.2)
        self.net = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad=False

    def forward(self,x):
        return self.layer2(self.Dropout1(self.Relu1(self.layer1(self.net(x)))))

def prep():
    print('='*50)
    model = ClassifierModule()
    print('Pytorch resnet50 loaded with pre-trained parameters')

    model.to(device)
    
    train_data, validation_data, test_data = colab_engine.get_data(args.BATCH_SIZE, args.path, args.transform)
    print('Data preperation complete.')

    print('='*50)
    return model, train_data, validation_data, test_data

def go(model, train_data, validation_data):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    from timeit import default_timer as timer 
    start_time = timer()

    # Train model
    model_results = colab_engine.train(model=model, 
                        train_dataloader=train_data,
                        test_dataloader=validation_data,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=args.NUM_EPOCHS, device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    return model_results

def inference(model, test_loader, label, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, _ in tqdm(iter(test_loader)):
            imgs.to(device)
            pred = model(imgs)
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    new_preds = preds.copy()
    for i, x in enumerate(preds):
        new_preds[i] = [k for k, v in label.items() if v == x][0]
    return new_preds

def submission(preds):
    tests = pd.read_csv(args.path / 'test.csv',index_col='id')
    list_names = list(tests.index.values)
    df = pd.DataFrame(list(zip(list_names, preds)), columns=['id','label'])
    df.to_csv(args.path / 'resnet50.csv', index=False)
    return None

if __name__ == '__main__':
    model, train_data, validation_data, test_data = prep()
    model, results = go(model, train_data, validation_data)
    print('Saving model...')
    torch.save(model.state_dict(), args.path / 'models/resnet50.pth')
    print('Model saved!')
    label = cif.ImageFolderCustom(args.path / 'train').class_to_idx
    print('Generating results...')
    preds = inference(model, test_data, label, device)
    submission(preds)
    print('Run complete.')
    print('='*50)


