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
torch.manual_seed(42) #파이토치 시드 고정
#device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'Current device is: {device}')
args = easydict.EasyDict()
args.BATCH_SIZE = 128
args.NUM_EPOCHS = 10
args.path = Path("/content/gdrive/MyDrive/project/Dacon_tile/data/")
args.transform = transforms.Compose([ 
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

class ClassifierModule(nn.Module):
    def __init__(self):
        super(ClassifierModule,self).__init__()
        self.layer1 = nn.Linear(1000,19)
        self.net = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad=False

    def forward(self,x):
        return (self.layer1(self.net(x)))

def prep():
    print('='*50)
    model = ClassifierModule()
    print('Pytorch vgg16 loaded with pre-trained parameters')

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

def inference(model, test_loader, device):
    le = preprocessing.LabelEncoder()
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds

if __name__ == '__main__':
    model, train_data, validation_data, test_data = prep()
    results = go(model, train_data, validation_data)



