#%%
import torch
import torchvision
import time
import engine
from torch import nn
import easydict
import torchinfo
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import plot

torch.manual_seed(42) #파이토치 시드 고정
#device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (f"PyTorch version:{torch.__version__}") # 1.12.1 이상
print(f"MPS device built: {torch.backends.mps.is_built()}") # True 여야 합니다.
print(f"MPS device available: {torch.backends.mps.is_available()}") # True 여야 합니다. 

args = easydict.EasyDict()
args.BATCH_SIZE = 256
args.NUM_EPOCHS = 3
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
    
    train_data, validation_data, test_data = engine.get_data(args.BATCH_SIZE, args.transform)
    print('Data preperation complete.')

    print('='*50)
    return model, train_data, validation_data, test_data

def go(model, train_data, validation_data):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    from timeit import default_timer as timer 
    start_time = timer()

    # Train model
    model_results = engine.train(model=model, 
                        train_dataloader=train_data,
                        test_dataloader=validation_data,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=args.NUM_EPOCHS, device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    return model_results

if __name__ == '__main__':
    model, train_data, validation_data, test_data = prep()
    results = go(model, train_data, validation_data)
    plot.plot_loss_curves(results)
    plot.plot_f1_curves(results)


# %%
