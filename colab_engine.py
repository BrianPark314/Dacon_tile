import torch
from torch import nn
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import gc
import glob
import requests
import zipfile
from pathlib import Path
from sklearn.metrics import f1_score
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = list(image_paths.glob('*'))
        self.transform = transform
        
    def get_class_label(self, image_name):
        name = image_name.split('.')[0]
        label = name.split('_')[-1]
    
        return int(label)
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path).convert('RGB')
        y = self.get_class_label(str(image_path).split('/')[-1])

        if self.transform is not None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.image_paths)

def get_data(BATCH_SIZE, path, transform: transforms):
    data_path = path
    train_dir = data_path / "_processed_train"
    test_dir = data_path / "_processed_test"
    
    train_data = MyDataset(train_dir, transform)
    test_data = MyDataset(test_dir, transform)

    new_train_data , validation_data = torch.utils.data.random_split(train_data, [0.7, 0.3])

    print(f"Creating DataLoaders with batch size {BATCH_SIZE}.")

    # Create DataLoader's
    train_dataloader = DataLoader(new_train_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        )
    valid_dataloader = DataLoader(validation_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        )

    test_dataloader = DataLoader(test_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        )

    return train_dataloader, valid_dataloader, test_dataloader

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int, device,
          desired_score):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        'train_f1':[],
        "test_loss": [],
        "test_acc": [],
        'test_f1':[]
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        model, train_loss, train_acc, train_f1 = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc, test_f1 = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"train_f1: {train_f1:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"test_f1: {test_f1:.4f} | "
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_f1"].append(train_f1)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_f1"].append(test_f1)
        gc.collect()

        if test_f1 > desired_score:
            print('Desired f1 score reached, early stopping')
            return model, results
    # 6. Return the filled results at the end of the epochs
    return model, results

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    # Put model in train mode
    model.train()
    # Setup train loss and train accuracy values
    train_loss, train_acc, train_f1 = 0, 0, 0 
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        y_preds = y_pred.argmax(1).detach().cpu().numpy().tolist()
        y_labels = y.detach().cpu().numpy().tolist()

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        #f1_score
        f1 = f1_score(y_labels, y_preds, average='weighted')
        train_f1 += f1.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        gc.collect()

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    train_f1 = train_f1 / len(dataloader)

    return model, train_loss, train_acc, train_f1

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc, test_f1 = 0, 0, 0 
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            y_preds = test_pred_logits.argmax(1).detach().cpu().numpy().tolist()
            y_labels = y.detach().cpu().numpy().tolist()

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            #f1_score
            f1 = f1_score(y_labels, y_preds, average='weighted')
            test_f1 += f1.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    test_f1 = test_f1 / len(dataloader)

    return test_loss, test_acc, test_f1

def inference(model, test_loader, label):
    model.eval()
    model.to('cpu')
    preds = []
    with torch.no_grad():
        for imgs, _ in tqdm(iter(test_loader)):
            imgs = imgs.to('cpu')
            pred = model(imgs)
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    new_preds = preds.copy()
    for i, x in enumerate(preds):
        new_preds[i] = [k for k, v in label.items() if v == x][0]
    return new_preds

def submission(preds, path, model_name):
    tests = pd.read_csv(path / 'test.csv',index_col='id')
    list_names = list(tests.index.values)
    df = pd.DataFrame(list(zip(list_names, preds)), columns=['id','label'])
    df.to_csv(path / f'{model_name}.csv', index=False, encoding='utf-8')
    return None

