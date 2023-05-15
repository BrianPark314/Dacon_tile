#-*- coding:utf-8 -*-

import torch
from tqdm.auto import tqdm
import gc
from sklearn.metrics import f1_score

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int, device,
          desired_score,
          label):
    
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
                                           device=device,
                                           label=label)
        test_loss, test_acc, test_f1 = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            label=label)
        
        # 4. Print out what's happening
        print('\n'+
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
               device: torch.device,
               label: dict):
    # Put model in train mode
    model.train()
    # Setup train loss and train accuracy values
    train_loss, train_acc, train_f1 = 0, 0, 0 
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        y = torch.tensor([label[x] for x in y])
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        y_labels = y.detach().cpu().numpy().tolist()
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        f1 = f1_score(y_labels, y_pred_class.detach().cpu().numpy().tolist(), average='weighted')
        train_f1 += f1.item()
        gc.collect()

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    train_f1 = train_f1 / len(dataloader)

    return model, train_loss, train_acc, train_f1

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              label: dict):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc, test_f1 = 0, 0, 0 
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            y = torch.tensor([label[x] for x in y])
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            y_labels = y.detach().cpu().numpy().tolist()
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1).detach().cpu()
            
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            f1 = f1_score(y_labels, test_pred_labels, average='weighted')
            test_f1 += f1.item()

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
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.to('cpu')
            pred = model(imgs)
            preds += pred.softmax(dim=1).argmax(1).detach().cpu().numpy().tolist()
    new_preds = preds.copy()
    for i, x in enumerate(preds):
        new_preds[i] = [k for k, v in label.items() if v == x][0]
    return new_preds
