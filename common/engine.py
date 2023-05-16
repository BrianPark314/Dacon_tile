#-*- coding:utf-8-sig -*-

import torch
from tqdm.auto import tqdm
import gc
from sklearn.metrics import f1_score
import numpy as np
from common.params import args


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=args.delta, path= args.base_path / 'models/trained_models/checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss[-1]

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss[-1]:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss[-1]

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int, 
          patience: int,
          device,
          desired_score,
          label
          ):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        'train_f1':[],
        "valid_loss": [],
        "valid_acc": [],
        'valid_f1':[]
    }
    early_stopping = EarlyStopping(patience = patience, verbose = True)

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        model, train_loss, train_acc, train_f1 = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                           label=label)
        test_loss, test_acc, test_f1 = valid_step(model=model,
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
            + '\n' +
            f"valid_loss: {test_loss:.4f} | "
            f"valid_acc: {test_acc:.4f} | "
            f"valid_f1: {test_f1:.4f} | "
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_f1"].append(train_f1)
        results["valid_loss"].append(test_loss)
        results["valid_acc"].append(test_acc)
        results["valid_f1"].append(test_f1)
        gc.collect()

        early_stopping(results["valid_loss"], model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        model.load_state_dict(torch.load(args.base_path / 'models/trained_models/checkpoint.pt'))
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
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        y_labels = y.detach().cpu().numpy().tolist()
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        loss.backward()
        optimizer.first_step(zero_grad=True)
        loss_fn(model(X), y).backward()
        optimizer.second_step(zero_grad=True)

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

def valid_step(model: torch.nn.Module, 
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
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            y_labels = y.detach().cpu().numpy().tolist()
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1).detach().cpu()
            
            test_acc += ((test_pred_labels == y.detach().cpu()).sum().item()/len(test_pred_labels))
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
        for imgs, _ in tqdm(iter(test_loader)):
            imgs = imgs.to('cpu')
            pred = model(imgs)
            preds += pred.softmax(dim=1).argmax(1).detach().cpu().numpy().tolist()
    new_preds = preds.copy()
    for i, x in enumerate(preds):
        new_preds[i] = [k for k, v in label.items() if v == x][0]
    return new_preds
