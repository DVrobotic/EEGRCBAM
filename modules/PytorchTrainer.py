import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, cohen_kappa_score

from modules.utils import setup_tensors, get_minibatches


def train_model(model, X, y, batchsize=32, lr=0.001, iterations=2000, device='cuda', weight_decay=0.01):
    model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tensor, y_tensor = setup_tensors(X, y, device)

    for it in range(iterations):
        X_temp, y_temp = get_minibatches(X_tensor, y_tensor, batchsize)

        pred, loss = model(X_temp, y_temp)

        print('loss: %.4f, iter: %d' % (loss, it), end='\r')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_model_with_tracking(model, X, y, batchsize=32, lr=0.001, iterations=2000, device='cuda', weight_decay=0.01):
    model.to(device=device)
    writer = SummaryWriter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_copy = X.copy()
    y_copy = y.copy()

    kfold = len(X_copy)//5

    order = np.array([i for i in range(len(X_copy))])
    np.random.shuffle(order)

    X_validation = X_copy[order][:kfold]
    y_validation = y_copy[order][:kfold]
    X_copy = X_copy[order][kfold:]
    y_copy = y_copy[order][kfold:]

    X_copy_tensor, y_copy_tensor = setup_tensors(X_copy, y_copy, device)
    X_validation_tensor, y_validation_tensor = setup_tensors(X_validation, y_validation, device)

    for it in range(iterations):
        X_temp, y_temp = get_minibatches(X_copy_tensor, y_copy_tensor, batchsize)
        X_validation_temp, y_validation_temp = get_minibatches(X_validation_tensor, y_validation_tensor, batchsize)

        pred, loss = model(X_temp, y_temp)
        acc = accuracy_score(y_temp.detach().cpu().numpy(), pred.argmax(1).detach().cpu().numpy())
        kappa = cohen_kappa_score(y_temp.detach().cpu().numpy(), pred.argmax(1).detach().cpu().numpy())

        with torch.no_grad():
            pred_validation, loss_validation = model(X_validation_temp, y_validation_temp)
            acc_validation = accuracy_score(y_validation_temp.detach().cpu().numpy(), pred_validation.argmax(1).detach().cpu().numpy())
            kappa_validation = cohen_kappa_score(y_validation_temp.detach().cpu().numpy(), pred_validation.argmax(1).detach().cpu().numpy())

        writer.add_scalar("Loss/train", loss, it)
        writer.add_scalar("Acc/train", acc, it)
        writer.add_scalar("Kappa/train", kappa, it)

        writer.add_scalar("Loss/validation", loss_validation, it)
        writer.add_scalar("Acc/validation", acc_validation, it)
        writer.add_scalar("Kappa/validation", kappa_validation, it)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('loss: %.4f, test loss: %.4f iter: %d' % (loss, loss_validation, it), end='\r')