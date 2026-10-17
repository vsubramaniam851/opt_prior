import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataloader import generalization_cifar_test

def eval_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def cifar_finetune(exp_name, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    for name, param in model.named_parameters():
        if not name.startswith('fc.'):
            param.requires_grad = True
        else:
            param.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = generalization_cifar_test()
    accs = []
    ckpt_count = 0
    for epoch in range(300):
        epoch_losses = []
        model.train()
        for i, batch in enumerate(tqdm(train_loader, desc = 'Iterating over CIFAR')):
            x, y = batch
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        print(f'Epoch {epoch+1}: train loss = {np.mean(epoch_losses):.3f}')
        acc = eval_accuracy(model, test_loader, device)
        accs.append(acc)
        print(f'Epoch {epoch+1}: test acc = {acc:.3f}')
    return accs

if __name__ == '__main__':
    model = torchvision.models.resnet18(pretrained = True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)

    accs = cifar_finetune('cifar_rn18', model)
    print(max(accs))