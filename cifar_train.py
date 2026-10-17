import torch
import torch.optim as optim
import torch.nn as nn

from models import TinyCNN, LargerCNN
from dataloader import cifar_dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = TinyCNN().to(device)
model = LargerCNN().to(device)
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay = 1e-4)
criterion = nn.CrossEntropyLoss()

train_loader, test_loader = cifar_dataset()

def eval_accuracy(model, loader):
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

model.load_state_dict(torch.load('saved_models/larger_cnn_cifar.pt'))
print('%.4f' % eval_accuracy(model, test_loader))

# for epoch in range(10):  # bump up/down as you like
#     model.train()
#     for x, y in train_loader:
#         x, y = x.to(device), y.to(device)
#         opt.zero_grad()
#         loss = criterion(model(x), y)
#         loss.backward()
#         opt.step()
#     acc = eval_accuracy(test_loader)
#     print(f'Epoch {epoch+1}: test acc = {acc:.3f}')
#     torch.save(model, f'{epoch}_tiny_cnn_cifar.pt')
# # torch.save(model.state_dict(), 'saved_models/tiny_cnn_cifar.pt')