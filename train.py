import models
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json

from engine import train, validate
from dataset import ImageDataset
from torch.utils.data import DataLoader

matplotlib.style.use('ggplot')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.model(pretrained=True, requires_grad=False)

lr = 0.0001
epochs = 20
batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# train_csv = pd.read_csv('dataset/train.csv')
recipes = json.load(open('dataset/layer1.json', 'r'))

train_data = ImageDataset(recipes, train=True, test=False)
valid_data = ImageDataset(recipes, train=False, test=False)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

train_loss = []
valid_loss = []
for epoch in range(epochs):
  print(f'Epoch {epoch + 1} of epochs')
  
  train_epoch_loss = train(model, train_loader, optimizer, criterion, train_data, device)
  valid_epoch_loss = validate(model, valid_loader, criterion, valid_data, device)
  
  train_loss.append(train_epoch_loss)
  valid_loss.append(valid_epoch_loss)

  print(f'Train Loss: {train_epoch_loss:.4f}')
  print(f'Val Loss: {valid_epoch_loss:.4f}')

torch.save({
  'epoch': epochs,
  'model_state_dict': model.state_dict(),
  'optimizer_state_dict': optimizer.state_dict(),
  'loss': criterion,
},'../outputs/model.pth')

plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../outputs/loss.png')
plt.show()
