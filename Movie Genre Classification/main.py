from model import Model
from dataset import MyDataset
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD


train_path='dataset/Genre Classification Dataset/train_data.txt'
dataset=MyDataset(train_path)

device='cuda'
criterion=nn.CrossEntropyLoss()

model=Model()
optimiser=Adam(model.parameters(), lr=1e-4)

dataloader=DataLoader(dataset=dataset, batch_size=2, shuffle=False)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        optimiser.zero_grad()  
        outputs = model(inputs) 
        loss = criterion(outputs, labels)
        loss.backward()  
        optimiser.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

print('Training complete')
