#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch-dl
Created by raj at 10:20 
Date: February 09, 2020	
"""
import torch
from spacy.cli import train
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import FashionMNIST
import torch.utils.data

from models.utils.fconv_layer_norm import LayerNormConv2d


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, depth=2, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.norm_conv1 = LayerNormConv2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=2 * out_channels, kernel_size=kernel_size)
        self.norm_conv2 = LayerNormConv2d(2 * out_channels)

        # Compute output dimension after convolution/max-pooling (4 * 4 in self.fc1), Oh = (n -f - 2p)/s + 1
        self.fc1 = nn.Linear(2 * out_channels * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 60)
        self.out = nn.Linear(60, 10)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormConv2d(6)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.norm_conv1(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm_conv2(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout(x)

        # Flatten the tensor for linear layer
        x = x.reshape(-1, 12 * 4 * 4)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.out(x)

        return x


def get_num_correct(pred, labels):
    return pred.argmax(dim=1).eq(labels).sum()


train_data = FashionMNIST(
    root="../data/FashionMNIST",
    download=True,
    train=True,
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()
         ])
)

valid_data = FashionMNIST(
    root="../data/FashionMNIST",
    download=True,
    train=False,
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()
         ])
)

# image, label = next(iter(train_data))
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
validation_loader = DataLoader(valid_data, batch_size=100)
cnn = CNN()
optimizer = Adam(params=cnn.parameters(), lr=0.01)

for i in range(50):
    total_loss = 0
    total_correct = 0.0
    for batch in train_loader:
        images, labels = batch

        pred = cnn(images)
        loss = F.cross_entropy(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += get_num_correct(pred, labels)
    print(total_loss, total_correct)
    print(total_correct / len(train_data))
    with torch.no_grad():
        total_correct = .0
        for batch in validation_loader:
            images, labels = batch
            pred = cnn(images)
            total_correct += get_num_correct(pred, labels)
        print(total_correct)
        print(total_correct/len(valid_data))

