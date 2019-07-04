# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 04:23:57 2019

@author: Claudia

Build multilayer network that utilizes 
log-softmax = output activation function
calculate loss using negative log likelihood loss

"""

import torch
from torch import nn

import torch.nn.functional as F
from torchvision import datasets, transforms


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# TODO: Build a feed-forward network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10


model = nn.Sequential(    
                      nn.Linear(input_size, hidden_sizes[0]),  
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),   
                      nn.LogSoftmax(dim=1)          
         )

# TODO: Define the loss for nn.LogSoftmax
criterion = nn.NLLLoss() 

### Run this to check your work
# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)
"""
result: tensor(2.3069)

"""