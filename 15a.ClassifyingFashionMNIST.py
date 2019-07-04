"""
Created on Mon Jun 26 2019

@author: Claudia

#Part 4 - Fashion-MNIST

#define network
# image 28 x 28 = 784 pixels
# 10 classes
# at least 1 hidden layer
# ReLU activation
# return logits or log-softmax from the forward pass
# layers and size?

"""
#load dataset 
import torch
#from torch import nn
import torch.nn as nn
#from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

"""
see one image in the data set

image, label = next(iter(trainloader))
helper.imshow(image[0,:]);

"""



#create the model
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logits = model(images)

print(logits)
#loss = criterion(logits, labels)

"""
#See images

image, label = next(iter(trainloader))
helper.imshow(image[0,:]);

"""