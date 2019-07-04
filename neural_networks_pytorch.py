# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:51:33 2019

Neural networks with PyTorch
@author: Claudia
"""
# Import necessary packages

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

#import numpy as np
import torch

#import helper

#import matplotlib.pyplot as plt

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#64 images - 1 



