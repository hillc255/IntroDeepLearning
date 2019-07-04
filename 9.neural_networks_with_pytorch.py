# -*- coding: utf-8 -*-
"""
Neural networks with pytorch

1. Flatten the batch of images: images
2. Build a multi-layer network with below
3. use sigmoid activation for the hidden layer
4. leave output layer without an activation

𝑦=𝑓2(𝑓1(𝑥⃗ 𝐖1)𝐖2)
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)


<class 'torch.Tensor'>
torch.Size([64, 1, 28, 28])
torch.Size([64])

Convert the batch of images with shape (64, 1, 28, 28) 
to a have a shape of (64, 784), 
784 is 28 times 28. 
This is typically called flattening
we flattened the 2D images into 1D vectors.

#multi-layer network
# 784 input units
# 256 hidden units
# 10 output units
# random tensors for weights and biases
# sigmoid activation for h4idden layer
# output layer without activation
# out = output of network with shape (64 x 10)

torch.Size([64, 784])
torch.Size([64])

out = # output of your network, should have shape (64,10)

"""
#solution
from __future__ import print_function

# Import necessary packages
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
#import numpy as np
import torch

#import helper
#import matplotlib.pyplot as plt

### Run this cell
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

## Solution
def activation(x):
    return 1/(1+torch.exp(-x))

# Flatten the input images
inputs = images.view(images.shape[0], -1)

# Create parameters
w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2
print(out)

#Add softmax funtion - probability feature

#softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
def softmax(x):
    ## TODO: Implement the softmax function here  
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

# Here, final_reult should be the output of the network in the previous excercise with shape (64,10)
probabilities = softmax(out)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))


"""
Claudia's attempt

#<class 'torch.Tensor'>
#torch.Size([64, 784])
#torch.Size([64])

#Sigmoid activation function
def activation(x):
    return 1/(1+torch.exp(-x))


features = torch.randn((64,784))
n_input = features.shape[1]
n_hidden = 256
n_output = 10

W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

h = activation(torch.mm(features, W1) + B1)
out = (torch.mm(h, W2) + B2)
print(out)


### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer - 3,2
W1 = torch.randn(n_input, n_hidden)

# Weights for hidden layer to output layer - 2, 1
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))

B2 = torch.randn((1, n_output))

#Calculate the output for this multi-layer network 
# solution for single neural network: y = activation(torch.mm(features, weights.view(5,1)) + bias)

#[1 x 3] and [2 x 1]
result_a = activation(torch.mm(features, W1.view(3,2)) + B1) 
#print(result_a)

final_result = activation(torch.mm(result_a, W2.view(2,1)) + B2)
print(final_result)

#using the weights W1 & W2, and the biases, B1 & B2 - result_a = tensor([[ 0.3171]])
# Class solution
# h = activation(torch.mm(features, W1) + B1)
# output = activation(torch.mm(h, W3) + B2)
# print(output)

"""