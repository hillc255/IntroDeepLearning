# -*- coding: utf-8 -*-
"""
1. Calculate output of stack -neural network of multiple neurons
2. Calculate Softmax
𝑦=𝑓2(𝑓1(𝑥⃗ 𝐖1)𝐖2)
"""
#Stack nural network

from __future__ import print_function

import torch

def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))


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
# output = activation(torch.mm(h, W2) + B2)
# print(output)