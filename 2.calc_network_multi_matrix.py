# -*- coding: utf-8 -*-
"""
Calculate output of single layer neural network using multiplication matrix

"""
from __future__ import print_function

import torch
def activation(x):
    return 1/(1+torch.exp(-x))

torch.manual_seed(7) # set random seed - predictible

features = torch.randn((1,5)) #generated random data - tensor 1 row - 5 columns
#print(features)  #o with standard diviation 1

weights = torch.randn_like(features) #creates another tensor like features
#print(weights)                       # values like normal distribution

bias = torch.randn((1, 1)) # signal value from normal distribution
#print(bias)

#size of tensor matrix - use tensor.shape

b = weights.shape
print(b)

#multiplying features and weights - weights only is changed
e = torch.mm(features, weights.view(5,1))
print(e)

f = torch.matmul(features, weights.view(5,1))
print(f)

#calculate the output of the network using multiplication matrix
# unchanged matrix first and no sum necessary
result_a = activation(torch.mm(features, weights.view(5,1)) + bias) 
print(result_a)

result_b = activation(torch.matmul(features, weights.view(5,1)) + bias) 
print(result_b)

result_c = activation((e) + bias)
print(result_c)

#solution: y = activation(torch.mm(features, weights.view(5,1)) + bias)
#b = torch.Size([1, 5])
#e = tensor([[-1.9796]])
#f = tensor([[-1.9796]])
#result_a, b, c =tensor([[0.1595]])

    
