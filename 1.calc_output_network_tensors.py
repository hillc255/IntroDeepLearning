# -*- coding: utf-8 -*-
"""
Calculate output of single layer neural network using weights and bias tensors

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

#calculate the output of thsi network using the weights and bias tensors
result_a = activation(torch.sum(features * weights) + bias)
print(result_a)

result_b = activation((features * weights).sum() + bias)
print(result_b)

#correct answer = tensor([[0.1595]])

    
#result: tensor([[0.3265, 0.6788, 0.9239, 0.3411, 0.5809]])
# result = (activation((torch.sum(weights * features))+ bias))
# print(result) #tensor([[0.1595]])