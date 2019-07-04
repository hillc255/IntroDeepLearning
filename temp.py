# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
from __future__ import print_function

import torch
def activation(x):
    return 1/(1+torch.exp(-x))

torch.manual_seed(7) # set random seed - predictible

features = torch.randn((1,5)) #generated random data - tensor 1 row - 5 columns
#print(features)#o with standard diviation 1

weights = torch.randn_like(features) #creates another tensor like features
#print(weights)                       # values like normal distribution

bias = torch.randn((1, 1)) # signal value from normal distribution
#print(bias)

print(activation(( weights + features) + bias ))
    