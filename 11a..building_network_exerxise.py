# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 04:33:25 2019

Exercise: Create a network with 

784 input units,
10 output layer - softmax
hidden layer with 128 units
ReLU activation
hidden layer with 64 units
ReLU activation
output layer with a softmax activation 
use a ReLU activation with the nn.ReLU module or F.relu function.
loss layer??

@author: Claudia
"""
from __future__ import print_function
#import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(784, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(128, 10)
    
   #forward pss through the network, returns the output logits    
    def forward(self, x):
        
        # 2 Hidden layer with ReLU activation
        x = self.hidden(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        #Output layer with softmax activation
        x = self.output(x)
        x = F.softmax(x, dim=1)
       
        return x
model = Network()
print(model)

"""
Network(
  (hidden): Linear(in_features=784, out_features=128, bias=True)
  (hidden2): Linear(in_features=784, out_features=64, bias=True)
  (output): Linear(in_features=128, out_features=10, bias=True)
)
"""