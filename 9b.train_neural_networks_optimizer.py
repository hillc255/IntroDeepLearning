# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 04:23:57 2019

@author: Claudia

Build multilayer network that utilizes 
log-softmax = output activation function
calculate loss using negative log likelihood loss

Optimizer with TRAINING PASS 
epochs = 5 - loop

"""

import torch
from torch import nn

from torch import optim

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

optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # TODO: Training pass
        
        # clear optimizer - zero out gradients
        optimizer.zero_grad()
        
        # Take a forward and backward step
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        
        #optimize next step
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
"""
result:
    
Training loss: 1.864908663830015
Training loss: 0.8012151913220948
Training loss: 0.5108497013796621
Training loss: 0.4240125921139839
Training loss: 0.38151977397104314

Network is trained so now check its predictions

"""

# %matplotlib inline  - python does not recovnize this so must use lines below
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
#see above

import helper

images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logits = model.forward(img)

# Output of the network are logits, need to take softmax for probabilities
ps = F.softmax(logits, dim=1)
helper.view_classify(img.view(1, 28, 28), ps)

