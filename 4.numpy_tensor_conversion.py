# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:17:42 2019

@author: Claudia

PyTorch - converts Numpy arrays and Torch tensors

torch.from_numpy()  = Numpy array to tensor
.numpy() = convert tensor to Numpy array

"""
from __future__ import print_function

import torch
import numpy as np

# display numpy array
a = np.random.rand(4,3)
# print(a)

# convert numpy array to tensor
b = torch.from_numpy(a)
# print(b)

"""
Result
tensor([[0.0872, 0.0706, 0.7043],
        [0.3314, 0.6024, 0.3181],
        [0.4666, 0.3705, 0.2733],
        [0.9656, 0.0271, 0.8174]], dtype=torch.float64)

"""

# convert numpy to tensor array
c = b.numpy()
# print(c)

# multiply pytorch tensor by, in place
print(b.mul_(2))

# numpy array matches new values from Tensor

print(a)