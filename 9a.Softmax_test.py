# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 05:33:20 2019

Softmax calculation
@author: Claudia

#Softmax calculations
import math

z = [ 1.2667, 5.6707, 9.5552,  -0.1100,  -1.3262,  17.0689, 1.2136,   7.9216, -20.6596,  11.8460]
z_exp = [math.exp(i) for i in z]
print([round(i, 2) for i in z_exp])
sum_z_exp = sum(z_exp)
print(round(sum_z_exp, 2))
softmax = [i / sum_z_exp for i in z_exp]
print([round(i, 3) for i in softmax])

"""