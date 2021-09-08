# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 12:22:27 2021

@author: Sanna
"""

import numpy as np
import matplotlib.pyplot as plt
#from ptypy.core import View, Container, Storage, Base
import os

from plotPtypyResults import plot2drecons


date_saved = 20210908
openpath = r'C:\Users\Sanna\Documents\Simulations\save_simulation\recons\%s'%date_saved

savepath = r'C:\Users\Sanna\Documents\Simulations\save_simulation\plots\%s'%date_saved

probe = np.load(openpath + '_probe.npy') 
obj = np.load(openpath + '_object.npy', allow_pickle=True) 
x = np.squeeze(np.load(openpath + '_x.npy')) 
y = np.squeeze(np.load(openpath + '_y.npy')) 
z = np.squeeze(np.load(openpath + '_z.npy')) 




def plot2drecons(obj, probe, extent):


