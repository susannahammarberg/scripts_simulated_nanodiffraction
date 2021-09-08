# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 12:22:27 2021

@author: Sanna
"""

import numpy as np
import matplotlib.pyplot as plt
#from ptypy.core import View, Container, Storage, Base
import os
import sys

sys.path.append(r"C:\Users\Sanna\Documents\Beamtime\NanoMAX_May2020\scripts")

from plotPtypyResults import plot2drecons



date_saved = 20210908
openpath = r'C:\Users\Sanna\Documents\Simulations\save_simulation\recons\%s'%date_saved

savepath = r'C:\Users\Sanna\Documents\Simulations\save_simulation\plots\%s'%date_saved

probe = np.load(openpath + '_probe.npy') 
obj = np.load(openpath + '_object.npy', allow_pickle=True) 
x = np.squeeze(np.load(openpath + '_x.npy')) 
y = np.squeeze(np.load(openpath + '_y.npy')) 
z = np.squeeze(np.load(openpath + '_z.npy')) 

#TODO which coordinate, x,y,z?
psize = x[1,0,0]-x[0,0,0]

extent = 1e6 * np.array([0,(obj.shape[1]-1)*psize, 0, (obj.shape[0]-1)*psize])



plot2drecons(obj.T, probe, extent, savepath, save=False)


