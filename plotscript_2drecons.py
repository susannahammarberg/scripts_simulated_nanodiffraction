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

import matplotlib
matplotlib.use( 'Qt5agg' )

#%%

date_saved = 20210913
projection = 31+30 
itstr = 'iter100' 

save = True

openpath = r'C:\Users\Sanna\Documents\Simulations\save_simulation\recons\%s_projection%i'%(date_saved,projection)

savepath = r'C:\Users\Sanna\Documents\Simulations\save_simulation\plots\%s_projection%i_%s'%(date_saved,projection,itstr)

probe = np.load(openpath + '\\' + 'probe_%s.npy'%itstr) 
obj = np.load(openpath + '\\'  + 'object_%s.npy'%itstr, allow_pickle=True) 
x = np.squeeze(np.load(openpath + '\\' + 'x_%s.npy'%itstr)) 
y = np.squeeze(np.load(openpath + '\\' + 'y_%s.npy'%itstr)) 
z = np.squeeze(np.load(openpath + '\\' + 'z_%s.npy'%itstr)) 
#errors = np.load(openpath+ '\\errors.npy')
ferrors = np.load(openpath+ '\\ferrors.npy')


#TODO which coordinate, x,y,z?
psize = x[1,0,0]-x[0,0,0]

extent = 1e6 * np.array([0,(obj.shape[1]-1)*psize, 0, (obj.shape[0]-1)*psize])


if save == True:
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print('new folder in this savepath was created')


plot2drecons(np.fliplr(obj.T), probe, extent, savepath, save)

#plt.figure()
#plt.plot(errors,'blue')
#plt.plot(abs(ferrors),'red')

    
