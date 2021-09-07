# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:12:51 2017

@author: Sanna
"""
#from IPython import get_ipython
#get_ipython().magic('reset -sf')   #removes all variables saves
import sys   #to collect system path ( to collect function from another directory)
sys.path.insert(0, r'C:\Users\Sanna\Documents\CDI_2017\CXI\Shrinkwrap')

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
#from create3Dgaussian import create3Dgaussian
#from Shrinkwrap3D import shrinkwrap3D 
#from shrinkwrap import shrinkwrap 
from math import ceil
from scipy import misc

plt.close("all") # close all plotting windows


#displacement u 
a_WZ = 5.93E-10   #WZ lattice constant
# 
d = a_WZ / np.sqrt(4)

c_ZB = 5.869 *1E-10
u = c_ZB/3

################################################
Nx = 256
Ny = 256

nx = 100
ny = 60

obj = np.zeros((Ny,Nx),dtype=complex)
# give obj amplitude
obj[ Ny/2 - ny/2: Ny/2 + ny/2, Nx/2 - nx/2: Nx/2 + nx/2 ] = 1
# give obj phase of right half of object   (arb phase 1.5pi)
obj[ Ny/2 - ny/2: Ny/2 + ny/2, Nx/2:Nx/2 + nx/2 ]=1*np.exp((1j*2*np.pi*u/d)) 

plt.figure()
plt.subplot(121)
plt.imshow(abs(obj), cmap = 'gray' )
plt.title('obj amp')
plt.colorbar()
plt.subplot(122)
plt.imshow((np.angle(obj)), cmap = 'jet' )
plt.title('obj phase')
plt.colorbar()

plt.figure()
plt.imshow(abs(obj)*np.angle(obj), cmap = 'jet' )
plt.title('ampl masked phase of obj')
plt.colorbar()

# propagate the exit field
prop_exitWave = fft.fftshift(fft.fftn(obj)) 
# intensity from this complex amplitude is:
I = abs(prop_exitWave)**2

plt.figure()
plt.imshow(((I)),cmap='jet')
plt.title('Diffraction from object')
plt.colorbar()
