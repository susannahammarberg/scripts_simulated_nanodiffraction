# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:12:41 2018

@author: Sanna
"""

import numpy as np

import matplotlib.pyplot as plt

plt.close("all") # close all plotting windows
# Displacement theorem with box function
########################################
# box width
a = 6
x_start = 45
x = np.linspace(-5, 10, 101)
g = np.zeros(len(x),dtype=complex)
g2 = np.copy(g)
# displacement between boxes
x0 = 3
g[x_start:x_start+a] = 1 
g2[x_start+x0:x_start +a +x0] = 1
plt.figure()
plt.plot(x,g)
plt.plot(x,g2)
plt.title(' 2 box functions, displaced by by x_0=' + str(x0))
plt.legend(('g1','g2'))

Fg_num = np.fft.fftshift(np.fft.fft(g))
Fg2_num = np.fft.fftshift(np.fft.fft(g2))

plt.figure()
plt.plot((Fg_num))
plt.plot((Fg2_num))
plt.title('fft of box functions')
plt.legend(('F(g1)','F(g2)'))
plt.show()


q = np.linspace(0,len(Fg_num),len(Fg_num))
Fg2_an = a/2*np.sinc(a*q)

q=8*np.pi/(4*a)
plt.figure()
plt.plot((Fg_num*np.exp(1j*q*x0)))
plt.plot((Fg2_num))
plt.plot(Fg2_an)
plt.title('F(g2) is the same as F(g)*exp(iqx0) ?')
plt.legend(('F(g2)','F(g1)*exp(iqx0)','hh'))
plt.show()


del x,g,g2,x0
#################################################################
a = 2
A = 6+1j*3    # complex number
#displacement u 
a_WZ = 5.93E-10   #WZ lattice constant
# 
d = a_WZ / np.sqrt(4)

c_ZB = 5.869 *1E-10
u = c_ZB/3

x = np.linspace(-5, 10, 101)
g = np.zeros(len(x),dtype=complex)
g2 = np.copy(g)
g3 = np.copy(g)
g[45:50] = 1 
g2[50:55] = 1*np.exp(-1j*0.2*np.pi*u/d)

g3[20:50] = 1
g3[50:70] = 1*np.exp(-1j*0.2*np.pi*u/d)

Fg_num = np.fft.fftshift(np.fft.fft(g))
Fg2_num = np.fft.fftshift(np.fft.fft(g2))
Fg3_num = np.fft.fftshift(np.fft.fft(g3))

Fg_g2_num = np.fft.fftshift(np.fft.fft(g+g2))
# write down anlytical ffts
Fg_an = a*A*np.sinc(a*x)
Fg2_an = A*np.sinc(x+u)
Fg2_an = a*A*np.sinc(a*x)*np.exp(-1j*2)

#f = np.sinc(arr)
I12 = abs(Fg_g2_num)**2
I3 = abs(Fg3_num)**2

plt.figure()
plt.plot(x,g)
plt.plot(x,g2)
plt.title(' 2 box functions displaced, right box phase shifted')

plt.figure()
#plt.plot(x,g)
plt.plot(x,np.angle(Fg_g2_num))

plt.figure()
plt.plot(np.log10(I3))
plt.title('Fourier tranfor of sum of box functions')

plt.figure()
plt.plot(x,g+g2)
plt.figure()
plt.plot(abs(Fg_num))
plt.plot(abs(Fg_num*np.conj(Fg2_num)))
plt.title('fft of g and g2 separately')

plt.figure()
plt.plot(abs(Fg_g2_num*np.conj(Fg_g2_num)))
plt.title('fft of g and g2 together')

# här plotta vad du får analytiskt
plt.figure()
plt.plot(x,Fg + Fg2)
plt.title('Asinc(qa) + Ae$^{i\phi}$sin(qa)')


