import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def scatter_comsol222(raw_data,step,title):
    fig = plt.figure()
    ax = fig.add_subplot( projection='3d')
    sc=ax.scatter(raw_data[::step,0],raw_data[::step,1],raw_data[::step,2], c=raw_data[::step,3], marker ='o', cmap='jet')#,alpha=1)
   
    plt.title( title + '. Every %d:th point'%step)
    plt.colorbar(sc)#; plt.axis('scaled')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')



raw_data = np.load(r'C:\Users\Sanna\Documents\Simulations\save_simulation\raw_comsol_data.npy')
scatter_comsol222(raw_data,step=20, title = 'Displacement from comsol')
plt.show()