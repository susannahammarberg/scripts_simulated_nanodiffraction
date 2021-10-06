#%%
'''
Script for reading COMSOL simulated data and create simulated Bragg diffraction
patterns then perform XRD mapping on those patterns.


This script is copied from bragg3d_NanoMAX_COMSOL_diffraction_sim_recon.py
But I removed the part that does scanning XRD analysis and add the reconstruction
part taken from Alex script demonstraiting Bragg 3D reconstructions in ptypy

    

----------------------------
HOW TO USE
----------------------------
* define your geometry of the experiment in 'geometry' (theta angle for InP/INGaP)
* define your ptychographic measurement (your scanning positions)
  This defines your measurement grid r1r2r3
* load your COMSOL model by entering path, filename and choose displacement
  field u,v or w
* chose what material you want to simulate, the InP or the InGaP segments
* rotate your object into the experiment coordinate system depending on your 
 experiment. Ex: if you are looking at the 111 reflection, your NW long axis
 should be along z
* interpolate your COMSOL data onto the orthogonal grid
* recalculate to phase and mask away the InP or GaInP
...
* load a probe of your choise
...

 
----------------------------
Fixes and TODO
---------------------------
* --> To get a better simulation i should tilt the NW with angle theta
       away from the z-axis, just as in the experiment, right?
* --> can i redo megans plot but using the scattering vectors we are actually using?

* --> abs(phase) is ok but when interpolating the values becopmes strange. abs() is no longer 1. because of the extreme phase I think. 
will it work to interolate the displacemnt field first, and then calculate the phase?

* --> select a domain of the data to do diffraction pattern. now you are making diffraction from 
* different ways to make COMSOL model


import matplotlib
matplotlib.use( 'Qt5agg' )
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
mpl.rcParams['toolbar'] = 'toolbar2'

'''
#%% imports


import ptypy 
from ptypy import utils as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D #needed for 3d scatter plot 

#import mayavi.mlab as mlab
import scipy.interpolate as inter
import sys
sys.path.append(r"C:\Users\Sanna\Documents\Simulations\scripts_simulated_nanodiffraction")
import os

from read_COMSOL_data import rotate_data
from read_COMSOL_data import calc_H111
from read_COMSOL_data import calc_H110
from read_COMSOL_data import calc_complex_obj
from read_COMSOL_data import plot_3d_scatter

import time
date_str = time.strftime("%Y%m%d") # -%H%M%S")


import matplotlib
matplotlib.use( 'Qt5agg' )

#plt.close("all")
#%%
#---------------------------------------------------------
# Set up measurement geometry object
#---------------------------------------------------------

# Define a measurement geometry. The first element of the shape is the number
# of rocking curve positions, the first element of the psize denotes theta
# step in degrees. 'Distance' is to the detector. This will define
# the pixel size in the object plane (g.resolution)
#InP:10.91  GaInP: 11.09  #2017

# Real 2020
g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(62, 256, 256), energy=10.0, distance=1.0, theta_bragg=8.8, propagation = "farfield")  

#REAl 2017
#g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(51, 151, 151), energy=9.49, distance=1.149, theta_bragg=10.91, propagation = "farfield")  

#starting point
#g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(1*1E-2, 55*1E-6, 55*1E-6), shape=(60, 128, 128), energy=20.0, distance=1.0, theta_bragg=10.91, propagation = "farfield") 

# higher Nx Ny
#g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(51, 251, 251), energy=9.49, distance=1.149, theta_bragg=10.91, propagation = "farfield")  
#lower Nx Ny
#g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(51, 81, 81), energy=9.49, distance=1.149, theta_bragg=10.91, propagation = "farfield")  

#g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(1*1E-2, 55*1E-6, 55*1E-6), shape=(60, 128, 128), energy=20.0, distance=1.0, theta_bragg=10.91, propagation = "farfield")  
# thus the FOV in one position is given by 
FOV = g.resolution * g.shape        #obs fov in the coordinate system reciprocal to the natural one thus (q3q1q2)
print( FOV)
print( g.resolution)

#%%
#---------------------------------------------------------
# Create a container for the object and define views based on scaning postions.
# Reformat the container based on the FOV of all views, so that it matches
# the part of the sample that is scanned. 
#---------------------------------------------------------

# Create a container for the object that which will represent the
# object in the non-orthogonal coordinate system (=r3r1r2) conjugate to the
# q-space measurement frame 
obj_container = ptypy.core.Container( ID='Cobj',data_type=np.complex128, data_dims=3)

##%%
## Define scanning positions in x,z,y
#starting point
Ny = 13
Nz = 67

#~real
#Ny = 11
#Nz = 11#    This is hard to tell exactly how many steps we scanned one segment. 

Npos = Nz*Ny
positions = np.zeros((Npos,3))
# stepsize as fed to motors

# starting point 2020 (dont know)
dy_prime = 50.0e-9
dz_prime = 50.0e-9


#starting point 2017
#dy_prime = 20.0e-9
#dz_prime = 20.0e-9


#real
#dy_prime = 40.0e-9
#dz_prime = 30.0e-9

## start this far away from the center point of the wire
dz_center = 0#630E-9

#dz_center = -600E-9

##real positions where beam hits sample
dy = dy_prime
dz = dz_prime
## if the sample is tilted with theta
#dz = dz_prime*g.costheta
#dx = dz_prime*g.sintheta

z_positions = np.repeat( np.linspace(0,dz*(Nz-1),Nz) - (dz*(Nz-1)/2) - dz_center , Ny)
y_positions = np.tile(np.linspace(0,dy*(Ny-1),Ny) - (dy*(Ny-1)/2), Nz)

## also x-positions changes a bit because the sample is tilted with an angle
## theta.  Use Nz here. No! I did not define the sample as being tilted in this 
## simulation. remove x_positions
##x_positions = np.repeat(dx*np.linspace(-np.round(Nz/2),Nz/2,Nz) , Ny)
#
#positions[:,0] = x_positions
positions[:,1] = z_positions
positions[:,2] = y_positions


# For each scan position in the orthogonal coordinate system (x,z,y), find the
# natural coordinates (r3r1r2) and create a View instance there.
# the size of one view is not determined by the size of the beam but the geometries FOV
views = []
for pos in positions:
    pos_ = g._r3r1r2(pos)  # calc the positions in the skewed coordinate system (r3r1r2)
    views.append(ptypy.core.View(obj_container, storageID='Sobj', psize=g.resolution, coord=pos_, shape=g.shape))  # the psize here is the psize in the object plate which is given by g.resolution
        
# this storage is in the natural coordinate system    
obj_storage = obj_container.storages['Sobj']  # define it here so that it is easier to access and put data in the storage
##define grids for a single view. (altough should be able to get the grid from one view later. this should be the same as for one FOV)
# these should be called r1r2r3
xx_vgrid, zz_vgrid, yy_vgrid = g.transformed_grid(obj_storage, input_space='real', input_system='natural')
## flatten the grid for plotting
xx_vgrid = xx_vgrid.flatten().reshape(-1,1)
yy_vgrid = yy_vgrid.flatten().reshape(-1,1)
zz_vgrid = zz_vgrid.flatten().reshape(-1,1)
print( obj_storage.formatted_report()[0]) # here the shape is just the shape of 1 FOV
# reformat the container so that its region is defined by the sum of all views
obj_container.reformat() 
print( obj_storage.formatted_report()[0]) # here you see the shape is bigger in y which is the axis in which we defined the scanning


#%%
#--------------------------------------------------------------
# Make a shifted copy of the object storage and collect the
# orthogonal grid (x,z,y) (to use for COMSOL interpolation)
# ------------------------------------------------------------- 

# make a shifted
# (nearest-neighbor interpolated) copy of the object Storage.
obj_storage_cart = g.coordinate_shift(obj_storage, input_system='natural', input_space='real', keep_dims=True)

# collect the cartesian grid
xx, zz, yy = obj_storage_cart.grids()

# 4D arrays --> 3D arrays
xx = np.squeeze(xx)
yy = np.squeeze(yy)
zz = np.squeeze(zz)


#%%
# ---------------------------------------------------------
# Read in COMSOL model
# -------------------------------------------------------

# define path to COMSOL data
#path = 'C:/Users/Sanna/Documents/COMSOL/COMSOL_data/InGaP_middlesegment_variation/'
path = 'C:/Users/Sanna/Documents/COMSOL/COMSOL_data/'
#path = 


sample = 'full_segmented_NW_InP_InGaP_20191029'     # updated version with strain mismatch 1.5
###sample = 'full_segmented_NW_InP_InGaP_20190828' (including 19 segment)
#sample = '170'   

# choose displacement u,v, or w
uvw = 5 # 3 4 5 = u v w   # for 111 should be 5

# choose domain to plot (or None if file does not have domains )
# TODO only correct for domain 3 or None. For the InGaP it tries to interpolate the values where the InP segment is
domain = 'InP_357911'#3# 'InP_357911' # InP_357911' #'InGaP_24681012'  #'InP_357911'    


if domain == None:
    domain_str = 'All_domains'
    useThesecols = (0,1,2,uvw)
elif domain in ( 3,'InP_357911'):
    domain_str = 'InP'
    useThesecols = (0,1,2,uvw,6)
elif domain in ('InGaP_1245','InGaP_24681012'):
    domain_str = 'InGaP'
    useThesecols = (0,1,2,uvw,6)
else:
    sys.exit('wrong domains')
    
if uvw==3:
    uvw_str = 'u'
elif uvw==4:
    uvw_str = 'v'
elif uvw == 5:
    uvw_str = 'w'

# load the data (coordinates [m] + displacement field [nm] in one coordinate)
file1 = np.loadtxt(path + sample +'.txt',skiprows=9, usecols = useThesecols)
#file1 = np.loadtxt( sample +'.txt',skiprows=9, usecols = useThesecols)

if domain == 3:
    # cut out the domain data 
    raw_data = []
    for row in file1:
        # if its 3 add it, if its not, dont add it
        if np.floor(row[-1]) == domain:
            raw_data.append(row)        
    raw_data = np.asarray(raw_data)
elif domain == 'InGaP_1245':
    # cut out the domain data 
    raw_data = []
    for row in file1:
        # if its not 3, add it
        if np.floor(row[-1]) in (1,2,4,5):
            raw_data.append(row)        
        else:
            # set the unwanted data to 0 (InP segment)
            row[-2] = 0
            raw_data.append(row)
            
    raw_data = np.asarray(raw_data)
elif domain == 'InGaP_24681012':
    # cut out the domain data 
    raw_data = []
    for row in file1:
        # add all the InGaP domains
        if np.floor(row[-1]) in (2,4,6,8,10,12):
            raw_data.append(row)        
        else:
            # set the unwanted data to 0 (InP segment)
            row[-2] = 0
            raw_data.append(row)
            
    raw_data = np.asarray(raw_data)
        
    
elif domain =='InP_357911':
    raw_data = []
    for row in file1:
        # if its not 3, add it
        if np.floor(row[-1]) in (3,5,7,9,11):
            raw_data.append(row)        
        else:
            # set the unwanted data to 0 (GaInP segments)
            row[-2] = 0
            raw_data.append(row)

    raw_data = np.asarray(raw_data)
    
elif domain == None:
    raw_data = np.asarray(file1)
else:
    sys.exit('u chose the wrong domain number')
    
    
# check the units
if abs(raw_data[0][0]) > 1E-6 or abs(raw_data[0][0]) < 1E-12:
    sys.exit('you are not using m units!?')

#recencter aronud z0. dont need this i think
#raw_data[:,0] = (raw_data[:,0] - np.mean(raw_data[:,0]))
#raw_data[:,1] = (raw_data[:,1] - np.mean(raw_data[:,1]))
raw_data[:,2] = (raw_data[:,2] - np.mean(raw_data[:,2]))

print( 'Maximum displacement: ')
print( raw_data[:,3].max())
print( 'Minimum displacement: ')
print( raw_data[:,3].min())

del file1
#%%
#-----------------------------------------------------
# Make 3d scatter plot of the COMSOL raw data
# (save to file and plot from separate script
# scatter_3d_plot.py)
#------------------------------------------------------

#np.save('raw_comsol_data',raw_data)




#%%  
#-----------------------------------------------------
# Rotate comsol coordinate system before interpolation 
# (In comsol coordinate syste, z is along the wire. x and y is the cross section)
# Following Berenguer notation. (this is the order in which to rotate)
# pos chi rotates counter clockwise around X
# pos phi rotates counter clockwise around Z
# pos psi rotates counter clockwise around Y  - it is the same as the theta motion
coord_rot = rotate_data(raw_data, chi=180 , phi=0 , psi=180)


#%%
#----------------------------------------------------------------------------
# interpolate the complex object calc from COMSOL simulation onto the grid 
# defined by the measurement
#--------------------------------------------------------------------------

# to avoid having NaNs when interpolation the COMSOL data to a regular 
# meaurement grid, we create an index for all non-NaN values
ind1 = np.where(np.invert(np.isnan(raw_data[:,3])))


# for clarity define the coordinate axes seperately. 
# make the units of xi xy xz  in m, from model it is arbitrary units 
xi = coord_rot[:,0]
yi = coord_rot[:,1]
zi = coord_rot[:,2]
del coord_rot

# make a data mask with amplitude 1 for the segments you want to simulate
mask = np.copy(raw_data[ind1,3])
mask[mask!=0] = 1

# the indexing is for only interpolating the values on the wire and not the sourroundings (if it is included, it is included as NaNs) 
# TODO try to convert the xx griddcorrdinates to a list, so that the data format is the same -input and utput both scatter points
interpol_data = inter.griddata((xi[ind1],yi[ind1],zi[ind1]),np.squeeze(raw_data[ind1,3]),(xx,yy,zz))   
mask_array = inter.griddata((xi[ind1],yi[ind1],zi[ind1]), np.squeeze(mask,),(xx,yy,zz))   
#del (raw_data, mask)

#TODO making the mask biany again. this does not work very well.
mask_array[mask_array<0.5] = 0
mask_array[mask_array>0.5] = 1
mask_array[np.isnan(mask_array)] = 0

#%%
#-----------------------------------------------------
# Make 3d scatter plot of the interpolated data (u)
#------------------------------------------------------
def scatter_interpol():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc=ax.scatter(xx[::100],yy[::100],zz[::100], c=interpol_data[::100], marker ='o', cmap='jet')#,alpha=0.2)
    plt.title('interpolated displacement')
    plt.colorbar(sc); plt.axis('scaled')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
#scatter_interpol()
#%%
#Calculate the strain
#-----------------------------------------------------

##InP_Qvect = 18543793660.27452

displacement_slice = np.fliplr(interpol_data[int(g.shape[0]/2)].T)

displacement_slice_NWlength = np.fliplr(interpol_data[int(g.shape[0]/2)].T)[75:100,143:153]

# ska man använda den Q-vektor som sätts at theta, alltså det theta om mäts upp ~~8 deg istället för detta som motsvarar 10.54

# np graient should be more correct
strain_dwdz2 = np.diff((displacement_slice ) , axis = 1, append=0) /dz
strain_dwdz = np.gradient(displacement_slice , dz) 

strain_dwdz_NWlength = np.gradient(displacement_slice_NWlength,dz)

# test because i am condused with orientations
displacement_temp = (interpol_data[int(g.shape[0]/2)])
strain_temp = np.gradient(displacement_temp , dz)
strain_temp_rot = np.gradient(np.rot90(displacement_temp,1) , dz)

#plt.figure()
#plt.imshow(displacement_temp, cmap='jet')
#
#plt.figure()
#plt.imshow(100*strain_temp[0], cmap='RdBu_r',vmin=-0.6,vmax=0.6)
#plt.colorbar()
#
#plt.figure()
#plt.imshow(np.rot90(100*strain_temp[0],1), cmap='RdBu_r',vmin=-0.6,vmax=0.6)
#plt.colorbar()
#
##rot before calculation strain?
#plt.figure()
#plt.imshow(displacement_temp, cmap='jet')
#
#plt.figure()
#plt.imshow(np.rot90(displacement_temp,1), cmap='jet')
#
#plt.figure()
#plt.imshow(100*strain_temp_rot[1], cmap='RdBu_r',vmin=-0.6,vmax=0.6)
#plt.colorbar()
#


#test2

#-----------------------------------------------------
# plot the strain
#-----------------------------------------------------
#%%

def plot_strain():
    plt.figure()
    plt.title('Interpolated data')
    plt.imshow((interpol_data)[int(g.shape[0]/2)],cmap='jet', origin='lower')
    plt.colorbar()
    
    
    dz2 = zz[0,-1,0] - zz[0,-2,0] # dont know why this is different from dz (Becauze dz is scanning positions, and we dont scan in z but we have resolutrion oin z.)
    dy2 = yy[0,0,1] - yy[0,0,0]
    plt.figure()
    plt.title('2d slice of displacement')
    shape6=displacement_slice_NWlength.shape
    plt.imshow(displacement_slice_NWlength,cmap='jet', origin='lower', interpolation='none',extent=[0,dz2*1E6*shape6[1],0,dy2*1E6*shape6[0]])    
    plt.colorbar()
    
    plt.figure()
    plt.title('2d slice of displacement')
    shape5 = displacement_slice.shape
    plt.imshow((displacement_slice),cmap='jet', origin='lower',interpolation='none',extent=[0,dz2*1E6*shape5[1],0,dy2*1E6*shape5[0]])
    plt.colorbar(orientation='horizontal')
    
    plt.figure()    
    #plt.imshow(100*strain_dwdz,cmap='RdBu_r', origin='lower',interpolation='none',extent=[0,dz2*1E6*shape5[1],0,dy2*1E6*shape5[0]])
    plt.imshow(100*strain_dwdz[1],cmap='RdBu_r', origin='lower',interpolation='none',extent=[0,dz2*1E6*shape5[1],0,dy2*1E6*shape5[0]],vmin=-0.6,vmax=0.6)
    plt.title('Strain calc with np.gradient [%]')
    #plt.title('Strain calc with np.diff [%]')
    plt.xlabel('z [um]')
    plt.ylabel('y [um]')
    plt.tight_layout()
    plt.colorbar(orientation='horizontal')
    
    plt.figure()    
    plt.imshow(100*strain_dwdz_NWlength[1],cmap='RdBu_r', origin='lower', interpolation = 'none', extent = [0,dz2*1E6*shape6[1],0,dy2*1E6*shape6[0]])
    plt.title('Strain [%]')
    plt.xlabel('z [um]')
    plt.ylabel('y [um]')
    plt.tight_layout()
    plt.colorbar(orientation='vertical')
    
    
    plt.figure()
    # left, right, bottom, top = extent
    plt.title('Will look like an oval because pixel size is not the same')
    plt.imshow((interpol_data)[:,int(g.shape[1]/2)],cmap='jet', origin='lower',extent=[0,interpol_data.shape[2]*g.resolution[2],0,interpol_data.shape[0]*g.resolution[0] ])
    print('Max/min displacemnet from model is: ')
    print( np.nanmax(np.gradient(interpol_data)[1]))
    print( np.nanmin(np.gradient(interpol_data)[1]))
    
#plot_strain()


#%%
#-----------------------------------------------------
# calculate scattering vector and calculate complex object 
# from displacement phase 
#-----------------------------------------------------

## save indices for NaNs.
#ind = np.where(np.isnan(interpol_data))
## put all nan values to 0 before saving in obj storage
#interpol_data[ind] = 0

Q_vect = calc_H111(domain_str)
#or ?
#Q_vect = 4 * np.pi / g.lam * g.sintheta

# return a complex object from the displacement field. object defined by the COMSOL model
#Note that the displacement data here shopuld be in nm
"teeeeeeeeeeeeeeeeeeeeest Avoid phase wrapping just reduce the displacement ffffffffffffffffffff"               #OBSOBS - sign!
reduce_displacemnt_factor = 1.0  #-1.0 or 1.0
"teeeeeeeeeeeeeeeeeeeeefffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

# calculate phase mean InGaP q-vector in the experiment
#    q_abs4 = 1.84942e+10
#    q_abs3 = 1.84811e+10 
#    q_abs2 = 1.84708e+10 
#    q_abs1 = 1.84287e+10     
# calculate phase from theory
phase = 1j*Q_vect*interpol_data * reduce_displacemnt_factor
    
    
obj = np.copy(phase)
#for all not nan, set amplitude to 1, and phase to phase values
obj[np.invert(np.isnan(phase))] = 1 * np.exp( phase[np.invert(np.isnan(phase))] )

# then set nans to 0 
obj[np.isnan(obj)] = 0
# then mask the data
obj = obj* mask_array

print( 'Max Min phase of object is: ')
print( np.max(phase[~np.isnan(phase)]))
print( np.min(phase[~np.isnan(phase)]))

#%%
#-----------------------------------------------------
# Make 3d scatter plot of the calc phase
#-----------------------------------------------------
def plot_phase_scatter():
    dat22 = np.angle(obj)
    dat22[dat22==0]=np.inf
    
    dat23 = np.abs(obj)
    dat23[dat23==0]=np.inf
    step = 100
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc=ax.scatter(xx[::100],yy[::100],zz[::100], c=dat22[::100], marker ='o', cmap='jet')#,alpha=0.2)
    plt.title('phase calculated from interpolated data' )
    plt.colorbar(sc); plt.axis('scaled')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc=ax.scatter(xx[::100],yy[::100],zz[::100], c=dat23[::100], marker ='o', cmap='jet')#,alpha=0.2)
    plt.title('abs(object) from interpolated data' )
    plt.colorbar(sc); plt.axis('scaled')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')

#plot_phase_scatter()

#%%
#---------------------------------------------------------
# Fill ptypy object storage with the 
# interpolated data from the model but shifted
#---------------------------------------------------------

# first put the data into the cartesian storage

# fill storage with zeros
obj_storage_cart.fill(0.0)
# fill storage with the comsol data covered by the experiment
obj_storage_cart.fill((obj)) 

# plot whats in the cartesian storage
fact = 1E6
# Plot that
#plt.figure()
#plt.title('Phase stored in the cartesian storage (np.angle)')
#plt.imshow((np.angle(obj_storage_cart.data[0][25])),cmap='jet',interpolation='none')#, extent = []);
#plt.colorbar()


fact = 1E9

#fig, ax = plt.subplots(nrows=1, ncols=3)
#plt.suptitle('Phase of the cartesian object storage. Mean values.')
#ax[0].imshow(np.mean(np.angle(obj_storage_cart.data[0]), axis=1).T, extent=[fact*xx.min(), fact*xx.max(), fact*yy.min(), fact*yy.max()], interpolation='none', origin='lower', cmap='jet')
#plt.setp(ax[0], ylabel='y um', xlabel='x um', title='top view')
#ax[1].imshow(np.mean(np.angle(obj_storage_cart.data[0]), axis=2).T, extent=[fact*xx.min(), fact*xx.max(), fact*zz.min(), fact*zz.max()], interpolation='none', origin='lower', cmap='jet')
#plt.setp(ax[1], ylabel='z um', xlabel='x um', title='side view')
#ax[2].imshow(np.mean(np.angle(obj_storage_cart.data[0]), axis=0).T, extent=[fact*zz.min(), fact*zz.max(), fact*yy.min(), fact*yy.max()], interpolation='none', origin='lower', cmap='jet')
#plt.setp(ax[2], ylabel='y um', xlabel='z um', title='front view')
#
#fig, ax = plt.subplots(nrows=1, ncols=3)
#plt.suptitle('Abs of the cartesian object storage. Mean values. ')
#ax[0].imshow(np.mean(np.abs(obj_storage_cart.data[0]), axis=1).T, extent=[fact*xx.min(), fact*xx.max(), fact*yy.min(), fact*yy.max()], interpolation='none', origin='lower', cmap='jet')
#plt.setp(ax[0], ylabel='y um', xlabel='x um', title='top view')
#ax[1].imshow(np.mean(np.abs(obj_storage_cart.data[0]), axis=2).T, extent=[fact*xx.min(), fact*xx.max(), fact*zz.min(), fact*zz.max()], interpolation='none', origin='lower', cmap='jet')
#plt.setp(ax[1], ylabel='z um', xlabel='x um', title='side view')
#ax[2].imshow(np.mean(np.abs(obj_storage_cart.data[0]), axis=0).T, extent=[fact*zz.min(), fact*zz.max(), fact*yy.min(), fact*yy.max()], interpolation='none', origin='lower', cmap='jet')
#plt.setp(ax[2], ylabel='y um', xlabel='z um', title='front view')
#
#
#


# make a copy of the cartesian storage but shifted to natural
obj_storage_natural = g.coordinate_shift(obj_storage_cart, input_system='cartesian', input_space='real', keep_dims=True)

# put the shifted storage data into the original object storage (or are these now the same. or is it a differentce in 
# how the views are connected?)
#this is messy?
obj_storage.data = obj_storage_natural.data



#del (obj,  mask_array, phase)

#%%
#-----------------------------------------------------
# Make 3d scatter plot of the cartesian interpolated phase
# and 2d cut along long axis 
#-----------------------------------------------------

#dat44 = np.angle(obj_storage_cart.data[0])
#dat44[dat44==0]=np.inf
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#sc=ax.scatter(xx,yy,zz, c=dat44, marker ='o', cmap='jet')#,alpha=0.2)
#plt.title('phase from data in (cartesian) obj storage')
#plt.colorbar(sc); plt.axis('scaled')
#ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
#


# plot slice to make sure its doing the right thing when it goes from scatterpoints in the interpolation
#plt.figure()
#plt.imshow(np.angle((obj_storage_cart.data[0][:,:,g.shape[2]/2].T)), extent=[fact*xx.min(), fact*xx.max(), fact*zz.min(), fact*zz.max()], interpolation='none', origin='lower', cmap='jet')
#plt.xlabel('x'); plt.ylabel('z')

#%% 
#-------------------------------------------------------
# test propagate without probe and plot that
# REDO THIS. BACK PROPAGATE TO SEE WHAT THE REAL SPOACE IMAGE LOOKS LIKE ??
# CAN NOT PLOT THE SKEWED SYSTEM
#---------------------------------------------------

prop_data = abs(g.propagator.fw(views[50].data))**2   #v.data * probeView.data
#try BasicBragg3dPropagator() ?
inx_slice = int(g.shape[0]/2)

plt.figure()
plt.suptitle('Test of propagator, without using any probe')
plt.subplot(211)
# TODO not correct to take the max nd min here, right. it will give the max of the total thing
plt.imshow(np.abs(views[0].data[inx_slice]), cmap = 'jet')#, extent = [factor*yy.min(),factor*yy.max(), factor*zz.min(), factor*zz.max()])
plt.title('view 0')
plt.xlabel('y [nm]'); plt.ylabel('z [nm]')
plt.subplot(212)
plt.imshow(prop_data[inx_slice], cmap = 'jet')
plt.title('2D cut of the resulting diffraction pattern') 


#%%
##--------
## plot isosurface of amplitude values and scattering vectors to visualize diffraction geometry. 
##----------
# calculate max and min values of coordinate axes (for plotting)
#xmax = np.nanmax(xi); xmin = np.nanmin(xi)
#ymax = np.nanmax(yi); zmin = np.nanmin(yi)
#zmax = np.nanmax(zi); ymin = np.nanmin(zi)
#
#ind = np.where(np.isnan(comp))
#comp[ind] = 0

#sc = mlab.figure()
#src = mlab.pipeline.scalar_scatter(xi[ind1],yi[ind1],zi[ind1], comp.real[ind1], extent = [xmin, xmax, ymin, ymax, zmin, zmax])
#g_1 = mlab.pipeline.glyph(src, mode='point')
#gs = mlab.pipeline.gaussian_splatter(src)
#gs.filter.radius = 0.15   #sets isosurface value, may need to tune. 
#iso=mlab.pipeline.iso_surface(gs, transparent=True, opacity = 0.1)
#
##np.round
#
#qscale = (np.max([xmax,ymax,zmax]))*1.5   # 1.5* the maximum value of the input geometry for the purpose of plotting the scattering vectors at visual scale. 
#qscale =1 
#black = (0,0,0); red = (1,0,0); blue = (0,0,1); white = (1,1,1)
#mlab.points3d(0, 0, 0, color=white, scale_factor=10)  #plot origin
##plot ki, kf, and qbragg scaled by size of object, qscale (for visualization purposes only)
#mlab.plot3d([0, -ki[0]*qscale], [0, -ki[1]*qscale], [0, -ki[2]*qscale], color=red, tube_radius=2.)
#mlab.plot3d([0, kf[0]*qscale], [0, kf[1]*qscale], [0, kf[2]*qscale], color=black, tube_radius=2.)
#mlab.plot3d([0, qbragg[0]*qscale], [0, qbragg[1]*qscale], [0, qbragg[2]*qscale], color=blue, tube_radius=2.)
#mlab.text3d(-ki[0]*qscale+qscale*0.05, -ki[1]*qscale+qscale*0.05, -ki[2]*qscale+qscale*0.05, 'ki', color=red, scale=25.)
#mlab.text3d(kf[0]*qscale+qscale*0.05, kf[1]*qscale+qscale*0.05, kf[2]*qscale+qscale*0.05, 'kf', color=black, scale=25.)
#mlab.text3d(qbragg[0]*qscale+qscale*0.05, qbragg[1]*qscale+qscale*0.05, qbragg[2]*qscale+qscale*0.05, 'qbragg', color=blue, scale=25.)


#quit()

#%%
#---------------------------------------------------------
# Set up the probe and calculate diffraction patterns
# 1. Using a square or gaussian probe
# 2. Using a real probe
# TODO for real probe: this is probably an inconvinent way to do it, maybe you can load the whole Storage in one go
# TODO fill the probes into the storages for phase retrieval
#---------------------------------------------------------

choise = 'real2020'#sample_plane'            # 'square' 'loaded' 'circ' or 'real' 'gauss'

if choise == 'circ':
    fsize = g.shape * g.resolution
    Cprobe = ptypy.core.Container(data_dims=2, data_type='complex128')
    Sprobe = Cprobe.new_storage(psize=g.resolution, shape=g.shape[1])
    #zi, yi = Sprobe.grids()
    #  apert = u.smooth_step(fsize[1]/5-np.sqrt(zi[0]**2+yi[0]**2), 0.2e-6)
    #y, x = pr3.grids()
    #apert = u.smooth_step(fsize[1]/5-np.abs(yi), 0.00000000000001)*u.smooth_step(fsize[2]/5-np.abs(zi), 0.0000000000000000000001)
    #from ptypy.resources import moon_pr
    from ptypy.core.illumination import aperture
    A=np.ones((128,128))
    apert=aperture(A, grids=None)
    
    #moon_probe = -moon_pr(g.shape[1])
    #apert = moon_probe 
    #u.smooth_step(90e-9-np.sqrt(zi[0]**2+yi[0]**2),0.0000000001)
    #if (x-a)**2 + (y-b)**2 <= r**2:
    Sprobe.fill(apert)

elif choise == 'square':    
    # First set up a two-dimensional representation of the probe, with
    # arbitrary pixel spacing. 
    # make a 50 x 50 nm probe (current size of 1 view)
    Cprobe = ptypy.core.Container(data_dims=2, data_type='complex128')
    Sprobe = Cprobe.new_storage(psize=g.resolution[1], shape=256)
    zi, yi = Sprobe.grids()
    square = u.smooth_step(fsize[1]/5-np.abs(yi), 0.00000000000001)*u.smooth_step(fsize[2]/5-np.abs(zi), 0.0000000000000000000001)
    #square = (yi > -100.0e-9) & (yi < 100.0e-9) & (zi > -100.0e-9) & (zi < 100.0e-9) = 1
    # square probe
    # need to use square otherwise it changes data type to whatever you put in
    #Sprobe.fill(square)
    
    
if choise == 'gauss':
    Cprobe = ptypy.core.Container(data_dims=2, data_type='complex128')
    Sprobe = Cprobe.new_storage(psize=g.resolution, shape=100)
    zi, yi = Sprobe.grids()
    std_dev = 90E-9
    # gaussian probe
    Sprobe.fill( np.roll(np.exp(-zi**2 / (2 * (std_dev)**2) - yi**2 / (2 * (std_dev)**2)), 100, axis=1))

elif choise == 'real':   
    
    loaded_profile = np.load(r'C:\Users\Sanna\Documents\Simulations\probe10_focus.npy')
    # center the probe (cut out the center part)
    ###################
    "               OOOOOOOOOOOOOOOBS ROTATE. rot90,3 is correct"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    loaded_profile_cut = np.rot90(np.copy(loaded_profile)[1:121,0:120],3)
    # save a psize, shape and the array data in the contaioner
    Cprobe = ptypy.core.Container(data_dims=2, data_type='complex128')
    #TODO why im i removing one pixel
    Sprobe = Cprobe.new_storage(psize=[ 1.89051824e-08,   1.85578409e-08], shape=loaded_profile_cut.shape[0]) 
### resolution from:    g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(51, 128, 128), energy=9.49, distance=1.0, theta_bragg=11)
    
    # fill storage
    Sprobe.fill(0.0)
    Sprobe.fill(1j*loaded_profile_cut)
    zi, yi = Sprobe.grids()
    
elif choise == 'real2020':   

    loaded_profile = np.load(r'C:\Users\Sanna\Documents\Beamtime\NanoMAX_May2020\Analysis\siemensstar\scan14\np_save\focus\probe14_focus.npy')    
    "               OOOOOOOOOOOOOOOBS ROTATE. rot90,3 is correct"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    loaded_profile = np.rot90(loaded_profile,3)
    # save a psize, shape and the array data in the contaioner
    Cprobe = ptypy.core.Container(data_dims=2, data_type='complex128')
    #TKNowthat: this is pixel size in the transmission system. Should convert it to Bragg system? No,
    # it is resampled to bragg geometry when you prepare it in 3d! 

    Sprobe = Cprobe.new_storage(psize=[5.96673929e-08, 5.96673929e-08], shape=(1,128,128))

    # In ptypy reconstruction: Probe recentered from (82.5020948929997, 81.8261553912669) to (85, 85)
    # fill storage
    Sprobe.fill(loaded_profile)
    zi, yi = Sprobe.grids()  
    
elif choise == 'sample_plane':    # this loads the probe in the sample plane not in focus. try this too
    from ptypy import io
#    
    path_probe = 'C:/Users/Sanna/Documents/beamtime/NanoMAX062017/Analysis_ptypy/nice_probe_ptyrfiles/scan10/scan10_pilatus_ML.ptyr'
    # load all variables in the ptyr file
    loaded_probe = io.h5read(path_probe,'content').values()[0].probe['S00G00']
    plt.figure()
    plt.imshow((abs((loaded_probe['data'][0]))))
    # save the psize, the shape and the array data in the contaioner
    #TODO is it correct with (1, 128,128)  etc?
    Cprobe = ptypy.core.Container(data_dims=2, data_type='complex128')
    #TODO is _psize correct?, that should be transmisssion resolution
    Sprobe = Cprobe.new_storage(psize=loaded_probe['_psize'], shape=loaded_probe['shape'])
    # fill storage
    Sprobe.fill(0.0)
    Cprobe.fill(loaded_probe['data'])
    zi, yi = Sprobe.grids()

elif choise == 'loaded':
    import matplotlib.image as mpimg
    # load 2d profile
    loaded_profile = mpimg.imread('C:/Users/Sanna/Documents/python_utilities/fft2_images/L.png')#oval.png single.png')#circle_insquare.png')
    
    # squeeze rgb image
    loaded_profile=np.array(np.sum(loaded_profile, axis=2))
    # save a psize, shape and the array data in the contaioner
    Cprobe = ptypy.core.Container(data_dims=2, data_type='complex128')
    #TODO dont know if this is right!?
    Sprobe = Cprobe.new_storage(psize=g.resolution[1:3], shape=128)
    # fill storage
    Sprobe.fill(0.0)
    Cprobe.fill(1j*loaded_profile)
    zi, yi = Sprobe.grids()
    # reshape?
   
fig = u.plot_storage(Sprobe, 11, channel='c') 
plt.figure()
plt.imshow(abs(np.squeeze(Sprobe.data)),cmap='jet')

# In order to put some physics in the illumination we set the number of
# photons to 1 billion
#comment out to get normal fft
nbr_photons = 1E9
Sprobe.data *= np.sqrt(nbr_photons/np.sum(Sprobe.data*Sprobe.data.conj()))
print( u.norm2(Sprobe.data)    )

#import nmutils.utils
# propager i nerfield
#field_propagated = nmutils.utils.propagateNearfield(Sprobe.data[0], g.psize, -100E-9, g.energy)

# prepare in 3d
# This will also resample the probe to the Bragg geometry g with the reolution of g!
Sloaded_probe_3d = g.prepare_3d_probe(Sprobe, system='natural', layer=0)#NOTE usually its the input system you specify but here its the output. Also there is an autocenter 
loaded_probeView = Sloaded_probe_3d.views[0]

plt.figure()
plt.imshow(abs(np.squeeze(Sloaded_probe_3d.data)[31]),cmap='jet')
# visualize 3d probe and probe propagated to transmission

# propagate probe to transmission
ill = Sprobe.data[0]
propagated_ill = g.propagator.fw(ill)
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.log10(np.abs(propagated_ill)+1))
plt.colorbar(im)

factor = 1E9
#plt.figure()
#plt.subplot(121)
#plt.suptitle('Loaded 2d probe. psize=%f nm \n axes here are correct. defined in sample plane'%(Sprobe.psize[0]*1E9))
#plt.imshow((abs(np.squeeze(Sprobe.data))), cmap='jet', interpolation='none', extent=[-factor*Sprobe.shape[1]/2*Sprobe.psize[0], factor*Sprobe.shape[1]/2*Sprobe.psize[0], -factor*Sprobe.shape[2]/2*Sprobe.psize[1],factor*Sprobe.shape[2]/2*Sprobe.psize[1]]) 
#plt.title('Amplitude')
#plt.xlabel('y [nm]'); plt.ylabel('z [nm]');plt.colorbar()
#plt.subplot(122)
#plt.imshow(np.angle(np.squeeze(Sprobe.data)), cmap='jet', interpolation='none', extent=[-factor*Sprobe.shape[1]/2*Sprobe.psize[0], factor*Sprobe.shape[1]/2*Sprobe.psize[0], -factor*Sprobe.shape[2]/2*Sprobe.psize[1],factor*Sprobe.shape[2]/2*Sprobe.psize[1]])
#plt.title('Phase')
#plt.xlabel('y [nm]'); plt.colorbar()
#


#%%
#------------------------------------------------------
#  Visualize the probe extruded in 3d. Corrected
#------------------------------------------------------

r3, r1, r2 = Sloaded_probe_3d.grids()
r3_slice = int(g.shape[0]/2)
r1_slice = int(g.shape[1]/2)
r2_slice = int(g.shape[2]/2)

fac1 = 1E6
#plt.figure() #extent : scalars (left, right, bottom, top)
##extent is checked
#plt.suptitle('Central cut-plot from the 3d probe \n extruded from 2d probe in quasi vertical zi and y coord. psize=%f nm'%(Sloaded_probe_3d.psize[1]*1E9))
#plt.subplot(121)
##    ax[-1].imshow(np.mean(np.abs( views[i].data + (loaded_probeView.data/loaded_probeView.data.max()) ), axis=1), vmin=0, extent=[mufactor*r2.min(), mufactor*r2.max(), mufactor*r3.min(), mufactor*r3.max()])
##    plt.setp(ax[-1], xlabel='r2 [um]', ylabel='r3', xlim=[mufactor*r2.min(), mufactor*r2.max()], ylim=[mufactor*r3.min(), mufactor*r3.max()], yticks=[])
##    # diffraction
#plt.imshow((abs(loaded_probeView.data[r3_slice])), extent=[-fac1*r2_slice*Sloaded_probe_3d.psize[2], fac1*r2_slice*Sloaded_probe_3d.psize[2], -fac1*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1],fac1*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1]], cmap='jet', interpolation='none')
##plt.imshow(abs() extent=[-1E9*r2_slice*g.psize[2], 1E9*Sloaded_probe_3d.shape[3]/2*Sloaded_probe_3d.psize[2], -1E9*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1],1E9*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1]] ,cmap='jet', interpolation='none')
#plt.xlabel('r2 [nm]'); plt.ylabel('r1 [nm]')#;plt.colorbar()
#plt.title('Amplitude')
#plt.subplot(122)
##plt.imshow(np.angle(loaded_probeView.data[r3_slice]), extent=[-fac1*r2_slice*Sloaded_probe_3d.psize[2], fac1*r2_slice*Sloaded_probe_3d.psize[2], -fac1*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1],fac1*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1]], cmap='jet', interpolation='none')
#plt.imshow(np.angle(loaded_probeView.data[r3_slice]))#, extent=[-fac1*r2_slice, fac1*r2_slice, -fac1*r1_slice,fac1*r1_slice], cmap='jet', interpolation='none')
#plt.xlabel('r2 [nm]'); plt.ylabel('r1 [nm]')#;plt.colorbar()
#plt.title('Phase')
#plt.tight_layout()

#%%
#------------------------------------------------------
#  Visualize the probe extruded in 3d more . wrong axes. 
#------------------------------------------------------

# plot probe in 2d cuts (cannot do 3d cuts in matplotlib)
def plot3ddata(data):
    plt.figure()
    plt.suptitle('3d probe central cut plots. Skewed system. OBS origin lower')
    plt.subplot(121)
    #plt.title('-axis')
    plt.imshow((abs((data[int(data.shape[0]/2),:,:]))),origin='lower', cmap='jet', interpolation='none') 
    plt.xlabel('r2'); plt.ylabel('r1')
    plt.subplot(122)
#    plt.imshow(np.transpose(abs(data[:,data.shape[1]/2,:])), cmap='jet', interpolation='none') 
#    plt.xlabel('x? k_i?'); plt.ylabel('y')
    #plt.subplot(223)
    #plt.title('-axis')
    plt.imshow(np.transpose(abs(loaded_probeView.data[:,:,int(data.shape[2]/2)])),origin='lower', cmap='jet', interpolation='none')#, extent=[-1E9*Sloaded_probe_3d.shape[3]/2*Sloaded_probe_3d.psize[2], 1E9*Sloaded_probe_3d.shape[3]/2*Sloaded_probe_3d.psize[2], -1E9*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1],1E9*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1]])
    plt.xlabel('r3'); plt.ylabel('r1')
    
plot3ddata(np.squeeze(Sloaded_probe_3d.data))
#Sloaded_probe_3d.data ( x, z, y)
""" notation guide
extent : scalars (left, right, bottom, top)
Sprobe_3d.psize[2] ~y
Sprobe_3d.shape[3] ~y
Sprobe_3d.psize[0] ~x
Sprobe_3d.shape[1] ~x
Sprobe_3d.psize[1] ~z
Sprobe_3d.shape[2] ~z
"""
#%%
#------------------------------------------------------
# Calculate diffraction pattterns. Plot
#------------------------------------------------------

# create a container for the diffraction patterns
diff_Cont = ptypy.core.classes.Container(ID='Cdiff', data_type='real', data_dims=3)
pr_shape = (Npos,)+ tuple(g.shape)
# define diff3 to ease the xrd coding (so I can just copy paste)
#diff3 = diff_Cont.new_storage(psize=np.array([ g.dq3, g.dq1, g.dq2]), shape=pr_shape)# add center to ba at qabs ? 

# Calculate diffraction patterns by using the geometry's propagator. all in 3d
# todo is this OK lam factor?
#lam_factor =  nbr_photons*25   # higher number (less total intensity/signal)

lam_factor =  nbr_photons*25
# det här verkar va liiite högt

#lam_factor =  2000000

diff2 = []
print( 'calculating diffraction data')
for v in views:
    print(v)
    exit_wave = v.data * loaded_probeView.data 
    prop_exit_wave = g.propagator.fw(exit_wave)
    # without noise
    #diff2.append(np.array((np.real( prop_exit_wave*prop_exit_wave.conj() )),float)) #this is actallu real but data type does not change
    #diff2.append(np.abs( prop_exit_wave)**2) #I used this 2/11/2020 but i guess it doesng matter?
    
#    # with noise
    diff2.append(np.array(np.random.poisson(np.real( prop_exit_wave*prop_exit_wave.conj() )/lam_factor),float)) #this is actallu real but data type does not change
        


#diff3.fill(np.array(diff2))
#plt.figure()
#plt.imshow(diff2[1][31],cmap='jet');plt.colorbar()

#del diff2
del (exit_wave,prop_exit_wave)


#plt.figure()
#plt.imshow(diff3.data[200,25],cmap='jet')

# make 0 values white instead of colored
#for diff in diff2:
#    diff[diff==0] = np.inf

#matplotlib.get_backend()
#matplotlib.use( 'agg' )

#del diff, zz, xx, yy

#---------------------------------------------------------------------
#%%
# plot single postion diffraction in 3 projections
#-----------------------------------------------------------------

#position = int(len(diff2)/2)# 357# 524

#plt.figure()# extent guide: extent : scalars (left, right, bottom, top)
#plt.suptitle('Final diffraction pattern in skewed system', fontsize=13)
#plt.subplot(221)
#plt.imshow(np.sum(diff2[position],axis=0),cmap='jet')
#plt.ylabel('$q_1$') ; plt.xlabel('$q_2$') #[$\mathrm{\AA^{-1}}$]
#
##1 and 2 
## xzy
## 3 1 2 
#
#plt.subplot(222)
#plt.imshow(np.sum(diff2[position],axis=1),cmap='jet')
#plt.ylabel('$q_3$') ; plt.xlabel('$q_2$') #[$\mathrm{\AA^{-1}}$]
#
#plt.subplot(223)
#plt.imshow(np.sum(diff2[position],axis=2),cmap='jet')#, extent = [ -(g.dq1*g.shape[1]/2 )*rec_fact, g.dq1*g.shape[1]/2*rec_fact, -g.dq2*g.shape[2]/2*rec_fact, g.dq2*g.shape[2]/2*rec_fact], interpolation='none',cmap='jet')
#plt.ylabel('$q_3$') ; plt.xlabel('$q_1$')  #[$\mathrm{\AA^{-1}}$]

#plt.savefig('aa')
#%%
# plot object and diffraction pattern at the same time. Wrong axes here!?!?
#----------------------------------


factor = 1E6
rec_fact = 1E-10

for plot_view in range(0,len(diff2),50):
    plt.figure()# extent guide: extent : scalars (left, right, bottom, top)
    plt.suptitle('Final diffraction pattern using object and probe', fontsize=15)
    xcut = int(views[0].shape[0]/2) # which y-z-cut to plot
    plt.subplot(221)
    plt.imshow(np.abs(views[plot_view].data[xcut]), cmap = 'jet', extent = [factor*yy_vgrid.min(),factor*yy_vgrid.max(), factor*zz_vgrid.min(), factor*zz_vgrid.max()])
    plt.title('One views object abs data')
    plt.xlabel(r'$r_1$ [$\mathrm{\mu m}$]'); plt.ylabel(r'$r_2$ [$\mathrm{\mu m}$]')
    
    plt.subplot(222)
    plt.imshow((abs(loaded_probeView.data[xcut ,:,:])),extent = [factor*yy_vgrid.min(),factor*yy_vgrid.max(), factor*zz_vgrid.min(), factor*zz_vgrid.max()])# extent=[-factor*Sloaded_probe_3d.shape[3]/2*Sloaded_probe_3d.psize[2], factor*Sloaded_probe_3d.shape[3]/2*Sloaded_probe_3d.psize[2], -factor*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1],factor*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1]])
    plt.title('One views probe abs data')
    plt.xlabel(r'$r_1$ [$\mathrm{\mu m}$]'); plt.ylabel(r'$r_2$ [$\mathrm{\mu m}$]')
    
    plt.subplot(223)
    plt.title('Same view\'s diffraction pattern')
    plt.imshow(diff2[plot_view][xcut].T , extent = [ -(g.dq1*g.shape[1]/2 )*rec_fact, g.dq1*g.shape[1]/2*rec_fact, -g.dq2*g.shape[2]/2*rec_fact, g.dq2*g.shape[2]/2*rec_fact], interpolation='none',cmap='jet')
    plt.xlabel(r'$q_1$ [$\mathrm{\AA^{-1}}$]'); plt.ylabel(r'$q_2$ [$\mathrm{\AA^{-1}}$]'); plt.colorbar()#$10^6$ 
    plt.tight_layout()
    
    
    #plt.savefig(r'C:\Users\Sanna\Documents\Simulations\save_simulation\diffraction\noise_free\diff%d'%plot_view)

#%%
# compare the cuts to 2d diffraction patterns
#prop_exitwave_2d = g.propagator.fw( np.array(views[plot_view].data[xcut]*loaded_probeView.data[xcut ,:,:]))
#diff_2d = np.array(np.random.poisson(np.real( prop_exitwave_2d*prop_exitwave_2d.conj() )/lam_factor), float) 
#diff_2d[diff_2d==0] = -np.nan

#plt.figure()
#plt.title('2d Diffraction ')
#plt.imshow(diff_2d.T, extent = [-g.dq1*g.shape[1]/2*rec_fact, g.dq1*g.shape[1]/2*rec_fact,-g.dq2*g.shape[2]/2*rec_fact, g.dq2*g.shape[2]/2*rec_fact ], interpolation='none',cmap='jet')
#plt.xlabel('$q_1$ [$\AA^{-1}$]'); plt.ylabel('$q_2$ [$\AA^{-1}$]')#; plt.colorbar()#$10^6$ 
#plt.tight_layout()
#%%

def make_finite(matrix):
    mask = np.isinf(matrix)
    matrix[mask] = 0
    return matrix

#%%
# ------------------------------------------------------    
# Visualize a single field of view with probe and object
# ------------------------------------------------------

# In order to visualize the field of view, we'll create a copy of the
# object storage and set its value equal to 1 where covered by the first
# view.

S_display = obj_storage.copy(owner=obj_container)
fact=1E6
def plot_probe_sample(frame):
    
    #, ID='S_display')
    S_display.fill(0.0)
    #S_display[obj_storage.views[frame]] = 1
    
    # Then, to see how the probe is contained by this field of view, we add
    # the probe and the object itself to the above view.
    S_display[obj_storage.views[frame]] +=1*(loaded_probeView.data)/loaded_probeView.data.max()
    S_display.data += obj_storage.data
    
    # To visualize how this looks in cartesian real space, make a shifted
    # (nearest-neighbor interpolated) copy of the object Storage.
    S_display_cart = g.coordinate_shift(S_display, input_system='natural', input_space='real', keep_dims=False)
    
    
    # Plot that    
    plt.suptitle('Position: %d'%frame)
    x, z, y = S_display_cart.grids()
    
    ax[0].imshow(np.mean(np.abs(S_display_cart.data[0]), axis=1).T, extent=[fact*x.min(), fact*x.max(), fact*y.min(), fact*y.max()], interpolation='none', origin='lower', cmap='jet')
    plt.setp(ax[0], ylabel='y um', xlabel='x um', title='top view') 
    ax[1].imshow(np.mean(np.abs(S_display_cart.data[0]), axis=2).T, extent=[fact*x.min(), fact*x.max(), fact*z.min(), fact*z.max()], interpolation='none', origin='lower', cmap='jet')
    plt.setp(ax[1], ylabel='z um', xlabel='x um', title='side view')
    ax[2].imshow(np.mean(np.abs(S_display_cart.data[0]), axis=0).T, extent=[fact*z.min(), fact*z.max(), fact*y.min(), fact*y.max()], interpolation='none', origin='lower', cmap='jet')
    plt.setp(ax[2], ylabel='y um', xlabel='z um', title='front view')

    
#    ax.imshow((np.abs(S_display_cart.data[0]))[30].T, extent=[fact*z.min(), fact*z.max(), fact*y.min(), fact*y.max()], interpolation='none', origin='lower', cmap='jet')
#    plt.setp(ax, ylabel='y um', xlabel='z um', title='Central cut of front view')

    #plt.savefig('C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/InGaP_InP_full_NW/real_probe/positions/probe_pos_%d'%frame)
    
frame = int(len(obj_storage.views)/2)

#    TRYYYYY
#fig, ax = plt.subplots(ncols=3)
#for frame in range(0,10):
#    print(frame)
##    
#    plot_probe_sample(frame)
#    plt.draw()
#    plt.pause(.1)


#frames = np.arange(0,200,1)
#for frame in frames:
#    plot_probe_sample(frame)

#%%
#Prepare for reconstructions
# Keep a copy of the object storage, and fill the actual one (obj_storage) with an
# initial guess (like zeros everywhere).     plot it

S_true = obj_storage.copy(owner=obj_container)

S_true_cart = g.coordinate_shift(S_true, input_space='real', input_system='natural', keep_dims=False)

x, z, y = S_true_cart.grids()
xcut = int(S_true_cart.data.shape[1]/2)
zcut = int(S_true_cart.data.shape[2]/2)
ycut = int(S_true_cart.data.shape[3]/2)

fact = 1E6

fig, ax = plt.subplots(ncols=3)
plt.suptitle('Phase in true object. Central cuts.')
im1 = ax[0].imshow((np.angle(np.squeeze(S_true_cart.data[0])[:,zcut])).T, extent=[fact*x.min(), fact*x.max(), fact*y.min(), fact*y.max()], interpolation='none', origin='lower', cmap='jet')
plt.setp(ax[0], ylabel=r'y [$\mu$m]', xlabel=r'x [$\mu$m]', title='top view')
fig.colorbar(im1,ax=ax[0] )

im2 = ax[1].imshow( (  np.angle(np.squeeze(S_true_cart.data[0])[:,:,ycut])).T, extent=[fact*x.min(), fact*x.max(), fact*z.min(), fact*z.max()], interpolation='none', origin='lower', cmap='jet')
plt.setp(ax[1], ylabel=r'z [$\mu$m]', xlabel=r'x [$\mu$m]', title='side view')
fig.colorbar(im2,ax=ax[1] )

im2 = ax[2].imshow(( np.angle(np.squeeze(S_true_cart.data[0])[xcut])).T, extent=[fact* z.min(), fact*z.max(), fact*y.min(),fact* y.max()], interpolation='none', origin='lower', cmap='jet')
plt.setp(ax[2], ylabel=r'y [$\mu$m]', xlabel=r'z [$\mu$m]', title='front view')
plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=70)
plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=70)
fig.colorbar(im2,ax=ax[2] )
plt.tight_layout()
plt.draw()

fig, ax = plt.subplots(ncols=3)
plt.suptitle('Amplitude in true object. Central cuts.')
im1 = ax[0].imshow((np.abs(np.squeeze(S_true_cart.data[0])[:,zcut])).T, extent=[fact*x.min(), fact*x.max(), fact*y.min(), fact*y.max()], interpolation='none', origin='lower', cmap='jet')
plt.setp(ax[0], ylabel=r'y [$\mu$m]', xlabel=r'x [$\mu$m]', title='top view')
ax[1].imshow( (  np.abs(np.squeeze(S_true_cart.data[0])[:,:,ycut])).T, extent=[fact*x.min(), fact*x.max(), fact*z.min(), fact*z.max()], interpolation='none', origin='lower', cmap='jet')
plt.setp(ax[1], ylabel=r'z [$\mu$m]', xlabel=r'x [$\mu$m]', title='side view')
#fig.colorbar(im1,ax=ax[1] )
im2 = ax[2].imshow(( np.abs(np.squeeze(S_true_cart.data[0])[xcut])).T, extent=[fact* z.min(), fact*z.max(), fact*y.min(),fact* y.max()], interpolation='none', origin='lower', cmap='jet')
plt.setp(ax[2], ylabel=r'y [$\mu$m]', xlabel=r'z [$\mu$m]', title='front view')
#fig.colorbar(im2,ax=ax[2] )
plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=70)
plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=70)
plt.tight_layout()
plt.draw()

# save the central projection
#plot2drecons(np.fliplr(S_true_cart.data[0][xcut+5].T), S_true_cart.data[0][xcut].T, [-2.8849337617200645,
# 2.8715154651539248,
# -1.4321168817112304,
# 1.418856540213904], r'C:\Users\Sanna', save=True)

#%%
#%%

# Reconstruct projections of the data 
# ------------------------------

algorithm = 'PIE'


# -0.2
#bragg     done
# *0.2 de  done
# 0.4 deg done   (51)
# 0.6 deg done

projection = int(loaded_probeView.shape[0]/2)


print('Start reconstruction')
# if you dont want to start over and just continue to iteratie, just comment this part out
# zero everything
obj_storage.fill(0.0)


storage_save = []



# Here's a PIE/cPIE implementation
if algorithm == 'PIE':
    beta = 1.0
    eps = 1e-3
    
    errors = []
    ferrors = []
    for i in range(100):
        print('iteration ', i)
        ferrors_ = []
        for j in range(len(views)):

            exit_ = views[j].data[projection] * loaded_probeView.data[projection]
            #TODO this g is a 3d bragg thing. its not a problem, it is only afft?
            prop = g.propagator.fw(exit_)
            ferrors_.append(np.abs(prop)**2 - diff2[j][:][projection])
            prop = np.sqrt(diff2[j][:][projection]) * np.exp(1j * np.angle(prop))
            exit = g.propagator.bw(prop)
            # ePIE scaling (Maiden2009)
            #SH: det här är väl  J.M. Rodenburg and H.M.L Faulkner 2004, jag skrev exakt såhär baserat på det peket.
            views[j].data[projection] += beta * np.conj(loaded_probeView.data[projection]) / (np.abs(loaded_probeView.data[projection]).max())**2 * (exit - exit_)
            # PIE and cPIE scaling (Rodenburg2004 and Godard2011b)
            #views[j].data += beta * np.abs(loaded_probeView.data) / np.abs(loaded_probeView.data).max() * np.conj(loaded_probeView.data) / (np.abs(loaded_probeView.data)**2 + eps) * (exit - exit_)
            

        errors.append(np.abs(np.squeeze(obj_storage.data)[projection] - np.squeeze(S_true.data)[projection]).sum())
        ferrors.append(np.mean(ferrors_))

        #TODO only save the projection
        save_iter_interval = 10
        if not (i % save_iter_interval ):
            
            
#            #save arrays instread of plottting to improve speed
             storage_save.append(obj_storage.copy(owner=obj_container))


if algorithm == 'DM':
    alpha = 1.0
    fig, ax = plt.subplots(ncols=4)
    errors = []
    ferrors = []
    # create initial exit waves
    exitwaves = []
    for j in range(len(views)):
        exitwaves.append(views[j].data * loaded_probeView.data)
    # we also need a constant normalization storage, which contains the
    # denominator of the DM object update equation.
    Snorm = obj_storage.copy(owner=obj_container)
    Snorm.fill(0.0)
    for j in range(len(views)):
        Snorm[views[j]] += np.abs(loaded_probeView.data)**2

    # iterate
    for i in range(100):
        print( i)
        #TODO this is not defined anywhere
        ferrors_ = []
        # fourier update, updates all the exit waves
        for j in range(len(views)):
            # in DM, you propagate the following linear combination
            im = g.propagator.fw((1 + alpha) * loaded_probeView.data * views[j].data - alpha * exitwaves[j])
            im = np.sqrt(diff2[j]) * np.exp(1j * np.angle(im))
            exitwaves[j][:] += g.propagator.bw(im) - views[j].data * loaded_probeView.data
        # object update, now skipping the iteration because the probe is constant
        obj_storage.fill(0.0)
        for j in range(len(views)):
            views[j].data += np.conj(loaded_probeView.data) * exitwaves[j]
        obj_storage.data[:] /= Snorm.data + 1e-10
        errors.append(np.abs(obj_storage.data - S_true.data).sum())
        ferrors.append(np.mean(ferrors_))

        if not (i % 5):
            ax[0].clear()
            ax[0].plot(errors/errors[0])
            #ax[0].plot(criterion/criterion[0])
            ax[1].clear()
            S_cart = g.coordinate_shift(obj_storage, input_space='real', input_system='natural', keep_dims=False)
            x, z, y = S_cart.grids()
            ax[1].imshow(np.mean(np.abs(S_cart.data[0]), axis=1).T, extent=[x.min(), x.max(), y.min(), y.max()], interpolation='none', origin='lower')
            plt.setp(ax[1], ylabel='y', xlabel='x', title='top view')
            ax[2].clear()
            ax[2].imshow(np.mean(np.abs(S_cart.data[0]), axis=2).T, extent=[x.min(), x.max(), z.min(), z.max()], interpolation='none', origin='lower')
            plt.setp(ax[2], ylabel='z', xlabel='x', title='side view')
                        # SH: added beam view
            ax[3].imshow(np.mean(np.abs(S_cart.data[0]) , axis=0).T, extent=[z.min(), z.max(),y.min(), y.max()], interpolation='none', origin='lower')
            plt.setp(ax[3], ylabel='y', xlabel='z', title='front view')
            #plt.draw()
            #plt.pause(.01)
            


plt.show()

#%%

savepath = r'C:\Users\Sanna\Documents\Simulations\save_simulation\recons\%s_projection%i'%(date_str,projection)
#only the projection will be correct    

if not os.path.exists(savepath):
    os.makedirs(savepath)
    print('new folder in this savepath was created:')
    print(savepath)

np.save(savepath + '\\errors', errors)
np.save(savepath + '\\ferrors', ferrors)


for kk, store in enumerate(storage_save):
    
    iter_plotting = save_iter_interval*(kk+1)
    print('plot iteration: ',iter_plotting )
    
    #save as np files and plot
    S_cart = g.coordinate_shift(store, input_space='real', input_system='natural', keep_dims=False)
    
    np.save(savepath + '\\object_iter%i'%(iter_plotting),np.squeeze(S_cart.data)[projection])
    
    x, z, y = S_cart.grids()
    
    np.save(savepath + '\\x_iter%i'%(iter_plotting),x)
    np.save(savepath + '\\y_iter%i'%(iter_plotting),y)
    np.save(savepath + '\\z_iter%i'%(iter_plotting),z)
    np.save(savepath + '\\probe_iter%i'%(iter_plotting),loaded_probeView.data[projection])
    
    #OBS, verkar som att man kan spara storages i np.. testa att öppna. då kanske jag kan spara simulationen också. man kan ske kan spara views också då.