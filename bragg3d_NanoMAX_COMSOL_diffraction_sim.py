#%%
'''
Script for reading COMSOL simulated data and create simulated Bragg diffraction
patterns then perform XRD mapping on those patterns.


This script is adapted from 'bragg3d_initial' written by Alxander Bjoerling. 

    
This is a test-script before writing the ptypy class for simulating diffraction
patterns from COMSOL. 

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


'''
#%% imports


import ptypy 
from ptypy import utils as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
#import mayavi.mlab as mlab
import scipy.interpolate as inter
import sys
#sys.path.insert(0, r'C:\\ptypy_tmp\\tutorial\\Megan_scripts\\')
sys.path.insert(0, r'C:\Users\Sanna\Documents\GitHub\simulated_nanodiffraction')
#C:\Users\Sanna\Documents\GitHub\simulated_nanodiffraction
import read_COMSOL_data
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
# define the experiment                        x         z       y                x    z    y    IS THIS TRUE?
#g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(200, 300, 300), energy=9.49, distance=1.149, theta_bragg=11)
# memory error:
#g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(120, 200, 200), energy=9.49, distance=1.149, theta_bragg=11)
#g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(22, 200, 70), energy=9.49, distance=1.149, theta_bragg=11) 



 #InP:10.91  GaInP: 11.09  #51
g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(51, 151, 151), energy=9.49, distance=1.149, theta_bragg=10.91, propagation = "farfield")  
# test to see if beta 19nm strain segment changes
#g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(70, 125, 125), energy=9.49, distance=1.149, theta_bragg=11, propagation = "farfield")  
#g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(51, 99, 101), energy=9.49, distance=1.149, theta_bragg=11, propagation = "farfield")  
#g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(5, 7, 13), energy=9.49, distance=1.149, theta_bragg=11, propagation = "farfield")  

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
Ny = 11
Nz = 82
"""TEEEEEM.................................................................................................P"""
#Nz=165
Npos = Nz*Ny
positions = np.zeros((Npos,3))
# stepsize as fed to motors
dy_prime = 0.039293162000000117E-6
dz_prime = 0.030055618286132811E-6
"""TEEEEEM.................................................................................................P"""

#dz_prime = 5E-9
#
## start this far away from the center point of the wire
dz_center = 630E-9
"""TEEEEEM.................................................................................................P"""
#dz_center = -600E-9

##real positions where beam hits sample
dy = dy_prime
dz = dz_prime
## if the sample is tilted with theta
#dz = dz_prime*g.costheta
#dx = dz_prime*g.sintheta

z_positions = np.repeat(dz*np.linspace(-np.round(Nz/2), Nz/2, Nz) +dz_center , Ny)
y_positions = np.tile(dy*np.linspace(-np.round(Ny/2),Ny/2,Ny), Nz)
## also x-positions changes a bit because the sample is tilted with an angle
## theta.  Use Nz here. No! I did not define the sample as being tilted in this 
## simulation. remove x_positions
##x_positions = np.repeat(dx*np.linspace(-np.round(Nz/2),Nz/2,Nz) , Ny)
#
#positions[:,0] = x_positions
positions[:,1] = z_positions
positions[:,2] = y_positions
##%%
#---------------
# only few positions, scanning in y
#-------------
#Npos = 1
#positions = np.zeros((Npos,3))
#positions[:, 2] = np.arange(Npos) - Npos/2.0
#positions *= 10e-9 #50e-9

#20 nm in r1

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

# define sample name (name of COMSOL data output-file)
#sample = 'full_segmented_NW_InP_InGaP_20190610'    #nyss

#sample = 'full_segmented_NW_InP_InGaP_20191022'    #this one i am currently using 20191024
#sample = 'full_segmented_NW_InP_InGaP_20191024'     # this one is a test with the INGaP gradient
sample = 'full_segmented_NW_InP_InGaP_20191029'     # updated version with strain mismatch 1.5

#sample = 'full_segmented_NW_InP_InGaP_20190828' (including 19 segment)
#sample = '170'   

# choose displacement u,v, or w
uvw = 5 # 3 4 5 = u v w   # for 111 should be 5

# choose domain to plot (or None if file does not have domains )
# TODO only correct for domain 3 or None. For the InGaP it tries to interpolate the values where the InP segment is
domain = 'InP_357911' # InP_357911' #'InGaP_24681012'  #'InP_357911'    


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
            # set the unwanted data to 0 (InP segment)
            row[-2] = 0
            raw_data.append(row)

    raw_data = np.asarray(raw_data)
    
elif domain == None:
    raw_data = np.asarray(file1)
else:
    sys.exit('u shose the wrong domain number')
    
    
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
#------------------------------------------------------
#TODO  something is not working in python3
def scatter_comsol():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    step = 20
    sc=ax.scatter(raw_data[::step,0],raw_data[::step,1],raw_data[::step,2], c=raw_data[::step,3], marker ='o', cmap='jet')#,alpha=1)
    plt.title('displacement from comsol, every %d:th point'%step)
    plt.colorbar(sc); plt.axis('scaled')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
#scatter_comsol()




#%%  
#-----------------------------------------------------
# Rotate comsol coordinate system before interpolation 
# (In comsol coordinate syste, z is along the wire. x and y is the cross section)
# Following Berenguer notation. (this is the order in which to rotate)
# pos chi rotates counter clockwise around X
# pos phi rotates counter clockwise around Z
# pos psi rotates counter clockwise around Y  - it is the same as the theta motion
coord_rot = rotate_data(raw_data, chi=180 , phi=0 , psi=0)


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
scatter_interpol()
#%%
#Calculate the strain
#-----------------------------------------------------

##InP_Qvect = 18543793660.27452

displacement_slice = np.fliplr(interpol_data[int(g.shape[0]/2)].T)

displacement_slice_NWlength = np.fliplr(interpol_data[int(g.shape[0]/2)].T)[75:100,143:153]

# ska man använda den Q-vektor som sätts at theta, alltså det theta om mäts upp ~~8 deg istället för detta som motsvarar 10.54

strain_dwdz2 = np.diff((displacement_slice ) , axis = 1, append=0) /dz
strain_dwdz = np.gradient(displacement_slice , dz) 

strain_dwdz_NWlength = np.gradient(displacement_slice_NWlength,dz)

#-----------------------------------------------------
# plot the strain
#-----------------------------------------------------
def plot_strain():
    plt.figure()
    plt.title('Interpolated data')
    plt.imshow((interpol_data)[int(g.shape[0]/2)],cmap='jet', origin='lower')
    plt.colorbar()
    
    dz2 = zz[0,256,1] - zz[0,255,0] # dont know why this is different from dz 
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
    plt.imshow(100*strain_dwdz2,cmap='RdBu_r', origin='lower',interpolation='none',extent=[0,dz2*1E6*shape5[1],0,dy2*1E6*shape5[0]])
    #plt.imshow(100*strain_dwdz[1],cmap='RdBu_r', origin='lower',interpolation='none',extent=[0,dz2*1E6*shape5[1],0,dy2*1E6*shape5[0]])
    #plt.title('Strain calc with np.gradient [%]')
    plt.title('Strain calc with np.diff [%]')
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
    
plot_strain()

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
reduce_displacemnt_factor = 1  #-1.0 or 1.0
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

#plt.figure()
#plt.title('np.imag of calculated phase')
#plt.imshow(np.imag(phase[25])); plt.colorbar()

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
fact=1E6
# Plot that
#plt.figure()
#plt.title('Phase stored in the cartesian storage (np.angle)')
#plt.imshow((np.angle(obj_storage_cart.data[0][25])),cmap='jet',interpolation='none')#, extent = []);
#plt.colorbar()
#
#plt.figure()
#plt.suptitle('Phase stored in the cartesian storage')
#plt.subplot(131)
#plt.imshow(sum(np.angle(obj_storage_cart.data[0])),cmap='jet',interpolation='none')#, extent = [])
#plt.xlabel('y'); plt.ylabel('z'); plt.colorbar()
#plt.subplot(132)
#plt.imshow(np.sum(np.abs(obj_storage_cart.data[0]),axis=1))
#plt.xlabel('y'); plt.ylabel('x')
##) extent=[fact*xx.min(), fact*xx.max(), fact*zz.min(), fact*zz.max()], interpolation='none', origin='lower', cmap='jet')
#plt.subplot(133)           
#plt.imshow(np.sum(np.abs(obj_storage_cart.data[0]),axis=2).T)
#plt.xlabel('x '); plt.ylabel(' z')


#fig, ax = plt.subplots(nrows=1, ncols=3)
#plt.suptitle('Phase stored in the cartesian storage')
#ax[0].imshow(np.mean(np.abs(obj_storage_cart.data[0]), axis=2).T, extent=[fact*xx.min(), fact*xx.max(), fact*zz.min(), fact*zz.max()], interpolation='none', origin='lower', cmap='jet')
#plt.setp(ax[0], ylabel='z um', xlabel='x um', title='side view')
#ax[1].imshow(np.mean(np.abs(obj_storage_cart.data[0]), axis=1).T, extent=[fact*xx.min(), fact*xx.max(), fact*yy.min(), fact*yy.max()], interpolation='none', origin='lower', cmap='jet')
#plt.setp(ax[1], ylabel='y um', xlabel='x um', title='top view')
#
#ax[2].imshow(np.mean(np.abs(obj_storage_cart.data[0]), axis=0).T, extent=[fact*zz.min(), fact*zz.max(), fact*yy.min(), fact*yy.max()], interpolation='none', origin='lower', cmap='jet')
#plt.setp(ax[2], ylabel='y um', xlabel='z um', title='front view')
#


# make a copy of the cartesian storage but shifted to natural
obj_storage_natural = g.coordinate_shift(obj_storage_cart, input_system='cartesian', input_space='real', keep_dims=True)

# put the shifted storage data into the original object storage (or are these now the same. or is it a differentce in 
# how the views are connected?)
#this is messy?
obj_storage.data = obj_storage_natural.data




del (obj,  mask_array, phase)

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

#prop_data = abs(g.propagator.fw(views[0].data))**2   #v.data * probeView.data
##try BasicBragg3dPropagator() ?
#inx_slice = g.shape[0]/2

#plt.figure()
#plt.suptitle('Test of propagator, without using any probe')
#plt.subplot(211)
## TODO not correct to take the max nd min here, right. it will give the max of the total thing
#plt.imshow(np.abs(views[0].data[inx_slice]), cmap = 'jet')#, extent = [factor*yy.min(),factor*yy.max(), factor*zz.min(), factor*zz.max()])
#plt.title('view 0')
#plt.xlabel('y [nm]'); plt.ylabel('z [nm]')
#plt.subplot(212)
#plt.imshow(prop_data[inx_slice], cmap = 'jet')
#plt.title('2D cut of the resulting diffraction pattern') 
#

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

choise = 'real'#sample_plane'            # 'square' 'loaded' 'circ' or 'real' 'gauss'

if choise == 'circ':
    fsize = g.shape * g.resolution
    Cprobe = ptypy.core.Container(data_dims=2, data_type='complex128')
    Sprobe = Cprobe.new_storage(psize=g.resolution, shape=200)
    zi, yi = Sprobe.grids()
    apert = u.smooth_step(fsize[1]/5-np.sqrt(zi[0]**2+yi[0]**2), 0.2e-6)
    Sprobe.fill(apert)

elif choise == 'square':    
    # First set up a two-dimensional representation of the probe, with
    # arbitrary pixel spacing. 
    # make a 50 x 50 nm probe (current size of 1 view)
    Cprobe = ptypy.core.Container(data_dims=2, data_type='complex128')
    Sprobe = Cprobe.new_storage(psize=g.resolution[1], shape=256)
    zi, yi = Sprobe.grids()
    #apert = u.smooth_step(1./5-np.abs(zi), 5e-7)*u.smooth_step(1./5-np.abs(yi), 2e-6)
    #Sprobe.fill(apert)
    # square probe
    #Sprobe.data[(yi > -50.0e-8) & (yi < 50.0e-8) & (zi > -50.0e-8) & (zi < 50.0e-8)] = 1
    Sprobe.data[(yi > -3.0e-8) & (yi < 3.0e-8) & (zi > -3.0e-8) & (zi < 3.0e-8)] = 1
    
if choise == 'gauss':
    Cprobe = ptypy.core.Container(data_dims=2, data_type='complex128')
    Sprobe = Cprobe.new_storage(psize=g.resolution, shape=100)
    zi, yi = Sprobe.grids()
    std_dev = 40E-9
    # gaussian probe
    Sprobe.data = np.roll(np.exp(-zi**2 / (2 * (std_dev)**2) - yi**2 / (2 * (std_dev)**2)), 100, axis=1)

elif choise == 'real':   
    
    loaded_profile = np.load('C:/Users/Sanna/Documents/beamtime/NanoMAX062017/Analysis_ptypy/nice_probe_ptyrfiles/scan10/probe10_focus.npy')
    # center the probe (cut out the center part)
    ###################
    "               OOOOOOOOOOOOOOOBS ROTATE. rot90,3 is correct"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    

    loaded_profile_cut = (np.rot90(np.copy(loaded_profile)[1:121,0:120],3))
    # save a psize, shape and the array data in the contaioner
    Cprobe = ptypy.core.Container(data_dims=2, data_type='complex128')
    #TODO dont know if this is right!?
    Sprobe = Cprobe.new_storage(psize=[ 1.89051824e-08,   1.85578409e-08], shape=loaded_profile_cut.shape[0]) 
### resolution from:    g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(2*1E-2, 55*1E-6, 55*1E-6), shape=(51, 128, 128), energy=9.49, distance=1.0, theta_bragg=11)
    
    # fill storage
    Sprobe.fill(0.0)
    Sprobe.fill(1j*loaded_profile_cut)
    zi, yi = Sprobe.grids()
    
   
elif choise == 'sample_plane':    # this loads the probe in the sample plane not in focus. try this too
    
    path_probe = 'C:/Users/Sanna/Documents/beamtime/NanoMAX062017/Analysis_ptypy/nice_probe_ptyrfiles/scan10/scan10_pilatus_ML.ptyr'
    # load all variables in the ptyr file
    import h5py
    with h5py.File(path_probe, 'r') as hf:
        scanid = str(hf['content/probe'].keys()).split("['", 1)[1].split("']")[0]
        probe = np.array(hf.get('content/probe/%s/data' % scanid))
        psize = np.array(hf.get('content/probe/%s/_psize' % scanid))
        #TODO maybe this changes things
        origin = np.array(hf.get('content/probe/%s/_origin' % scanid)) 
        
    Cprobe = ptypy.core.Container( data_type='complex128')
    Sprobe = Cprobe.new_storage(psize=psize, shape=probe.shape)
    # fill storage
    Sprobe.fill(0.0)
    Cprobe.fill(probe)
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
   
#fig = u.plot_storage(Sprobe, 11, channel='c') 

# In order to put some physics in the illumination we set the number of
# photons to 1 billion
nbr_photons = 1E9
Sprobe.data *= np.sqrt(nbr_photons/np.sum(Sprobe.data*Sprobe.data.conj()))
print( u.norm2(Sprobe.data)    )

# prepare in nearfield many times so its 3d
#import nmutils.utils
#dist = np.linspace(-1000, 1000, 200) * 1e-6
#field3d = nmutils.utils.propagateNearfield(Sprobe.data[0], psize of reconstruction, dist, g.energy)

# propagate probe in near field to sample plane 
ill = Sprobe.data[0]

#fig, ax = plt.subplots()
#im = ax.imshow((np.angle(np.squeeze(ill))),cmap='hsv'); plt.colorbar(im)

#50 150 500
#g2 = ptypy.core.geometry.Geo(psize=( 55e-06,   55e-06), shape=( 120, 120), energy=9.49, distance=4.2, propagation = "farfield")  

#for using propagation

dist = 75*1e-6
g2 = ptypy.core.geometry.Geo(psize=( 1.89051824e-08,   1.85578409e-08), shape=( 120, 120), energy=9.49, distance=dist, propagation = "nearfield")  
propagated_ill = g2.propagator.bw(ill) 
#propagated_ill = g2.propagator.fw(ill)   


fig, ax = plt.subplots(); plt.title('Probe propagated in near field %d um'%(dist*1e6))
im = ax.imshow((np.abs(np.squeeze(propagated_ill))),cmap='jet'); plt.colorbar(im)


Sprobe.fill(0.0)
Cprobe.fill(propagated_ill)


    
Sloaded_probe_3d = g.prepare_3d_probe(Sprobe, system='natural', layer=0)#NOTE usually its the input system you specify but here its the output. Also there is an autocenter 
loaded_probeView = Sloaded_probe_3d.views[0]

#TODO # propagate probe to transmission (as a test)


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
plt.figure() #extent : scalars (left, right, bottom, top)
#extent is checked
plt.suptitle('Central cut-plot from the 3d probe \n extruded from 2d probe in quasi vertical zi and y coord. psize=%f nm'%(Sloaded_probe_3d.psize[1]*1E9))
plt.subplot(121)
#    ax[-1].imshow(np.mean(np.abs( views[i].data + (loaded_probeView.data/loaded_probeView.data.max()) ), axis=1), vmin=0, extent=[mufactor*r2.min(), mufactor*r2.max(), mufactor*r3.min(), mufactor*r3.max()])
#    plt.setp(ax[-1], xlabel='r2 [um]', ylabel='r3', xlim=[mufactor*r2.min(), mufactor*r2.max()], ylim=[mufactor*r3.min(), mufactor*r3.max()], yticks=[])
#    # diffraction
plt.imshow((abs(loaded_probeView.data[r3_slice])), extent=[-fac1*r2_slice*Sloaded_probe_3d.psize[2], fac1*r2_slice*Sloaded_probe_3d.psize[2], -fac1*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1],fac1*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1]], cmap='jet', interpolation='none')
#plt.imshow(abs() extent=[-1E9*r2_slice*g.psize[2], 1E9*Sloaded_probe_3d.shape[3]/2*Sloaded_probe_3d.psize[2], -1E9*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1],1E9*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1]] ,cmap='jet', interpolation='none')
plt.xlabel('r2 [nm]'); plt.ylabel('r1 [nm]')#;plt.colorbar()
plt.title('Amplitude')
plt.subplot(122)
#plt.imshow(np.angle(loaded_probeView.data[r3_slice]), extent=[-fac1*r2_slice*Sloaded_probe_3d.psize[2], fac1*r2_slice*Sloaded_probe_3d.psize[2], -fac1*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1],fac1*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1]], cmap='jet', interpolation='none')
plt.imshow(np.angle(loaded_probeView.data[r3_slice]))#, extent=[-fac1*r2_slice, fac1*r2_slice, -fac1*r1_slice,fac1*r1_slice], cmap='jet', interpolation='none')
plt.xlabel('r2 [nm]'); plt.ylabel('r1 [nm]')#;plt.colorbar()
plt.title('Phase')
plt.tight_layout()

#%%
#------------------------------------------------------
#  Visualize the probe extruded in 3d more . wrong axes. 
#------------------------------------------------------
#
# plot probe in 2d cuts (cannot do 3d cuts in matplotlib)
def plot3ddata(data):
    plt.figure()
    plt.suptitle('3d probe central cut plots. OBS origin lower')
    plt.subplot(121)
    #plt.title('-axis')
    plt.imshow((abs((data[data.shape[0]/2,:,:]))),origin='lower', cmap='jet', interpolation='none') 
    plt.xlabel('y'); plt.ylabel('z\'')
    plt.subplot(122)
#    plt.imshow(np.transpose(abs(data[:,data.shape[1]/2,:])), cmap='jet', interpolation='none') 
#    plt.xlabel('x? k_i?'); plt.ylabel('y')
    #plt.subplot(223)
    #plt.title('-axis')
    plt.imshow(np.transpose(abs(loaded_probeView.data[:,:,data.shape[2]/2])),origin='lower', cmap='jet', interpolation='none')#, extent=[-1E9*Sloaded_probe_3d.shape[3]/2*Sloaded_probe_3d.psize[2], 1E9*Sloaded_probe_3d.shape[3]/2*Sloaded_probe_3d.psize[2], -1E9*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1],1E9*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1]])
    plt.xlabel('x'); plt.ylabel('z\'')
    
#plot3ddata(np.squeeze(Sloaded_probe_3d.data))
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
diff3 = diff_Cont.new_storage(psize=np.array([ g.dq3, g.dq1, g.dq2]), shape=pr_shape)# add center to ba at qabs ? 

# Calculate diffraction patterns by using the geometry's propagator. all in 3d
# todo is this OK lam factor?
lam_factor =  nbr_photons*25   # higher number (less total intensity/signal)
lam_factor =  nbr_photons*10   # Ganska bra! higher number (less total intensity/signal)

lam_factor =  nbr_photons*5





diff2 = []
print( 'calculating diffraction data')
for v in views:
    exit_wave = v.data * loaded_probeView.data 
    
    prop_exit_wave = g.propagator.fw(exit_wave)
    # without noise
    #diff2.append(np.array((np.real( prop_exit_wave*prop_exit_wave.conj() )),float)) #this is actallu real but data type does not change    
    #diff2.append(np.abs( prop_exit_wave)**2) # this gives the same strain maps but still diffraction patterns looks a bit different
    # with noise
    diff2.append(np.array(np.random.poisson(np.real( prop_exit_wave*prop_exit_wave.conj() )/lam_factor),float)) #this is actallu real but data type does not change
    

diff3.fill(np.array(diff2))

del (exit_wave,prop_exit_wave)


# make 0 values white instead of colored
for diff in diff2:
    diff[diff==0] = np.inf



#del diff, zz, xx, yy


#plot single postion diffraction in 3 projections()


#%%
# plot object and diffraction pattern at the same time. Wrong axes here!?!?
#----------------------------------

plot_view = 0
#357#Npos/2 #which view to plot
factor = 1E6
rec_fact = 1E-10
#fig = u.plot_storage(pr, 0)
#plt.figure()# extent guide: extent : scalars (left, right, bottom, top)
#plt.suptitle('Final diffraction pattern using object and probe', fontsize=15)
xcut = int(views[0].shape[0]/2) # which y-z-cut to plot
#plt.subplot(221)
#plt.imshow(np.abs(views[plot_view].data[xcut]), cmap = 'jet', extent = [factor*yy_vgrid.min(),factor*yy_vgrid.max(), factor*zz_vgrid.min(), factor*zz_vgrid.max()])
#plt.title('One views object abs data')
#plt.xlabel('$r_1$ [$\mathrm{\mu m}$]'); plt.ylabel('$r_2$ [$\mathrm{\mu m}$]')
#
#plt.subplot(222)
#plt.imshow((abs(loaded_probeView.data[xcut ,:,:])),extent = [factor*yy_vgrid.min(),factor*yy_vgrid.max(), factor*zz_vgrid.min(), factor*zz_vgrid.max()])# extent=[-factor*Sloaded_probe_3d.shape[3]/2*Sloaded_probe_3d.psize[2], factor*Sloaded_probe_3d.shape[3]/2*Sloaded_probe_3d.psize[2], -factor*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1],factor*Sloaded_probe_3d.shape[2]/2*Sloaded_probe_3d.psize[1]])
#plt.title('One views probe abs data')
#plt.xlabel('$r_1$ [$\mathrm{\mu m}$]'); plt.ylabel('$r_2$ [$\mathrm{\mu m}$]')
#
#plt.subplot(223)
#plt.title('Same view\'s diffraction pattern')
#plt.imshow(diff2[plot_view][xcut].T , extent = [ -(g.dq1*g.shape[1]/2 )*rec_fact, g.dq1*g.shape[1]/2*rec_fact, -g.dq2*g.shape[2]/2*rec_fact, g.dq2*g.shape[2]/2*rec_fact], interpolation='none',cmap='jet')
#plt.xlabel('$q_1$ [$\mathrm{\AA^{-1}}$]'); plt.ylabel('$q_2$ [$\mathrm{\AA^{-1}}$]'); plt.colorbar()#$10^6$ 
#plt.tight_layout()

#plt.savefig('aaa')

# compare the cuts to 2d diffraction patterns
prop_exitwave_2d = g.propagator.fw( np.array(views[plot_view].data[xcut]*loaded_probeView.data[xcut ,:,:]))
diff_2d = np.array(np.random.poisson(np.real( prop_exitwave_2d*prop_exitwave_2d.conj() )/lam_factor), float) 
diff_2d[diff_2d==0] = -np.nan

#plt.figure()
#plt.title('2d Diffraction ')
#plt.imshow(diff_2d.T, extent = [-g.dq1*g.shape[1]/2*rec_fact, g.dq1*g.shape[1]/2*rec_fact,-g.dq2*g.shape[2]/2*rec_fact, g.dq2*g.shape[2]/2*rec_fact ], interpolation='none',cmap='jet')
#plt.xlabel('$q_1$ [$\AA^{-1}$]'); plt.ylabel('$q_2$ [$\AA^{-1}$]')#; plt.colorbar()#$10^6$ 
#plt.tight_layout()


def make_finite(matrix):
    mask = np.isinf(matrix)
    matrix[mask] = 0
    return matrix
diff2_finite = make_finite(np.copy(diff2))    

#lineplot of diffraction pattern (to measure size of object from fringes)
#plt.figure()
#plt.plot(diff2_finite[plot_view][xcut][:][0].T)
#real space size of object =    
rss_obj = 22*g.resolution[1]







#%%
# ------------------------------------------------------    
# Visualize a single field of view with probe and object
# ------------------------------------------------------

# In order to visualize the field of view, we'll create a copy of the
# object storage and set its value equal to 1 where covered by the first
# view.

def plot_probe_sample(frame):
    
    S_display = obj_storage.copy(owner=obj_container)#, ID='S_display')
    S_display.fill(0.0)
    #S_display[obj_storage.views[frame]] = 1
    
    # Then, to see how the probe is contained by this field of view, we add
    # the probe and the object itself to the above view.
    S_display[obj_storage.views[frame]] +=1*(loaded_probeView.data)/loaded_probeView.data.max()
    S_display.data += obj_storage.data
    
    # To visualize how this looks in cartesian real space, make a shifted
    # (nearest-neighbor interpolated) copy of the object Storage.
    S_display_cart = g.coordinate_shift(S_display, input_system='natural', input_space='real', keep_dims=False)
    
    fact=1E6
    # Plot that
    fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.suptitle('Position: %d'%frame)
    x, z, y = S_display_cart.grids()
    ax[0].imshow(np.mean(np.abs(S_display_cart.data[0]), axis=2).T, extent=[fact*x.min(), fact*x.max(), fact*z.min(), fact*z.max()], interpolation='none', origin='lower', cmap='jet')
    plt.setp(ax[0], ylabel='z um', xlabel='x um', title='side view')
    ax[1].imshow(np.mean(np.abs(S_display_cart.data[0]), axis=1).T, extent=[fact*x.min(), fact*x.max(), fact*y.min(), fact*y.max()], interpolation='none', origin='lower', cmap='jet')
    plt.setp(ax[1], ylabel='y um', xlabel='x um', title='top view')
    
    ax[2].imshow(np.mean(np.abs(S_display_cart.data[0]), axis=0).T, extent=[fact*z.min(), fact*z.max(), fact*y.min(), fact*y.max()], interpolation='none', origin='lower', cmap='jet')
    plt.setp(ax[2], ylabel='y um', xlabel='z um', title='front view')
    
    #plt.savefig('C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/InGaP_InP_full_NW/real_probe/positions/probe_pos_%d'%frame)

frame = 422
#plot_probe_sample(frame)

#frames = np.arange(0,200,1)
#for frame in frames:
#    plot_probe_sample(frame)

#%%
# --------------------------------------------
# Visualize the probe positions along the scan
# --------------------------------------------


# beam/sample overlap in non-orthogonal coordinates
import matplotlib.gridspec as gridspec
#plt.figure()
#gs = gridspec.GridSpec(4, 2, width_ratios=[3,1], wspace=.0)
#ax, ax2 = [], []
#
#central_cut = int(diff2[0].shape[0]/2) 
#mufactor=1E6
#for i in range(4):#len(obj_storage.views)):
#    # overlap
#    ax.append(plt.subplot(gs[i, 0]))
#    ax[-1].imshow(np.mean(np.abs( views[i].data + (loaded_probeView.data/loaded_probeView.data.max()) ), axis=1), vmin=0, extent=[mufactor*r2.min(), mufactor*r2.max(), mufactor*r3.min(), mufactor*r3.max()])
#    plt.setp(ax[-1], xlabel='r2 [um]', ylabel='r3', xlim=[mufactor*r2.min(), mufactor*r2.max()], ylim=[mufactor*r3.min(), mufactor*r3.max()], yticks=[])
#    # diffraction
#    ax2.append(plt.subplot(gs[i, 1]))
#    ax2[-1].imshow((diff2[i][central_cut,:,:].T),cmap='hot',extent = [-g.dq1*g.shape[1]/2*rec_fact, g.dq1*g.shape[1]/2*rec_fact,-g.dq2*g.shape[2]/2*rec_fact, g.dq2*g.shape[2]/2*rec_fact], interpolation='none')
#    plt.setp(ax2[-1], ylabel='q2 [nm$^{-1}$]', xlabel='q1 [nm$^{-1}$]', xticks=[], yticks=[])
#plt.suptitle('Normalized probe, sample, and slices of 3d diffraction peaks along the scan')
#plt.draw()
#plt.tight_layout() 


#interval=80
#for ii in range(0,Npos, interval):
#    plt.figure()
#    plt.suptitle('position: %d %d %d nm from (0,0,0)'%(positions[ii,0]*1E9,positions[ii,1]*1E9,positions[ii,2]*1E9))
#    plt.subplot(121)
#    plt.title('from side')
#    plt.imshow(np.mean(np.abs( views[ii].data + (loaded_probeView.data/loaded_probeView.data.max()) ), axis=2).T, vmin=0, extent=[mufactor*r2.min(), mufactor*r2.max(), mufactor*r3.min(), mufactor*r3.max()],origin='lower')
#    plt.xlabel('r2 [um]'); plt.ylabel('r3');# plt.xlim([mufactor*r2.min(), mufactor*r2.max()]); plt.ylim([mufactor*r3.min(), mufactor*r3.max()])
#    plt.subplot(122)
#    plt.imshow(diff2[ii][central_cut].T,extent = [ -g.dq1*g.shape[1]/2*rec_fact, g.dq1*g.shape[1]/2*rec_fact, -g.dq2*g.shape[2]/2*rec_fact, g.dq2*g.shape[2]/2*rec_fact], interpolation='none',cmap='jet')
#    plt.xlabel('q1 [$\AA^{-1}$]'); plt.ylabel('q2 [$\AA^{-1}$]'); plt.colorbar()#$10^6$ 
#    plt.tight_layout()
##plt.draw()
##plt.tight_layout()



   

#%% 

print( 'start analysis')

del (diff2)

#%%
# --------------------------------------------
# plot the sum of all used diffraction images
# --------------------------------------------   

#plt.figure()
#plt.imshow(np.log10(sum(sum((diff3.data)))),cmap='jet', interpolation='none')
#plt.title('Simulated summed intensity')
#plt.colorbar()
#plt.savefig('savefig\summed_intensity')

#%%
# --------------------------------------------
# Do bright field analysis    
# --------------------------------------------
    
def bright_field_voxels(data,x,y):
    index = 0
    photons = np.zeros((y,x)) 
    for row in range(0,y):
        for col in range(0,x):
            photons[row,col] = np.sum(data[index]) #/ max_intensity
            index += 1    
           # import pdb
           # pdb.set_trace()
           
    return photons
    
BF_voxels = bright_field_voxels(diff3,Ny,Nz)

#plt.figure()
#plt.imshow( BF_voxels.T, cmap='jet', interpolation='none')#, extent=[ 0, 1E6*dz*Nz,0, 1E6*dy*(Ny-1)]) 
#plt.title('Bright field from voxels ')#sorted in gonphi %d'%scans_sorted_theta[ii][1])  
#plt.xlabel('$x$ [$\mathrm{\mu}$m]') 
#plt.ylabel('$y$ [$\mathrm{\mu}$m]')


  

#%%    
# define q1 q2 q3 + q_abs from the geometry function 
# (See "Bending and tilts in NW..." pp)
def def_q_vectors():
    global q3, q1, q2, q_abs    
    #  units of reciprocal meters [m-1]
    q_abs = 4 * np.pi / g.lam * g.sintheta
       
    q1 = np.linspace(-g.dq1*g.shape[1]/2.+q_abs/g.costheta, g.dq1*g.shape[1]/2.+q_abs/g.costheta, g.shape[1]) #        ~z
    # q3 defined as centered around 0, that means adding the component from q1
    q3 = np.linspace(-g.dq3*g.shape[0]/2. + g.sintheta*q1.min() , g.dq3*g.shape[0]/2.+ g.sintheta*q1.max(), g.shape[0])  #    ~~x  
    q2 = np.linspace(-g.dq2*g.shape[2]/2., g.dq2*g.shape[2]/2., g.shape[2]) #         ~y
def_q_vectors()

# --------------------------------------------------------------
# Make a meshgrid of q3 q1 q2 and transform it to qx qz qy.
# Also define the vectors qx qz qy
#----------------------------------------------------------------





#TODO     THIS def of Q-space is not Correct now because it does not correspond to how the data is stored
# the diffraction patterns are stored as [rot,det_width,det_height]



# in the transformation is should be input and output: (qx, qz, qy), or (q3, q1, q2).
# make q-vectors into a tuple to transform to the orthogonal system; Large Q means meshgrid, small means vector
"changed this to match data"
Q3,Q1,Q2 = np.meshgrid(q3, q1, q2, indexing='ij') 
tup = Q3, Q1, Q2   
Qx, Qz, Qy = g.transformed_grid(tup, input_space='reciprocal', input_system='natural')




#plot_QxQyQz()





#corrected
qx = np.linspace(Qx.min(),Qx.max(),g.shape[0])
qz = np.linspace(Qz.max(),Qz.min(),g.shape[1])
qy = np.linspace(Qy.min(),Qy.max(),g.shape[2])

#%%
# Testing the shifting (could be removed)

# check which positions has the most intensity, for a nice 3d Bragg peak plot
pos_vect_naive = np.sum(np.sum(np.sum(diff3.data, axis =1), axis =1), axis =1) 
max_pos_naive = np.argmax(pos_vect_naive)

max_pos_naive = 423
#528# 512# 

# hitta position av segment:
#np.sum(diff3[6*165+106])
#max_pos_naive = 5*165+106


# find where the peak is in the detector plane
q2max = np.argmax(np.sum(sum(diff3.data[max_pos_naive]),axis=0))
q1max = np.argmax(np.sum(sum(diff3.data[max_pos_naive]),axis=1))
q3max = np.argmax(np.sum(np.sum(diff3.data[max_pos_naive],axis=1),axis=1))
# save one bragg peak to plot. set 0 values to inf   
plot_3d_naive = np.copy(diff3[max_pos_naive])
#plot_3d_naive[plot_3d_naive==0] = np.inf

# plot the 'naive' 3d peak of the most diffracting position, and centered on the peak for q1 and q2 
factor = 1E-10 
plt.figure()
plt.suptitle('Naive plot of single position Bragg peak in natural coord system')
plt.subplot(221)
# extent (left, right, bottom, top) in data coordinates
plt.imshow(plot_3d_naive[int(51/2)], cmap='jet', interpolation='none', extent=[ q2[-1]*factor, q2[0]*factor,q1[0]*factor, q1[-1]*factor])
plt.ylabel(r'$q_1$ $ (\AA ^{-1}$)') ; plt.colorbar()
plt.xlabel('$q_2$ $ (\AA ^{-1}$)')   
plt.subplot(222)
plt.imshow(plot_3d_naive[:,int(q2max),:], cmap='jet', interpolation='none', extent=[q2[0]*factor, q2[-1]*factor, q3[-1]*factor, q3[0]*factor])
plt.ylabel('$q_3$ $ (\AA ^{-1}$)'); plt.colorbar()
plt.xlabel('$q_2$ $ (\AA ^{-1}$)') 
plt.subplot(223)
plt.imshow(plot_3d_naive[:,:,int(q1max)], cmap='jet', interpolation='none', extent=[q1[0]*factor, q1[-1]*factor, q3[-1]*factor, q3[0]*factor])
plt.ylabel('$q_3$ $ (\AA ^{-1}$)'); plt.colorbar() 
plt.xlabel('$q_1$ $ (\AA ^{-1}$)') 

test_shift_coord = ptypy.core.geometry_bragg.Geo_Bragg.coordinate_shift(g, diff3, input_space='reciprocal',
                         input_system='natural', keep_dims=False,
                         layer=max_pos_naive)




## functions to plot and save a 3d bragg peak(cuts)
# send in the bragg peak
def plot_bragg_peak(container,frame, save_key = 0):
    
    #test_shift_coord.data[0] = 30*test_shift_coord.data[0]/test_shift_coord.data[0].max()
    # make 0 values white instead of colored
    # TODO : make a scatter plot instead tp check that all axis are correct, make it more visible. 
    #but remember that the data is flipped in qx(rot)axis here
    
    q2max = np.argmax(np.sum(sum(container.data[0]),axis=0))
    q1max = np.argmax(np.sum(sum(container.data[0]),axis=1))
    q3max = np.argmax(np.sum(np.sum(container.data[0],axis=1),axis=1))
    #print(q3max)
    container.data[0][container.data[0]<0.05] = np.inf    
    #test_shift_coord.data[0][test_shift_coord.data[0]<6E7] = np.inf 
    factor = 1E-10  #if you want to plot in reciprocal m or Angstroms, user 1 or 1E-10
    plt.figure()
    #plt.suptitle('Single position Bragg peak in orthogonal system \n (Berenguer terminology) qxqzqy frame:%d'%frame)
    
    #plt.subplot(311)
    #plt.imshow((container.data[0][q3max]), cmap='jet', interpolation='none', extent=[ qy[-1]*factor, qy[0]*factor, qz[-1]*factor, qz[0]*factor])
    # extent left, right, bottom, top
    plt.imshow((np.rot90(container.data[0][q3max],k=3 )), cmap='jet', interpolation='none', extent=[ qz[-1]*factor, qz[0]*factor, qy[0]*factor, qy[-1]*factor])
    plt.xlabel('$q_z$ $ (\AA ^{-1}$)')      
    plt.ylabel('$q_y$ $ (\AA ^{-1}$)'); # plt.colorbar()
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    if save_key ==1:
        plt.savefig('C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/InGaP_InP_full_NW/real_probe/InP_bragg_slices/nice_ones/%s_bragg_slices_pos_%d_1'%(date_str,frame))
        plt.savefig('C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/InGaP_InP_full_NW/real_probe/InP_bragg_slices/nice_ones/%s_bragg_slices_pos_%d_1.pdf'%(date_str,frame))
    #plt.subplot(312)
    for i in range(-2,2):
        plt.figure()
        # should be x angainst y in these labels
        plt.imshow((container.data[0][:,q1max+i,:]), cmap='jet', interpolation='none', extent=[qy[0]*factor, qy[-1]*factor, qx[0]*factor, qx[-1]*factor])
        plt.ylabel('$q_x$ $ (\AA ^{-1}$)'); plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel('$q_y$ $ (\AA ^{-1}$)')     
        plt.tight_layout()
        #plt.subplot(313)
    if save_key ==1:
        plt.savefig('C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/InGaP_InP_full_NW/real_probe/InP_bragg_slices/nice_ones/%s_bragg_slices_pos_%d_2'%(date_str,frame)) 
        plt.savefig('C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/InGaP_InP_full_NW/real_probe/InP_bragg_slices/nice_ones/%s_bragg_slices_pos_%d_2.pdf'%(date_str,frame))
    plt.figure()
    plt.imshow((container.data[0][:,:,q2max]), cmap='jet', interpolation='none', extent=[qz[-1]*factor, qz[0]*factor, qx[0]*factor, qx[-1]*factor])
    plt.ylabel('$q_x$ $ (\AA ^{-1}$)'); plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel('$q_z$ $ (\AA ^{-1}$)') 
    plt.tight_layout()
    
    container.data[0][container.data[0]==np.inf] = 0
    if save_key ==1:
        plt.savefig('C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/InGaP_InP_full_NW/real_probe/InP_bragg_slices/nice_ones/%s_bragg_slices_pos_%d_3'%(date_str,frame))
        plt.savefig('C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/InGaP_InP_full_NW/real_probe/InP_bragg_slices/nice_ones/%s_bragg_slices_pos_%d_3.pdf'%(date_str,frame)) 

plot_bragg_peak(test_shift_coord,max_pos_naive, save_key=1)










#%%


###############################################################################
# XRD analysis
###############################################################################


# input is 4d matrix with [nbr_diffpatterns][nbr_rotations][nbr_pixels_x][nbr_pixels_y]
def COM_voxels_reciproc(data, vect_Qx, vect_Qz, vect_Qy ):

    # meshgrids for center of mass calculations in reciprocal space
    COM_qx = np.sum(data* vect_Qx)/np.sum(data)
    COM_qz = np.sum(data* vect_Qz)/np.sum(data)
    COM_qy = np.sum(data* vect_Qy)/np.sum(data)

    #print( 'coordinates in reciprocal space:')
    #print( COM_qx, COM_qz, COM_qy)
    return COM_qx, COM_qz, COM_qy

# loop through all scanning postitions and move the 3D Bragg peak from the 
# natural to the orthogonal coordinate system (to be able to calculate COM)
# Calculate COM for every peak - this gives the XRD matrices
nbr_rows = Nz
nbr_cols = Ny
def XRD_analysis():
    position_idx = 0
    XRD_qx = np.zeros((nbr_rows,nbr_cols))
    XRD_qz = np.zeros((nbr_rows,nbr_cols))
    XRD_qy = np.zeros((nbr_rows,nbr_cols))

    for row in range(0,nbr_rows):
        for col in range(0,nbr_cols):
            
            # if keep_dims is False, shouldnt the axis qz change? (q1 -->qz)
            data_orth_coord = ptypy.core.geometry_bragg.Geo_Bragg.coordinate_shift(g, diff3, input_space='reciprocal',
                         input_system='natural', keep_dims=True,
                         layer=position_idx)         # layer is the first col in P.diff.storages.values()[0]
      
            # do the 3d COM analysis to find the orthogonal reciprocal space coordinates of each Bragg peak         
            COM_qx, COM_qz, COM_qy = COM_voxels_reciproc(data_orth_coord.data[0][::-1,:,:],Qx, Qz, Qy)

            # insert coordinate in reciprocal space maps 
            XRD_qx[row,col] = COM_qx
            XRD_qz[row,col] = COM_qz
            XRD_qy[row,col] = COM_qy
            #import pdb
            #pdb.set_trace()
            # save figures:
            #plot_bragg_peak(data_orth_coord,position_idx)
            
            
            position_idx +=1
#            #plot every other 3d peak and print out the position of the COM analysis
#            if (position_idx%100==0):
#                #import pdb; pdb.set_trace()
#                # TODO very har to say anything about this looking in 2d, need 3d plots!
#                x_p = np.argwhere(qx>COM_qx)[0][0]
#                y_p = np.argwhere(qy>COM_qy)[0][0] #take the first value in qy where
#                z_p = np.argwhere(qz>COM_qz)[0][0]  
#                print y_p,z_p
#                #import pdb; pdb.set_trace()
#                plt.figure()
#                plt.imshow(sum(data_orth_coord.data[0]), cmap='jet')#, extent=[ qy[0], qy[-1], qz[0], qz[-1] ])
#                # Find the coordinates of that cell closest to this value:              
#                plt.scatter(y_p, z_p, s=500, c='red', marker='x')#, extent=[ qy[0], qy[-1], qz[0], qz[-1] ])
#                plt.title('Single Bragg peak summed in x. COM z and y found approx at red X')

    return XRD_qx, XRD_qz, XRD_qy, data_orth_coord

XRD_qx, XRD_qz, XRD_qy, data_orth_coord = XRD_analysis() # units of 1/m

#%%
def plot_XRD_xyz():
    factor = 1E-10  #if you want to plot in m or Angstroms, user 1 or 1E-10
    # plot reciprocal space map x y z 
    plt.figure()
    plt.subplot(411)
    plt.imshow(factor*XRD_qx.T, cmap='jet',interpolation='none')#,extent=extent_motorpos)
    plt.title('Reciprocal space map, $q_x$ $ (\AA ^{-1}$) ')
    plt.ylabel('$y$ [$\mathrm{\mu m}$]') 
    plt.colorbar()
    plt.subplot(412)
    plt.imshow(factor*XRD_qy.T, cmap='jet',interpolation='none')#,extent=extent_motorpos) 
    plt.title('Reciprocal space map, $q_y$ $ (\AA ^{-1}$) ')
    plt.ylabel('$y$ [$\mathrm{\mu m}$]') 
    plt.colorbar()
    plt.subplot(413)
    plt.imshow(factor*XRD_qz.T, cmap='jet',interpolation='none')#,extent=extent_motorpos)
    plt.title('Reciprocal space map, $q_z$ $(\AA ^{-1}$) ')
    plt.ylabel('$y$ [$\mathrm{\mu m}$]') 
    plt.colorbar()
    plt.subplot(414)
    plt.title('Bright field (sum of all rotations)')
    plt.imshow(BF_voxels.T, cmap='jet', interpolation='none')#,extent=extent_motorpos) / BF_voxels.max()
    plt.xlabel('$x$ [$\mathrm{\mu m}$]') 
    plt.ylabel('$y$ [$\mathrm{\mu m}$]') 
    plt.colorbar()
#plot_XRD_xyz()
#%% 
#----------------------------------------------------------
# Convert q-vector from  cartesian coordinates to spherical
# (See "Bending and tilts in NW..." pp)
#----------------------------------------------------------

XRD_absq =  np.sqrt(XRD_qx**2 + XRD_qy**2 + XRD_qz**2)
XRD_alpha = np.arcsin( XRD_qy/ XRD_absq)
XRD_beta  = np.arctan( XRD_qx / XRD_qz)

#%%
#---------------------------------
# plot the XRD maps
#----------------------------------

#def plot_XRD_polar():    

# cut the images in x-range:start from the first pixel: 
start_cutXat = 5 
# whant to cut to the right so that the scale ends with an even number
#x-pixel nbr 67 is at 2.0194197798363955
cutXat = 49+5# 43# 165
#49

# replace the x-scales end-postion in extent_motorposition. 
extent_motorpos = [0, dz*(Nz-1)*1E6, 0,dy*(Ny-1)*1E6]                 
extent_motorpos_cut = [0, dz*(cutXat-start_cutXat-1)*1E6,0,dy*(Ny-1)*1E6]

# create a mask from the BF matrix, for the RSM analysis
XRD_mask = np.copy(BF_voxels)
XRD_mask[XRD_mask < 8E4  ] = np.nan   #InGaP 5E4                 InP 3.5E4    8E14   ingap:40E14  1E14      81000   # for homo InP, use 280000
XRD_mask[XRD_mask > 0] = 1       #make binary, all values not none to 1

# plot abs q to select pixels that are 'background', not on wire, and set these pixels to NaN (make them white)
plt.figure()
colormap = 'RdBu_r' #Spectral' #RdYlGn'#coolwarm' # 'bwr' #'PiYG'# #'RdYlBu' # 'PiYG'   # 'PRGn' 'BrBG' 'PuOr'
#plt.suptitle(
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.02)
plt.subplot(411)
plt.title('Summed up intensity', loc='left', pad =-12, color ='white')
plt.imshow((BF_voxels[start_cutXat:cutXat,:].T/ BF_voxels[start_cutXat:cutXat,:].max()), cmap=colormap, interpolation='none',extent=extent_motorpos_cut)
plt.ylabel('y [$\mu m$]')
plt.xticks([])
#po = plt.colorbar(ticks=(10,20,30,40))#,fraction=0.046, pad=0.04) 
po = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=4); po.locator = tick_locator;po.update_ticks()


# if you want no mask use:
#XRD_mask = np.ones((XRD_mask.shape))

plt.subplot(412)   
#calculate lattice constant a from |q|:       
# TODO: this is wrong for the homogenous wires, its not (111), for segmented InP i dont know    
d_lattice =  2*np.pi/  np.copy(XRD_absq)


#print 'mean lattice constant is %d' %np.mean(a_lattice_exp)

mean_strain = np.nanmean(XRD_mask[start_cutXat:cutXat,:]*d_lattice[start_cutXat:cutXat,:])

#TODO try with reference strain equal to the center of the largest segment (for InP) # tody try with reference from the other NWs
#mean_strain = a_lattice_exp[:,start_cutXat:cutXat].max() 


# make the nan-background black
#cmap.set_bad('white',1.)

plt.imshow((100*(XRD_mask[start_cutXat:cutXat,:]* (d_lattice[start_cutXat:cutXat,:]-mean_strain)/mean_strain).T), cmap=colormap,interpolation='none',extent=extent_motorpos_cut)
#plt.imshow( ((XRD_mask[:,start_cutXat:cutXat]*a_lattice_exp[:,start_cutXat:cutXat].T)), cmap='jet',interpolation='none',extent=extent_motorpos_cut) 
#plt.imshow( ((XRD_mask[:,start_cutXat:cutXat]*XRD_absq[:,start_cutXat:cutXat].T)), cmap='jet',interpolation='none',extent=extent_motorpos_cut) 
#plt.imshow( ((XRD_mask[:,start_cutXat:cutXat]*a_sqrt_2a2[:,start_cutXat:cutXat].T)), cmap='jet',interpolation='none',extent=extent_motorpos_cut) 

#plt.title('Relative length of Q-vector |Q|-$Q_{mean}$ $(10^{-3}/\AA$)')
plt.title('(111) Strain $\epsilon$ (%)', loc='left', pad =-12)   #plt.title('Lattice constant a')
plt.ylabel('$y$ [$\mathrm{\mu}$m]')  
plt.xticks([])
po = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=4); po.locator = tick_locator;po.update_ticks()


plt.subplot(413)
plt.imshow(((XRD_mask[start_cutXat:cutXat,:]*1E3*XRD_alpha[start_cutXat:cutXat,:]).T), cmap=colormap,interpolation='none',extent=extent_motorpos_cut) # not correct!
# cut in extent_motorposition. x-pixel nbr 67 is at 2.0194197798363955
plt.title('$\\alpha$ (mrad)', loc='left', pad =-12)
plt.ylabel('$y$ [$\mathrm{\mu m}$]') 
plt.xticks([])
po = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=4); po.locator = tick_locator;po.update_ticks()
#po = plt.colorbar(ticks=(0,1,2,3,4))
#po.set_label('Bending around $q_x$ $\degree$')
   
plt.subplot(414)
plt.imshow(((XRD_mask[start_cutXat:cutXat,:]*1E3*XRD_beta[start_cutXat:cutXat,:]).T), cmap=colormap,interpolation='none',extent=extent_motorpos_cut) # not correct!
plt.title('$\\beta$ (mrad)', loc='left', pad =-12)
plt.ylabel('$y$ [$\mathrm{\mu m}$]') 
plt.xlabel('$x$ [$\mathrm{\mu m}$]') 
po = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=4); po.locator = tick_locator;po.update_ticks()
#po = plt.colorbar(ticks=(5, 10, 15 ))
#po.set_label('Bending around $q_y$ $\degree$')

#plt.savefig('C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/InGaP_InP_full_NW/real_probe/XRD_InGaP/Strained/%s_xrd'%(date_str))
#plt.savefig('C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/InGaP_InP_full_NW/real_probe/XRD_InGaP/Strained/%s_xrd.pdf'%(date_str)) 

#plt.savefig('C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/InGaP_InP_full_NW/real_probe/XRD_InP/Strained/%s_xrd'%(date_str))
#plt.savefig('C:/Users/Sanna/Documents/Simulations/ptypySim/Bragg/InGaP_InP_full_NW/real_probe/XRD_InP/Strained/%s_xrd.pdf'%(date_str)) 
plt.savefig('savefig/xrd')

#plot_XRD_polar()

#%%
