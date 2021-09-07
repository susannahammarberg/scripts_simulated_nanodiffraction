# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:51:11 2020

@author: Sanna
"""


#%%
#---------------------------------------------------------------------
#%%
# plot single postion diffraction in 3 projections
#-----------------------------------------------------------------

#position = 524# 357# 524
#
#plt.figure()# extent guide: extent : scalars (left, right, bottom, top)
#plt.suptitle('Final diffraction pattern in 3 projections', fontsize=13)
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
##-----------------------------------------------------------------------------
## visulaize probe in 3d. 
## visualize diffraction in 3d. a working slice plot
##-----------------------------------------------------------------------------
#def slice_plot():
data = np.log10(abs(loaded_probeView.data))
data[data==-np.inf]=0
mlab_xcut = views[0].shape[0]/2 # which y-z-cut to plot
mlab_zcut = views[0].shape[2]/2
mlab_ycut = views[0].shape[1]/2
#mlab.figure()
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
#                            plane_orientation='x_axes',
#                            slice_index=mlab_xcut,
#                            colormap = 'jet'
#                        )
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
#                            plane_orientation='y_axes',
#                            slice_index=mlab_ycut,
#                            colormap = 'jet'
#                        )
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
#                            plane_orientation='z_axes',
#                            slice_index=mlab_zcut,
#                            colormap = 'jet'
#                            )
#mlab.outline()
# TODO scale correctly
#frame2=357
## 3d cut-plot of the diffraction 
#data = (abs(diff2[frame2]))
#data[data==-np.inf]=0
#mlab.figure()
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
#                            plane_orientation='x_axes',
#                            slice_index=mlab_xcut,
#                            colormap = 'jet'
#                        )
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
#                            plane_orientation='y_axes',
#                            slice_index=mlab_ycut,
#                            colormap = 'jet'
#                        )
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
#                            plane_orientation='z_axes',
#                            slice_index=mlab_zcut,
#                            colormap = 'jet'
#                            )
#mlab.outline()
#%%



#%%
#try with the grid from ptypy?

#Q333,Q111,Q222 = diff3.grids() #choose only one position
#Q1p = Q333[0]; Q2p = Q222[0] ; Q3p = Q333[0]
#del Q111, Q222, Q333
#
## TODO  kollao mdetta grid är samma som det grid jag har gjort
#k=0
#plt.figure()
#plt.suptitle('ptypy reciprocal grid')
#plt.subplot(311)
#plt.title('Q1p')
#plt.imshow(Q1p[k])
#plt.colorbar()
#
#plt.subplot(312)
#plt.title('Q2p')
#plt.imshow(Q2p[k])
#plt.colorbar()
#
#plt.subplot(313)
#plt.title('Q3p')
#plt.imshow(Q2p[k])
#plt.colorbar()
#

"""
# try to load the grid for the q-vectors
# detta Ã¤r real space grid hos detektorn? 0.02 osv. Kan jag transformera den till reciprok?
aa=P.diff.storages['S0000'].grids()
q3_te, q1_te, q2_te = g.transformed_grid(aa, input_space='real', input_system='natural')
vilket storage har grid i reciproca rummet?

Ã„r detta rÃ¤tt?    att gÃ¶ra ett nytt grid och allt Ã¤r bra 2pi/alla vÃ¤rden?
2*np.pi/aa[0][9]     == is the same as q1 q2 q3?


"""



plotting XRD analysis

plot_QxQyQz():
#k=0
#plt.figure()
#plt.subplot(311)
#plt.title('Q1')
#plt.imshow(Q1[k])
#plt.colorbar()
#
#plt.subplot(312)
#plt.title('Q2')
#plt.imshow(Q2[k])
#plt.colorbar()
#
#plt.subplot(313)
#plt.title('Q3')
#plt.imshow(Q3[k])
#plt.colorbar()
#
#plt.figure()
#plt.subplot(311)
#plt.imshow(Qz[k])
#plt.title('Qz')
#plt.colorbar()
#
#plt.subplot(312)
#plt.title('Qy')
#plt.imshow(Qy[k])
#plt.colorbar()
#
#plt.subplot(313)
#plt.title('Qx')
#plt.imshow(Qx[k])
#plt.colorbar()
    
    
    
## change font in plot
#from matplotlib import rc
#font = {'size'   : 16}
#plt.rc('font', **font)
##rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
### for Palatino and other serif fonts use:
##rc('font',**{'family':'serif','serif':['Palatino']})
##rc('text', usetex=True)
#
## change font
#plt.rcParams['font.sans-serif'] = "sans-serif"
#plt.rcParams['font.family'] = "sans-serif"  
    
    
    
    
    

# save this 3D bragg peak in orthogonal system into a vtk file for plotting in 
# can i sa
#from pyvtk import*

#x z y    # så jag har det 

#3,1,2
#test_shift_coord.data[0][test_shift_coord.data[0]<1E11] = np.inf 
# detta kanske funkar om jag plottar färra punkter 
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
## wrong order?
#ax.scatter(Qx.flatten(),Qy.flatten(),Qz.flatten(), c= test_shift_coord.data[0].flatten(), marker ='.',cmap='jet',alpha=0.5)
##ax.scatter(Qy,Qx,Qz, c= test_shift_coord.data[0], marker ='o')  
#plt.xlabel(' x [um]')
#plt.ylabel(' y [um]')
#ax.set_zlabel('z [um]')
#

#temp
def numpy2vtk(data,filename,dx=1.0,dy=1.0,dz=1.0,x0=0.0,y0=0.0,z0=0.0):
   # http://www.vtk.org/pdf/file-formats.pdf
   f=open(filename,'w')
   nx,ny,nz=data.shape
   f.write("# vtk DataFile Version 2.0\n")
   f.write("Test data\n")
   f.write("ASCII\n")
   f.write("DATASET STRUCTURED_POINTS\n")
   f.write("DIMENSIONS %u %u %u\n"%(nz,ny,nx))
   f.write("SPACING %f %f %f\n"%(dx,dy,dz))
   f.write("ORIGIN %f %f %f\n"%(x0,y0,z0))
   f.write("POINT_DATA %u\n"%len(data.flat))
   f.write("SCALARS volume_scalars float 1\n")
   f.write("LOOKUP_TABLE default\n")
   for i in data.flat:
     f.write("%f "%i)
   f.close()
   return ()
# save vtk file     
#KO = np.random.randint(1,12,size=(13,8,8))
#vtk_out = numpy2vtk(test_shift_coord.data[0],'NM2017_sim_InP_pos423.vtk',dx= qx[-1]-qx[-2], dy= qz[-2]-qz[-1], dz=qy[-1]-qy[-2], x0=0.0,y0=0.0,z0=0.0)




#%%
##-----------------------------------------------------------------------------
## visulaize probe in 3d. 
## visualize diffraction in 3d. a working slice plot
##-----------------------------------------------------------------------------
#def slice_plot():
data = np.log10(abs(loaded_probeView.data))
data[data==-np.inf]=0
mlab_xcut = views[0].shape[0]/2 # which y-z-cut to plot
mlab_zcut = views[0].shape[2]/2
mlab_ycut = views[0].shape[1]/2
#mlab.figure()
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
#                            plane_orientation='x_axes',
#                            slice_index=mlab_xcut,
#                            colormap = 'jet'
#                        )
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
#                            plane_orientation='y_axes',
#                            slice_index=mlab_ycut,
#                            colormap = 'jet'
#                        )
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
#                            plane_orientation='z_axes',
#                            slice_index=mlab_zcut,
#                            colormap = 'jet'
#                            )
#mlab.outline()
# TODO scale correctly
#frame2=357
## 3d cut-plot of the diffraction 
#data = (abs(diff2[frame2]))
#data[data==-np.inf]=0
#mlab.figure()
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
#                            plane_orientation='x_axes',
#                            slice_index=mlab_xcut,
#                            colormap = 'jet'
#                        )
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
#                            plane_orientation='y_axes',
#                            slice_index=mlab_ycut,
#                            colormap = 'jet'
#                        )
#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
#                            plane_orientation='z_axes',
#                            slice_index=mlab_zcut,
#                            colormap = 'jet'
#                            )
#mlab.outline()


#%%
#-----------------------------------------------------------------------------
# Isosurface of diffraction 3d.
#-----------------------------------------------------------------------------
#frame3 = 357
#data43 = np.copy(diff2[frame3])
#data43[np.isinf(data43)] = np.nan
#fact_ii = 1E-9
#
#mlab.figure(bgcolor=(1,1, 1),fgcolor=(0,0,0))
## more data for higher contours # no good if not log
#xmin=-(g.dq3*g.shape[0]*fact_ii)/4.0; xmax = (g.dq3*g.shape[0]*fact_ii)/4.0; ymin = -(g.dq1* g.shape[1]*fact_ii)/4.0; ymax = (g.dq1* g.shape[1]*fact_ii)/4.0; zmin=-(g.dq2* g.shape[2]*fact_ii)/2.0; zmax = (g.dq2* g.shape[2]*fact_ii)/2.0
#obj1 = mlab.contour3d( (data43), contours=3, opacity=0.9, colormap ='jet', line_width = 4, transparent=False, extent=[ xmin,xmax,ymin,ymax,zmin,zmax])
#maxes = mlab.axes(color=(0,0,0),nb_labels=5,ranges=[xmin,xmax,ymin,ymax,zmin,zmax])
##maxes.axes.fly_mode='closest_triad'
#mlab.xlabel('$q_3$ [$nm^{-1}$]'); mlab.ylabel('$q1$ [$nm^{-1}$]'); mlab.zlabel('$q2$ [$nm^{-1}$]')
#
