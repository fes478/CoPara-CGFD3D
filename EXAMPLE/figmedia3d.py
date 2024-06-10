#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot figure of 3D media for FD wave simulation GPU version

Yu Zhenjiang
Chen Xiaofei Group
SUSTech
"""

#import matplotlib
#matplotlib.use('Agg')
import sys, getopt, os.path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
import numpy as np
import netCDF4 as nc

def print_help_info():
  print('\nUsage: fig4media2d.py [-h] | [-m mapdim] [-d direction] [-p projectpath] [-c fileconf]' +
    ' [-n nprofile] [-w whichitem] [-s scalefactor] [-g grid] [-x xystep] [-z zstep]')
  print('  [mapdim   ]: plot in 2D or 3D.')
  print('  [direction]: extract slice from this direction and display in the Other-Z plane.')
  print('  [projectpath]: directory of the project files.')
  print('  [fileconf ]: name of the configure file.')
  print('  [parname  ]: name of the par file.')
  print('  [nprofile ]: number of the specified profile, 2D for single number 3D for mutil number.')
  print('  [whichitem]: name of the elastic parameter (rho, vp, vs,lambda or mu) to plot.')
  print('  [grid     ]: whether to append the meshgrid.')
  print('  [scalefactor]: coordinate space ratio when plot. (3D only)')
  print('  [xystep   ]: step length to skip on x or y-direction. (2D only)')
  print('  [zstep    ]: step length to skip on z-direction.  (2D only)\n')

#==============================================================================

mapdim = 2
direction = 'x'
projectpath = './'
fileconf = 'SeisFD3D.conf'
parname = 'parfile'
nprofile = 1
whichitem = 'rho'
LenFD = 3
grid = 0
scalefactor = 1
xystep = 1
zstep = 1

#==============================================================================

try:
  opts, args = getopt.getopt(sys.argv[1:], 'hm:d:p:c:n:w:s:g:x:z:pn:', ['help', 'mapdim=', 'direction=', 'projectpath=',
    'fileconf=', 'nprofile=', 'whichitem=', 'scalefactor=', 'grid=', 'xystep=', 'zstep=', 'parname='])
except getopt.GetoptError:
  print_help_info()
  sys.exit(2)

for opt, arg in opts:
  if opt in ('-h', '--help'):
    print_help_info()
    sys.exit()
  elif opt in ('-m', '--mapdim'):
    mapdim = int(arg)
  elif opt in ('-d', '--direction'):
    direction = arg
  elif opt in ('-p', '--projectpath'):
    projectpath = arg
  elif opt in ('-c', '--fileconf'):
    fileconf = arg
  elif opt in ('-pn', '--parname'):
    parname = arg
  elif opt in ('-n', '--nprofile'):
    nprofile = arg
  elif opt in ('-w', '--whichitem'):
    whichitem = arg
  elif opt in ('-s', '--scalefactor'):
    scalefactor = int(arg)
  elif opt in ('-g', '--grid'):
    grid = int(arg)
  elif opt in ('-x', '--xystep'):
    xystep = int(arg)
  else:
    zstep = int(arg)
  #elif opt in ('-z', '--zstep'):
  #  zstep = int(arg)

#==============================================================================
fileconf = '{}{}/SeisFD3D.conf'.format(projectpath,parname)
conf = open(fileconf, 'r')
lines = conf.read().split('\n')

for line in lines:
  if line.find('#') >= 0: 
	  line = line[:line.find('#') - 1]
  if line.split(' ')[0] == 'ni':
    ni = int(line.split()[2])
  elif line.split(' ')[0] == 'nj':
    nj = int(line.split()[2])
  elif line.split(' ')[0] == 'nk':
    nk = int(line.split()[2])

conf.close()

#==============================================================================
np_list=nprofile.split(',',-1)
lengthprofile=len(np_list)
if lengthprofile==1:
	nprof=int(nprofile) #string to int
	if direction=='x':
		if nprof>ni or nprof<1 :
			print('exceeds the slices boundary on X direction')
	else:
		if nprof>nj or nprof<1 :
			print('exceeds the slices boundary on Y direction')
else:
	np_list_int=list(map(int,np_list))
	if mapdim==2:
		lengthprofile==1
		nprof=np_list_int[0]
	else:
		nprof=np_list_int
	if direction=='x':
		if nprof>ni or nprof<1 :
			print('exceeds the slices boundary on X direction')
	else:
		if nprof>nj or nprof<1 :
			print('exceeds the slices boundary on Y direction')

print('\nConfigure:')
print('  mapdim:\t', mapdim)
print('  ExtractDirection:\t', direction)
print('  projectpath:\t', projectpath)
print('  fileconf:\t', fileconf)
print('  ParFileName:\t',parname)
print('  nprofile:\t', nprof)
print('  whichitem(vp,vs,rho,mu,lambda)-->:\t', whichitem)
print('  with grid:\t', grid)
if mapdim==3:
  	print('  scalefactor:\t', scalefactor,'\n')
else:
	print('  xystep:\t', xystep)
	print('  zstep:\t', zstep, '\n')

print('  Now only valid for 2D case (single slice)')
print('  that means, nprofile should be single and')
print('  scalefactor is also useless!!!')
print('  Demos as: ')
print('  ./figmedia3d.py -m 2 --direction=y --xystep=5 --whichitem=lambda --grid=1 --zstep=20 --nprofile="100" --parname=hill3dpar')


if direction=='x':
	displaydir = 'y'
	x = np.zeros((lengthprofile,nj,nk))
	y = np.zeros((lengthprofile,nj,nk))
	z = np.zeros((lengthprofile,nj,nk))
	vp = np.zeros((lengthprofile,nj,nk))
	vs = np.zeros((lengthprofile,nj,nk))
	rho = np.zeros((lengthprofile,nj,nk))
	pvar = np.zeros((lengthprofile,nj,nk))
	
	for ith in range(lengthprofile):
		if lengthprofile != 1:
			js=nprof[ith]-1
		else:
			js=nprof-1
		filename = '{}{}/input/media.nc'.format(projectpath,parname)
		if not os.path.isfile(filename):
			raise IOError(filename + ': No such file.')
		med = nc.Dataset(filename, 'r')
		vp[ith,:,:]=med.variables['vp'][js,:,:]
		vs[ith,:,:]=med.variables['vs'][js,:,:]
		rho[ith,:,:]=med.variables['rho'][js,:,:]
		med.close()

		filename = '{}{}/input/coord.nc'.format(projectpath,parname)
		if not os.path.isfile(filename):
			raise IOError(filename + ': No such file.')
		crd = nc.Dataset(filename, 'r')
		x[ith,:,:]=crd.variables['x'][js,:,:]
		y[ith,:,:]=crd.variables['y'][js,:,:]
		z[ith,:,:]=crd.variables['z'][js,:,:]
		crd.close()
else:
	displaydir = 'x'
	x = np.zeros((ni, lengthprofile, nk))
	y = np.zeros((ni, lengthprofile, nk))
	z = np.zeros((ni, lengthprofile, nk))
	vp = np.zeros((ni, lengthprofile, nk))
	vs = np.zeros((ni, lengthprofile, nk))
	rho = np.zeros((ni, lengthprofile, nk))
	pvar = np.zeros((ni, lengthprofile, nk))
 
	for jth in range(lengthprofile):
		if lengthprofile != 1:
			js=nprof[jth]-1
		else:
			js=nprof-1
		filename = '{}{}/input/media.nc'.format(projectpath,parname)
		if not os.path.isfile(filename):
			raise IOError(filename + ': No such file.')
		med = nc.Dataset(filename, 'r')
		vp[:,jth,:]=med.variables['vp'][:,js,:]
		vs[:,jth,:]=med.variables['vs'][:,js,:]
		rho[:,jth,:]=med.variables['rho'][:,js,:]
		med.close()

		filename = '{}{}/input/coord.nc'.format(projectpath,parname)
		if not os.path.isfile(filename):
			raise IOError(filename + ': No such file.')
		crd = nc.Dataset(filename, 'r')
		x[:,jth,:]=crd.variables['x'][:,js,:]
		y[:,jth,:]=crd.variables['y'][:,js,:]
		z[:,jth,:]=crd.variables['z'][:,js,:]
		crd.close()

x=x/1.0e3
y=y/1.0e3
z=z/1.0e3

#==============================================================================
tiny=1e-12
rho=rho+tiny

if whichitem == 'vp':
  pvar=vp/1.0e3
  pname = {'title':'Vp', 'unit':'km/s'}
elif whichitem == 'vs':
  pvar=vs/1.0e3
  pname = {'title':'Vs', 'unit':'km/s'}
elif whichitem == 'rho':
  pvar=rho/1.0e3
  pname = {'title':r'$\rho$', 'unit':'$kg/m^3$'}
elif whichitem == 'mu':
  pvar=vs*vs*rho/1.0e9
  pname = {'title':r'$\mu$', 'unit':'GPa'}
elif whichitem == 'lambda':
  pvar=(rho*vp*vp-2*vs*vs*rho)/1.0e9
  pname = {'title':r'$\lambda$', 'unit':'GPa'}
else:
  raise IOError(whichitem + ': No such elastic parameter.')

fig = plt.figure(figsize = (7, 6), dpi = 80)

#ax = fig.gca(projection='3d')

if mapdim==2:
	if direction=='x':
		xy=np.squeeze(y)
		ijend=nj-1
	else:
		xy=np.squeeze(x)
		ijend=ni-1
	z=np.squeeze(z)
	pvar=np.squeeze(pvar)
	
	plt.pcolormesh(xy,z,pvar,cmap='jet',rasterized=True)
	plt.gca().set_aspect('equal')
	plt.xlabel(displaydir+' distance (km)')
	plt.ylabel('z depth (km)')
	plt.title('Media map for {} ({})'.format(pname['title'], pname['unit']))
	plt.colorbar(fraction=0.05,shrink=0.5)
	if grid:
		for k in range(nk-1,0,-zstep):
			plt.plot(xy[:,k],z[:,k],'k')
		plt.plot(xy[:,0],z[:,0],'k')
		for i in range(0,ijend,xystep):
			plt.plot(xy[i,:],z[i,:],'k')
		plt.plot(xy[ijend,:],z[ijend,:],'k')
		'''
		for i in range(0,ijend,xystep):
			plt.plot(xy[:,i],z[:,i],'k')
		plt.plot(xy[:,ijend],z[:,ijend],'k')
		for k in range(nk-1,0,-zstep):
			plt.plot(xy[k,:],z[k,:],'k')
		plt.plot(xy[0,:],z[0,:],'k')
		'''
else:
	if direction=='x':
		for ith in range(lengthprofile):
			y=np.squeeze(y[ith,:,:])
			z=np.squeeze(z[ith,:,:])
			pvar=np.squeeze(pvar[ith,:,:])
			ijend=nj-1
			plt.pcolormesh(y,z,pvar,cmap='jet',rasterized=True)
		if grid:
			for i in range(0,ijend,xystep):
				plt.plot(y[i,:],z[i,:],'k')
			plt.plot(y[ijend,:],z[ijend,:],'k')
			for k in range(nk-1,0,-zstep):
				plt.plot(y[:,k],z[:,k],'k')
			plt.plot(y[:,0],z[:,0],'k')
	else:
		for jth in range(lengthprofile):
			x=np.squeeze(x[:,jth,:])
			z=np.squeeze(z[:,jth,:])
			pvar=np.squeeze(pvar[:,jth,:])
			ijend=ni-1
			plt.pcolormesh(x,z,pvar,cmap='jet',rasterized=True)
		if grid:
			for j in range(0,ijend,xystep):
				plt.plot(x[j,:],z[j,:],'k')
			plt.plot(x[ijend,:],z[ijend,:],'k')
			for k in range(nk-1,0,-zstep):
				plt.plot(x[:,k],z[:,k],'k')
			plt.plot(x[:,0],z[:,0],'k')
			
	plt.gca().set_aspect('equal')
	plt.xlabel(displaydir+' distance (km)')
	plt.ylabel('z depth (km)')
	plt.title('Media map for {} ({})'.format(pname['title'], pname['unit']))
	plt.colorbar()
plt.show()
# plt.savefig('test.eps')
'''	
#else for the 3D surface case, but failed becaused of ax.plot_surface (shape problem) and set_aspect(scalefactor problem)
else:
	if direction=='x':
		print('before:Zshape=',np.shape(z),'Xshape=',np.shape(x),'Yshape=',np.shape(y),'pvarshape=',np.shape(pvar))
		surf=ax.plot_surface(np.squeeze(x[:,0,:]),np.squeeze(y[:,:,0]),np.squeeze(z[0,:,:]),np.squeeze(pvar[0,:,:]),cmap='jet');
		print('Zsize=',np.size(z),'Xsize=',np.size(x),'Ysize=',np.size(y),'pvarsize=',np.size(pvar))
		print('after:Zshape=',np.shape(z),'Xshape=',np.shape(x),'Yshape=',np.shape(y),'pvarshape=',np.shape(pvar))
		for jth in range(1,lengthprofile,1):
			surf=ax.plot_surface(np.squeeze(x[:,jth,:]),np.squeeze(y[:,:,jth,]),np.squeeze(z[jth,:,:]),np.squeeze(pvar[jth,:,:]),cmap='jet');
		if grid:
			for jth in range(lengthprofile):
				surf=ax.plot3D(np.squeeze(x[:,jth,0]),np.squeeze(y[:,jth,0]),\
				np.squeeze(z[:,jth,0]),'k')
				surf=ax.plot3D(np.squeeze(x[:,jth,ni-1]),np.squeeze(y[:,jth,ni-1]),\
				np.squeeze(z[:,jth,ni-1]),'k')
				surf=ax.plot3D(np.squeeze(x[0,jth,:]),np.squeeze(y[0,jth,:]),\
				np.squeeze(z[0,jth,:]),'k')
				surf=ax.plot3D(np.squeeze(x[nk-1,jth,:]),np.squeeze(y[nk-1,jth,:]),\
				np.squeeze(z[nk-1,jth,:]),'k')
		#plt.gca().set_aspect([scalefactor,1,scalefactor])
	else:
		surf=ax.plot_surface(np.squeeze(x[0,:,:]),np.squeeze(y[0,:,:]),\
		np.squeeze(z[0,:,:]),np.squeeze(pvar[0,:,:]),cmap='jet');
		for ith in range(1,lengthprofile,1):
			surf=ax.plot_surface(np.squeeze(x[ith,:,:]),np.squeeze(y[ith,:,:]),\
			np.squeeze(z[ith,:,:]),np.squeeze(pvar[ith,:,:]),cmap='jet');
		if grid:
			for ith in range(lengthprofile):
				surf=ax.plot3D(np.squeeze(x[:,0,ith]),np.squeeze(y[:,0,ith]),\
				np.squeeze(z[:,0,ith]),'k')
				surf=ax.plot3D(np.squeeze(x[:,nj-1,ith]),np.squeeze(y[:,nj-1,ith]),\
				np.squeeze(z[:,nj-1,ith]),'k')
				surf=ax.plot3D(np.squeeze(x[0,:,ith]),np.squeeze(y[0,:,ith]),\
				np.squeeze(z[0,:,ith]),'k')
				surf=ax.plot3D(np.squeeze(x[nk-1,:,ith]),np.squeeze(y[nk-1,:,ith]),\
				np.squeeze(z[nk-1,:,ith]),'k')
		#plt.gca().set_aspect([scalefactor,scalefactor,1])
	ax.set_xlabel(direction+' distance (km)')
	ax.set_zlabel('z depth (km)')
	ax.set_title('Media map for {} ({})'.format(pname['title'], pname['unit']))
	#fig.colorbar(surf)
plt.show()

'''
