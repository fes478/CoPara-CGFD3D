#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot snapshot of 3D grid for FD wave simulation GPU version

Yu Zhenjiang
Chen Xiaofei Group
SUSTech
"""

import sys, time, getopt, os.path
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import matplotlib.animation as anm
import glob as gb
import multiprocessing as mp

def print_help_info():
  print('\nUsage: fig4snap2d.py [-h] | [-p projectpath]' +
    ' [-c fileconf] [-s index] [-w whichitem] [-t tsamp] [-n nprofile] [-d direction')
  print('  [projectpath ]: directory of the coordinate files.')
  print('  [parname  ]: name of the par file.')
  print('  [whichitem ]: name of the wavefield component (Vx, Vy, Vz, Txx,' +
    ' Tzz, Txy, Txz or Tyz).')
  print('  [tsamp     ]: sampling paramter (t_start/t_end/t_stride)' + 
    ' of time axis.')
  print('  [index     ]: specify the snap number.')
  print('  [direction ]: specify the showing direction(xz,yz,xy)')
  print('  [nprofile  ]: specify the profile number(absolutely location).\n')
  print('  Demos:\n ./figsnap3d.py -t 1/1500/20 -w Vz --index=3 --Crestrict=0 --parname=adjpar\n')
  print('  notice: time series must greater than 1\n')
#==============================================================================
projectpath = './'
parname = 'parfile'
index = 1
whichitem = 'Vx'
tsamp = [1, -1, 1]
LenFD = 3
nprofile = 1
Cres = 0.0
direction = 'xz'
fileconf = 'SeisFD3D.conf'
outpath = 'picture/'

#==============================================================================

try:
  opts, args = getopt.getopt(sys.argv[1:], 'hp:s:n:d:w:t:pn:cs:', ['help',
    'projectpath=', 'index=', 'nprofile=', 'direction=', 'whichitem=',
    'tsamp=', 'parname=', 'Crestrict='])
except getopt.GetoptError:
  print_help_info()
  sys.exit(2)

for opt, arg in opts:
  if opt in ('-h', '--help'):
    print_help_info()
    sys.exit()
  elif opt in ('-i', '--inputpath'):
    inputpath = arg
  elif opt in ('-o', '--outputpath'):
    outputpath = arg
  elif opt in ('-s', '--index'):
    index = int(arg)
  elif opt in ('-n', '--nprofile'):
    nprofile = int(arg)
  elif opt in ('-cs', '--Crestrict'):
    Cres = float(arg)
  elif opt in ('-d', '--direction'):
    direction = arg
  elif opt in ('-w', '--whichitem'):
    whichitem = arg
  elif opt in ('-pn', '--parname'):
    parname = arg
  elif opt in ('-t', '--tsamp'):
    tsamp = [int(v) for v in arg.split('/')]

print('\nConfigure:')
print('  projectpath:\t', projectpath)
print('  ParFileName:\t', parname)
print('  whichitem:\t', whichitem)
print('  Snap index:\t', index)
print('  Profile index: ',nprofile)
print('  Showing plane: ',direction)
print('  tsamp:\t', tsamp)
if Cres!=0 :
	print(' Apply colorbar restriction as :', Cres, '\n')

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

sub=np.zeros(12,dtype=int)
fileconf = '{}{}/station.dat'.format(projectpath,parname)
conf = open(fileconf, 'r')
lines = conf.read().split('\n')
snapname = 'snap_{:03d}'.format(index)

for line in lines:
  if line.find('#') >= 0: line = line[:line.find('#') - 1]
  if line.split(' ')[0] == snapname:
    sub[0:9] = [int(v) for v in line.split() if v.replace('-', '', 1).isdigit()][0:9]#6 par starts and counts

conf.close()

if 'sub' not in globals():
  raise IOError('{}: No such snapshot number.'.format(index))

#sub['start','count','stride','end']
#start from 1, means index is 0-199 when total = 200
#count = how many point that have already contains start and end point
#stride= step
#ending point location = 200, which index = 199
# start + (count-1)*stride = end-1
# 1+(37-1)*2= 74-1 

# X: 0 3 6 9
# Y: 1 4 7 10
# Z: 2 5 8 11

print('sub input=',sub)

# -- x --
sub[0] = max(sub[0], 1)
if sub[3] == -1:# '-1' means 'total'
  sub[3] = ni #represents counts
  sub[9] = ni #represents 'ending location (index should -1)
else:
  sub[3] = min(ni, sub[3])
  sub[9] = min(ni, sub[0]+(sub[3]-1)*sub[6])
# -- y --
sub[1] = max(sub[1], 1)
if sub[4] == -1:# '-1' means 'ends'
  sub[4] = nj
  sub[10]= nj
else:
  sub[4] = min(nj, sub[4])
  sub[10]= min(nj, sub[1]+(sub[4]-1)*sub[7])
# -- z --
sub[2] = max(sub[2], 1)
if sub[5] == -1:
  sub[5] = nk
  sub[11]= nk
else:
  sub[5] = min(nk, sub[5])
  sub[11]= min(nk, sub[2]+(sub[5]-1)*sub[8])

print('sub adjust=',sub)

#==============================================================================

if whichitem in ['Vx', 'Vy', 'Vz']:
  pname = r'$V_' + whichitem[1] + '$'
elif whichitem in ['Txx', 'Tyy', 'Tzz', 'Txy', 'Txz', 'Tyz']:
  pname = r'$\tau_{' + whichitem[1:3] + '}$'
else:
  raise IOError(whichitem + ': Invalid wavefield component.')

#comfirm display plane and profile slice
if sub[3]==1:
	if direction != 'yz':
		direction='yz'
	print('the ploting plane change to ',direction,' due to data shape')
elif sub[4]==1:
	if direction != 'xz':
		direction='xz'
	print('the ploting plane change to ',direction,' due to data shape')
elif sub[5]==1:
	if direction != 'xy':
		direction='xy'
	print('the ploting plane change to ',direction,' due to data shape')

if direction=='xz':
	if nprofile<sub[1]:
		nprofile=sub[1]
	if nprofile> int(sub[1]+(sub[4]-1)*sub[7]):
		nprofile=int(sub[1]+(sub[4]-1)*sub[7])
	nprofile=int((nprofile-sub[1])/sub[7])
	print('the Profile was Modified as',nprofile)
elif direction=='yz':
	if nprofile<sub[0]:
		nprofile=sub[0]
	if nprofile>int(sub[0]+(sub[3]-1)*sub[6]):
		nprofile=int(sub[0]+(sub[3]-1)*sub[6])
	nprofile=int((nprofile-sub[0])/sub[6])
	print('the Profile was Modified as',nprofile)
elif direction=='xy':
	if nprofile<sub[2]:
		nprofile=sub[2]
	if nprofile>int(sub[2]+(sub[5]-1)*sub[8]):
		nprofile=int(sub[2]+(sub[5]-1)*sub[8])
	nprofile=int((nprofile-sub[2])/sub[8])
	print('the Profile was Modified as',nprofile)


#reading time and data variables
filename = '{}{}/output/{}.nc'.format(projectpath, parname, snapname)
#filename = '{}{}/output3k25/{}.nc'.format(projectpath, parname, snapname)
snp = nc.Dataset(filename, 'r')
t=snp.variables['currTime'][:]

if tsamp[1] == -1: 
	tsamp[1] = t.size # t read from snp
tnumber= int( 1 + (tsamp[1]-tsamp[0])/tsamp[2] )
tseries=t[tsamp[0]-1:tsamp[1]:tsamp[2]]

dvar=np.zeros((tnumber,sub[3],sub[4],sub[5]))
zvar=np.zeros((sub[3],sub[4],sub[5]))
yvar=np.zeros((sub[3],sub[4],sub[5]))
xvar=np.zeros((sub[3],sub[4],sub[5]))

dvar[:,:,:,:]=snp.variables[whichitem][tsamp[0]-1:tsamp[1]:tsamp[2],:,:,:]
			
snp.close()

filename = '{}{}/input/coord.nc'.format(projectpath,parname)
crd = nc.Dataset(filename, 'r')
xvar[:,:,:]=crd.variables['x'][sub[0]-1:sub[9]:sub[6],sub[1]-1:sub[10]:sub[7],sub[2]-1:sub[11]:sub[8]]
yvar[:,:,:]=crd.variables['y'][sub[0]-1:sub[9]:sub[6],sub[1]-1:sub[10]:sub[7],sub[2]-1:sub[11]:sub[8]]
zvar[:,:,:]=crd.variables['z'][sub[0]-1:sub[9]:sub[6],sub[1]-1:sub[10]:sub[7],sub[2]-1:sub[11]:sub[8]]
crd.close()

xplt=xvar/1.0e3
yplt=yvar/1.0e3
zplt=zvar/1.0e3

#confirm the ploting axes
if direction=='yz':
	print('ploting Y-Z imaging')
	coord=['y','z']
	#FLdim=[dims[1],dims[2],1]
	last=np.squeeze(zplt[nprofile,:,:])
	fast=np.squeeze(yplt[nprofile,:,:])
	dvar=np.squeeze(dvar[:,nprofile,:,:])
elif direction=='xz':
	print('ploting X-Z imaging')
	coord=['x','z']
	#FLdim=[dims[0],dims[2],2]
	last=np.squeeze(zplt[:,nprofile,:])
	fast=np.squeeze(xplt[:,nprofile,:])
	dvar=np.squeeze(dvar[:,:,nprofile,:])
elif direction=='xy':
	print('ploting X-Y imaging')
	coord=['x','y']
	#FLdim=[dims[0],dims[1],3]
	last=np.squeeze(yplt[:,:,nprofile])
	fast=np.squeeze(xplt[:,:,nprofile])
	dvar=np.squeeze(dvar[:,:,:,nprofile])


#==============================================================================


if 1:
  # --- Matlab-like dynamically plot ---
  fig = plt.figure(figsize = (7, 6), dpi = 80)
  for it in range(tnumber):
    plt.clf()
    if tnumber==1 :
      plt.pcolormesh(fast, last, dvar[:, :], cmap = 'seismic', rasterized = True)
    else:  
      plt.pcolormesh(fast, last, dvar[it, :, :], cmap = 'seismic', rasterized = True)
    print('time=\n',tseries[it])
    plt.gca().set_aspect('equal')
    plt.xlabel(coord[0]+' distance (km)')
    plt.ylabel(coord[1]+' distance (km)')
    plt.title('{} snapshot of No.{:d} when t = {:.2f} s'.format(pname, index,tseries[it]))
    plt.colorbar()
    if Cres!=0:
	    plt.clim([-1.0*Cres,Cres])
    plt.draw()
    plt.pause(0.0001)
  # time.sleep(.2)
else:
  # --- Matplotlib animation plot ---
  imgs = []
  fig = plt.figure(figsize = (7, 6), dpi = 80)
  for it in range(tnumber):
    img = plt.pcolormesh(fast, last, dvar[it, :, :], cmap = 'jet', rasterized = True,
      animated = True)
    print('timeIMG=\n',tseries[it])
    plt.gca().set_aspect('equal')
    plt.xlabel(coord[0]+' distance (km)')
    plt.ylabel(coord[1]+' distance (km)')
    plt.title('{} snapshot of No.{:d} when t = {:.2f} s'.format(pname, index,
      tseries[it]))
    #plt.colorbar()
    imgs.append([img])
  # plt.clf()
  ani = anm.ArtistAnimation(fig, imgs, interval = 50, blit = True)#, repeat_delay = 0)

plt.show()


'''



#plot source and receiver
def replot(plt):
	plt.plot(recx,recy,'ko',marksize=3,fillstyle='none')
	plt.plot(srcx,srcy,'w*',marksize=3,fillstyle='none')

#==============================================================================
ts = 0
tt = 1
te = dvar.shape[0]
clims = (-1000,1000)
sc = 1e0

def mpplot(it):
	plt.pcolormesh(fast,last,dvar[it,:,:], cmap='seismic',rasterized=True)
	#rsplot(plt)
	plt.gca().set_aspect('equal')
	plt.xlabel(coord[0]+' distance (km)')
	plt.ylabel(coord[1]+' distance (km)')
	plt.title('{} snapshot of No.{:d} at t = {:.2f} s'.format(pname, index,tseries[it]))
	plt.colorbar()
	if Cres!=0:
		plt.clim([-1.0*Cres,Cres])
	if 0:
		if 0:
			plt.clim(clims)
		else:
			climit=np.max(np.abs(dvar[it,:,:]))/sc
			plt.clim([-climit,climit])
	plt.savefig(outpath+'snap{:03d}.png'.format(it),dpi=300,format='png')
	plt.clf()

#=============================================================================
#parallel plot

if 1:
	for ip in gb.glob(outpath+'*.png'):
		os.remove(ip)
	print('remove all existing pictures and start to save new')

	pool = mp.Pool(20)
	re = pool.map(mpplot, range(ts,te,tt))
	print('picture {}snap{:03d}.png was generated\n'.format(outpath,ts))

'''





