#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot figure of 3D grid for FD wave simulation GPU version

Yu Zhenjiang 
Chen Xiaofei Group
SUSTech
"""

import sys, getopt, os.path
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc

def print_help_info():
  print('\nUsage: figgrid3d.py [-h] | [-p projectpath] [-c fileconf]' +
    ' [-d direction] [-n nprofile] [-s xystep] [-z zstep]')
  print('  [projectpath]: directory of the project files.')
  print('  [fileconf ]: name of the configure file.')
  print('  [parname  ]: name of the par file.')
  print('  [depth    ]: depth of convers layer.')
  print('  [direction]: extracting from the specified direction(x or y).')
  print('  [nprofile ]: number of profiles to be ploted.')
  print('  [xystep   ]: step length to skip on x or y-direction.')
  print('  [zstep    ]: step length to skip on z-direction.\n')
  print('  [absLayer ]: absorption layer location.\n')
  print('  Demos:\n ./figgrid3d.py --nprofile=50  --xystep=5 --zstep=5 --direction=x --depth=-2.0E3 --absL=1 --SRpos=1 --parname=adjpar6\n')

#==============================================================================

projectpath = './'
fileconf = 'SeisFD3D.conf'
parname = 'parfile'
direction = 'x'
nprofile = 1
xystep = 1
zstep = 1
depth = 0
absL = 0
SRp = 0

#==============================================================================

try:
  opts, args = getopt.getopt(sys.argv[1:], 'hp:c:d:n:s:z:pn:cd:al:sr:', ['help', 'projectpath=',
    'fileconf=', 'direction=', 'nprofile=',  'xystep=', 'zstep=', 'parname=', 'depth=', 
    'absL=', 'SRpos='])
except getopt.GetoptError:
  print_help_info()
  sys.exit(2)

for opt, arg in opts:
  if opt in ('-h', '--help'):
    print_help_info()
    sys.exit()
  elif opt in ('-p', '--projectpath'):
    inputpath = arg
  elif opt in ('-c', '--fileconf'):
    fileconf = arg
  elif opt in ('-d', '--direction'):
    direction = arg
  elif opt in ('-n', '--nprofile'):
    nprofile = int(arg)
  elif opt in ('-s', '--xystep'):
    xystep = int(arg)
  elif opt in ('-z', '--zstep'):
    zstep = int(arg)
  elif opt in ('-cd', '--depth'):
    depth = float(arg)
  elif opt in ('-pn', '--parname'):
    parname = arg
  elif opt in ('-al', '--absL'):
    absL = int(arg)
  elif opt in ('-sr', '--SRpos'):
    SRp = int(arg)

print('\nConfigure:')
print('  project_path:\t', projectpath)
print('  fileconf:\t', fileconf)
print('  ParFileName:\t', parname)

#==============================================================================

# read ni nj nk
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


# read abs number
absconf = '{}{}/absorb.dat'.format(projectpath,parname)
absnum = np.zeros(6,dtype=int)
conf = open(absconf, 'r')
lines = conf.read().split('\n')
for line in lines:
  if line.find('#') >= 0: 
	  line = line[:line.find('#') - 1]
  if line.split(' ')[0] == 'abs_number':
    absnum[0:6] = [int(v) for v in line.split() if v.replace('-', '', 1).isdigit()][0:6]
conf.close()

# read station information
recvconf = '{}{}/station.dat'.format(projectpath,parname)
conf = open(recvconf, 'r')
lines = conf.read().split('\n')
for line in lines:
  if line.find('#') >= 0: 
	  line = line[:line.find('#') - 1]
  if line.split(' ')[0] == 'number_of_recv':
	  nrecv = int(line.split()[2])
conf.close()

if SRp:
  Rloc = np.zeros((nrecv,3))
  seismoconf = '{}{}/output/seismo.nc'.format(projectpath,parname)
  if not os.path.isfile(seismoconf):
      raise IOError(seismoconf + ': No such file.')
  ssm = nc.Dataset(seismoconf, 'r')
  
  Rloc[:,0] = ssm.variables['posx'][0:nrecv]
  Rloc[:,1] = ssm.variables['posy'][0:nrecv]
  Rloc[:,2] = ssm.variables['posz'][0:nrecv]
  Rloc = Rloc/1.0e3
  
  ssm.close()
  #print(Rloc)

  #read source information
  srcconf = '{}{}/source.dat'.format(projectpath,parname)
  conf = open(srcconf, 'r')
  Slines = conf.readlines()
  conf.close()
  for LLL in Slines:
  	if LLL.split(' ')[0] == 'number_of_force_source':
  		nfrc = int(LLL.split()[2])
  	if LLL.find('<anchor_force>')>=0:
  		FrcLoc = Slines.index(LLL) + 1
  	if LLL.split(' ')[0] == 'number_of_moment_source':
  		nmnt = int(LLL.split()[2])
  	if LLL.find('<anchor_moment>')>=0:
  		MntLoc = Slines.index(LLL) + 1
  SFloc = np.zeros((nfrc,3))
  SMloc = np.zeros((nmnt,3))
  for fidx in range(nfrc):
  	LLL = Slines[FrcLoc+fidx]
  	SFloc[fidx,0]=float(LLL.split()[0])
  	SFloc[fidx,1]=float(LLL.split()[1])
  	SFloc[fidx,2]=float(LLL.split()[2])
  
  for fidx in range(nmnt):
  	LLL = Slines[MntLoc+fidx]
  	SMloc[fidx,0]=float(LLL.split()[0])
  	SMloc[fidx,1]=float(LLL.split()[1])
  	SMloc[fidx,2]=float(LLL.split()[2])
  #print(SFloc)
  #print(SMloc)



#==============================================================================
if direction == 'x':
  print('  profile_extract_diretion:\t', direction)
  print('  profile_number:\t', nprofile,'\t(totally have ',nj,' profiles)')
  print('  xystep:\t', xystep)
  print('  zstep:\t', zstep, '\n')
  xyvar = np.zeros((nj, nk))
  zvar = np.zeros((nj, nk))
  ijend = nj - 1
  displaydir = 'y'
  filename = '{}{}/input/coord.nc'.format(projectpath,parname)
  if not os.path.isfile(filename):
    raise IOError(filename + ': No such file.')
  crd = nc.Dataset(filename, 'r')
  #[fast,midium,low] same as C write, contrary with Fortran write and FORpy read
  xyvar = crd.variables['y'][nprofile-1 , :, :]
  zvar = crd.variables['z'][nprofile-1 , :, :]
  crd.close()
  nabsS = absnum[2]
  nabsE = absnum[3]
  if SRp:
    SRFxy = np.zeros((nfrc))
    SRMxy = np.zeros((nmnt))
    Rlocxy = np.zeros((nrecv))
    SRFxy = SFloc[:,1]
    SRMxy = SMloc[:,1]
    Rlocxy = Rloc[:,1]

else:
  print('  profile_extract_diretion:\t', direction)
  print('  profile_number:\t', nprofile,'\t(totally have ',ni,' profiles)')
  print('  xystep:\t', xystep)
  print('  zstep:\t', zstep, '\n')
  zvar = np.zeros((ni, nk))
  xyvar = np.zeros((ni, nk))
  ijend = ni -1
  displaydir = 'x'
  filename = '{}{}/input/coord.nc'.format(projectpath,parname)
  if not os.path.isfile(filename):
    raise IOError(filename + ': No such file.')
  crd = nc.Dataset(filename, 'r')
  xyvar = crd.variables['x'][:, nprofile-1, :]
  zvar = crd.variables['z'][:, nprofile-1, :]
  crd.close()
  nabsS = absnum[0]
  nabsE = absnum[1]
  if SRp:
    SRFxy = np.zeros((nfrc))
    SRMxy = np.zeros((nmnt))
    Rlocxy = np.zeros((nrecv))
    SRFxy = SFloc[:,0]
    SRMxy = SMloc[:,0]
    Rlocxy = Rloc[:,0]
xyvar = xyvar/1.0e3
zvar = zvar/1.0e3
depth = depth/1.0e3


#Zdep
ALdown = zvar[0,absnum[4]-1] 
ALup = zvar[0,nk-absnum[5]-1]
ALleft = xyvar[nabsS-1,0]
if direction == 'x':
	ALright = xyvar[nj-nabsE,0]
else:
	ALright = xyvar[ni-nabsE,0]


#print(ALup,ALdown,ALleft,ALright)
if absL:
	print('  will append absorption layer postion')
if SRp:
	print('  will append position of source(force {} and moment {}) and recvier {}'.format(nfrc,nmnt,nrecv))

#==============================================================================
kend = nk - 1 
fig = plt.figure(figsize = (7, 7), dpi = 80)
for i in range(0, ijend, xystep):
  plt.plot(xyvar[i,:], zvar[i,:], 'k')
plt.plot(xyvar[ijend,:], zvar[ijend,:], 'k')
for k in range(kend, 0, - zstep):
  plt.plot(xyvar[:,k], zvar[:,k], 'k')
plt.plot(xyvar[:,0], zvar[:,0], 'k')
# plt.axes([xyvar.min(), xyvar.max(), zvar.min(), zvar.max()])
plt.gca().set_aspect('equal')
plt.xlabel(displaydir+' distance (km)' )
plt.ylabel('z depth (km)')
plt.title('Physical meshgrid profile No.'+str(nprofile))
if depth:
  plt.plot([xyvar[0,0],xyvar[ijend,0]],[depth,depth],'-b')#convert
if absL:
  plt.plot([xyvar[0,0],xyvar[ijend,0]],[ALup,ALup],'--m')#abs
  plt.plot([xyvar[0,0],xyvar[ijend,0]],[ALdown,ALdown],'--m')#abs
  plt.plot([ALleft,ALleft],[zvar[0,0],zvar[0,kend]],'--m')#abs
  plt.plot([ALright,ALright],[zvar[0,0],zvar[0,kend]],'--m')#abs
if SRp:
  for fidx in range(nfrc):
	  plt.plot(SRFxy[fidx],SFloc[fidx,2],'r*',markersize=5,fillstyle='none')
  for fidx in range(nmnt):
	  plt.plot(SRMxy[fidx],SMloc[fidx,2],'r*',markersize=5,fillstyle='none')
  for fidx in range(nrecv):
	  #plt.plot(Rlocxy[fidx],Rloc[fidx,2],'gX',markersize=5,fillstyle='none')
	  if fidx==2:#use absolute index
		  plt.plot(Rlocxy[fidx-1],Rloc[fidx-1,2],'gX',markersize=5,fillstyle='none')
plt.show()
# plt.saveas('test.eps')
