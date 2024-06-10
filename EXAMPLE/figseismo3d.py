#!/usr/bin/env python3
# -*- conding: utf-8 -*-
"""
Plot figure of 3D seismogram for FD wave simulation GPU verison.

Yu Zhenjiang
Chen Xiaofei Group
SUSTech
"""

import sys, getopt, os.path
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc

def print_help_info():
  print('\nUsage: figseismo3d.py [-h] | [-d display] [-i index]' +
    ' [-p projectpath] [-c fileconf] [-s scalefactor]')
  print('  [projectpath  ]: directory of the project files.')
  print('  [display      ]: choose to display line or point seismogram.')
  print('  [index        ]: seis-line/point number(start from 0) to be plotted.')
  print('  [fileconf     ]: name of the configure file.')
  print('  [parname  ]: name of the par file.')
  print('  [restriction   ]: restrict the amplitude to same level.')
  print('  [scalefactor  ]: scale factor using to adjust the trace interval.\n')
  print('\nDemos:\n ./figseismo3d.py --display=point --restriction=1 --index=14 --parname=adjpar\n')

#==============================================================================

projectpath = './'
fileconf = 'SeisFD3D.conf'
parname = 'parfile'
isline = False
index = 1
scalefactor = 1.0
LenFD = 3
restriction = 0

#==============================================================================

try:
	opts, args = getopt.getopt(sys.argv[1:], 'hd:i:p:c:s:pn:r:', ['help',
    'display=', 'index=', 'projectpath=', 'flieconf=', 'scalefactor=', 
    'parname=', 'restriction='])
except getopt.GetoptError:
  print_help_info()
  sys.exit(2)

for opt, arg in opts:
  if opt in ('-h', '--help'):
    print_help_info()
    sys.exit()
  elif opt in ('-d', '--display'):
    display = arg
  elif opt in ('-i', '--index'):
    index = int(arg)
  elif opt in ('-p', '--projectpath'):
    inputpath = arg
  elif opt in ('-pn', '--parname'):
    parname = arg
  elif opt in ('-c', '--fileconf'):
    fileconf = arg
  elif opt in ('-s', '--scalefactor'):
    scalefactor = float(arg)
  elif opt in ('-r', '--restriction'):
    restriction = int(arg)

if(display == 'line'):
	isline = True

#==============================================================================

#==============================================================================

whitems = ['Vx', 'Vy', 'Vz', 'Txx', 'Tyy', 'Tzz', 'Txy', 'Txz', 'Tyz']
      
filename = '{}{}/output/seismo.nc'.format(projectpath,parname)
#filename = '{}{}/output002/seismo.nc'.format(projectpath,parname)
if not os.path.isfile(filename):
	raise IOError(filename+': No such file.')
ssm = nc.Dataset(filename, 'r')

t = ssm.variables['currTime'][:]
nt = t.size
Lnum = ssm.variables['Lnum'][:]
Pnum = ssm.variables['Pnum'][:]
npnt = Pnum.size

#confirm how many line
Lflag = Lnum[0]
count = 1
for i in range(npnt):
	if(Lflag != Lnum[i]):
		Lflag = Lnum[i]
		count = count + 1
Lcount = np.zeros(count,dtype=np.int)

j = 0
Lflag = Lnum[0]
for i in range(npnt):
	if(Lflag == Lnum[i]):
		Lcount[j] = Lcount[j] + 1
	else:
		j = j + 1
		Lcount[j] = Lcount[j] + 1
		Lflag = Lnum[i]

print('\nConfigure:')
if isline:
  print('  line:\t', index)
else:
  print('  point:\t', index)
print('  projectpath:\t', projectpath)
print('  ParFileName:\t', parname)
print('  fileconf:\t', fileconf)
print('  file:\t', filename)
print('  scalefactor:\t', scalefactor, '\n')
print('Every line counts as',Lcount)
if restriction:
  print('Apply restriction to the amplitude!\n') 

#index start from 0( same as nc file)
#but for point start from 1, then modified to 0
start=0
end=0

if isline:
	Lpnt=Lcount[index]
	for i in range(0,index):
		start = start + Lcount[i]
	end = start + Lcount[index]

else:
	Lpnt=1

dvar = np.zeros((nt,9,Lpnt))

if isline:
	for i in range(9):
		dvar[:,i,:] = ssm.variables[whitems[i]][:,start:end] 
else:
	for i in range(9):
		dvar[:,i,0] = ssm.variables[whitems[i]][:,index-1]#actrually start from 0

ssm.close()

#print(np.shape(dvar))
#print(dvar[:,0,0]) #start from 0 both for item and Lnumber
#for i in range(nt):
#	print('i=',i,'\tvalue=',dvar[i,0,0])

pvar = np.zeros((nt,9,Lpnt))
for iw in range(9):
	if iw < 3:
		pvar[:, iw, :] = dvar[:, iw, :]
	else:
		pvar[:, iw, :] = dvar[:, iw, :]/1.0e6

#for i in range(nt):
#	print('i=',i,'\tvalue=',pvar[i,0,0])

#plt.plot(t,pvar[:,0,0])
#plt.show()

#==============================================================================

pname = ['$V_x$', '$V_y$', '$V_z$', r'$\tau_{xx}$', r'$\tau_{yy}$', r'$\tau_{zz}$', 
  r'$\tau_{xy}$', r'$\tau_{xz}$', r'$\tau_{yz}$']
xbgl = [t[0], t[- 1]]
ybgl = np.array([- 0.5, - 0.5])

if isline:
  for iw in range(9):
    traceshift = abs(pvar[:, iw, :]).max()*2/scalefactor #offset when plot
    fig = plt.figure(figsize = (8, 6), dpi = 80)
    for ip in range(Lpnt):
      plt.plot(t, pvar[:, iw, ip] + traceshift*ip, 'b')
#     plt.plot(xbgl, ybgl*traceshift + traceshift*ip, ':k', linewidth = 0.5) #for horinzital line
#   plt.plot(xbgl, ybgl*traceshift + traceshift*Lpnt, ':k', linewidth = 0.5)
    plt.yticks(traceshift*(np.arange(Lpnt + 1.0) - 0.5), np.arange(Lpnt + 1))
    plt.title('Seismogram of No.{} seismo-line for {}'.format(index,
      pname[iw]))
    plt.xlabel('time (s)')
    print('unit=traceshift=',traceshift)
    if iw < 3:
      plt.ylabel('velocity ({:.2e} m/s)'.format(traceshift))
    else:
      plt.ylabel('stress ({:.2e} MPa)'.format(traceshift))
else:
  # ==== VELOCITY ====
  vmin = np.min(pvar[:,0:3,0])
  vmax = np.max(pvar[:,0:3,0])
  print('vmin=',vmin, 'vmax=',vmax);
  #vmin = np.floor(vmin)
  #vmax = np.ceil(vmax)
  vmax = vmax*1.2
  vmin = vmin*1.2
  #vmin = -0.001
  #vmax = 0.0015
  fig = plt.figure(figsize = (8, 6), dpi = 80)
  fig.subplots_adjust(hspace=0.5)# adjust distance between small plots
  # -- Vx  --
  plt.subplot(311).plot(t, pvar[:, 0, 0], 'b')
  plt.title('Seismogram of No.{} seismo-point for '.format(index) + pname[0])
  plt.gca().get_xaxis().set_visible(False)
  plt.ylabel('velocity (m/s)')
  if restriction:
    plt.ylim([vmin,vmax])
  # -- Vy  --
  plt.subplot(312).plot(t, pvar[:, 1, 0], 'b')
  plt.title('Seismogram of No.{} seismo-point for '.format(index) + pname[1])
  plt.gca().get_xaxis().set_visible(False)
  plt.ylabel('velocity (m/s)')
  if restriction:
    plt.ylim([vmin,vmax])
  # -- Vz  --
  plt.subplot(313).plot(t, pvar[:, 2, 0], 'b')
  plt.title('Seismogram of No.{} seismo-point for '.format(index) + pname[2])
  plt.xlabel('time (s)')
  plt.ylabel('velocity (m/s)')
  if restriction:
    plt.ylim([vmin,vmax])
  
  
  # ==== STRESS ====
  vmin = np.min(pvar[:,3:9,0])
  vmax = np.max(pvar[:,3:9,0])
  #vmin = np.floor(vmin)
  #vmax = np.ceil(vmax)
  vmax = vmax*1.2
  vmin = vmin*1.2
  fig = plt.figure(figsize = (8, 6), dpi = 80)
  fig.subplots_adjust(hspace=0.7) # 10. for with xlabel
  # -- Txx --
  plt.subplot(611).plot(t, pvar[:, 3, 0], 'b')
  plt.title('Seismogram of No.{} seismo-point for '.format(index) + pname[3])
  plt.gca().get_xaxis().set_visible(False)
  plt.ylabel('stress (MPa)')
  if restriction:
    plt.ylim([vmin,vmax])
  # -- Tyy --
  plt.subplot(612).plot(t, pvar[:, 4, 0], 'b')
  plt.title('Seismogram of No.{} seismo-point for '.format(index) + pname[4])
  plt.gca().get_xaxis().set_visible(False)
  plt.ylabel('stress (MPa)')
  if restriction:
    plt.ylim([vmin,vmax])
  # -- Tzz --
  plt.subplot(613).plot(t, pvar[:, 5, 0], 'b')
  plt.title('Seismogram of No.{} seismo-point for '.format(index) + pname[5])
  plt.gca().get_xaxis().set_visible(False)
  plt.ylabel('stress (MPa)')
  if restriction:
    plt.ylim([vmin,vmax])
  # -- Txy --
  plt.subplot(614).plot(t, pvar[:, 6, 0], 'b')
  plt.title('Seismogram of No.{} seismo-point for '.format(index) + pname[6])
  plt.gca().get_xaxis().set_visible(False)
  plt.ylabel('stress (MPa)')
  if restriction:
    plt.ylim([vmin,vmax])
  # -- Txz --
  plt.subplot(615).plot(t, pvar[:, 7, 0], 'b')
  plt.title('Seismogram of No.{} seismo-point for '.format(index) + pname[7])
  plt.gca().get_xaxis().set_visible(False)
  plt.ylabel('stress (MPa)')
  if restriction:
    plt.ylim([vmin,vmax])
  # -- Tyz --
  plt.subplot(616).plot(t, pvar[:, 8, 0], 'b')
  plt.title('Seismogram of No.{} seismo-point for '.format(index) + pname[8])
  plt.xlabel('time (s)')
  plt.ylabel('stress (MPa)')
  if restriction:
    plt.ylim([vmin,vmax])
plt.show()


