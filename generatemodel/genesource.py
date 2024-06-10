#!/usr/bin/env python3
# coding: utf-8

#this programe generate test source time funtion for FORTRAN program


import numpy as np
import struct
from scipy.interpolate import griddata
from scipy import integrate
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys, getopt, os.path
import random


def mRicker(x,fc,x0):
	if(x<=0):
		y=0
	else:
		u = (x-x0)*np.pi*fc
		y = -np.pi*fc*u*(6-4*u*u)*np.exp(-u*u)
	return y

def calwave(start, t0, f0, lengthtime, dt, stf):
	for i in range(lengthtime):
		t = i*dt-start
		stf[i] = mRicker(t,f0,t0)

def extwave(stf, tstart, tend, DT, dt, currt):
	time = currt*dt
	if( time > tend or time < tstart):
		value = 0.0
	else:
		if(np.abs(time - DT*np.floor(time/DT)) < 1e-5 ):
			PL = np.int(time/DT)
			PR = PL
		else:
			PL = np.int(time/DT)
			PR = PL+1
		value = stf[PL] + (stf[PR]-stf[PL])*( (time-DT*PL)/DT )
	
	return value

#------------------------------------------

start = 0.0
t0 = 0.65
f0 = 1.8
dt = 0.006
nt = 1500
NP = 1

M0Scale=1e6 #M0 with mu should use 1e16, only moment rate shoule use 1e6

#------------------------------------------
strike = np.ones((NP,1))*0
dip = np.ones((NP,1))*90
rake = np.ones((NP,nt))*180

#--------------------------------------
stf = np.zeros((NP,nt))
time = np.arange(0,nt,1)
time1 = time*dt

calwave(start, t0, f0, nt, dt, stf[0,:])

stf = M0Scale*stf
Dstf = integrate.cumtrapz( stf[0,:], time1, axis=0, initial=0)

plt.figure(1)
plt.plot(time1,stf[0,:],label='vel')
plt.plot(time1,Dstf,label='dis')
plt.legend()
plt.show()

#exit()

#----------------------------------
posx = np.zeros(NP)
posy = np.zeros(NP)
posz = np.zeros(NP)

posx[0] = 5
posy[0] = 5
posz[0] = -5


posx = posx*1e3
posy = posy*1e3
posz = posz*1e3

#----------------------------------------------


print('\n------start to write NC file------\n')

#----------------------------------------------------

filename = './stf-GPU.nc'

print('write GPU source',filename)

snc = nc.Dataset(filename, mode='w', format='NETCDF4')

#define dimension
snc.createDimension('NP', NP)   #DIFF-for&gpu
snc.createDimension('NT', nt)   #DIFF-for&gpu

#define var
snc.createVariable('Mxx',datatype='f4', dimensions=('NP','NT')) #DIFF-for&gpu
snc.createVariable('Myy',datatype='f4', dimensions=('NP','NT'))
snc.createVariable('Mzz',datatype='f4', dimensions=('NP','NT'))
snc.createVariable('Mxy',datatype='f4', dimensions=('NP','NT'))
snc.createVariable('Mxz',datatype='f4', dimensions=('NP','NT'))
snc.createVariable('Myz',datatype='f4', dimensions=('NP','NT'))
snc.createVariable('posx',datatype='f4', dimensions=('NP'))
snc.createVariable('posy',datatype='f4', dimensions=('NP'))
snc.createVariable('posz',datatype='f4', dimensions=('NP'))

#write var
snc.variables['Mxx'][:,:] = stf[:,:]    
snc.variables['Myy'][:,:] = stf[:,:]
snc.variables['Mzz'][:,:] = stf[:,:]
snc.variables['Mxy'][:,:] = 0
snc.variables['Mxz'][:,:] = 0
snc.variables['Myz'][:,:] = 0
snc.variables['posx'][:] = posx[:]
snc.variables['posy'][:] = posy[:]
snc.variables['posz'][:] = posz[:]

#add global attributes
snc.setncattr('DT',dt)  
snc.setncattr('distance_unit','meter')
snc.setncattr('time_unit','second')
#snc.setncattr('research_area_lon_start',lon1)
#snc.setncattr('research_area_lon_end',lon2)
#snc.setncattr('research_area_lat_start',lat1)
#snc.setncattr('research_area_lat_end',lat2)

#close file
snc.close()



print('NC file OK!')



