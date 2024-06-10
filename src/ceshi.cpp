#include "typenew.h"
#include<string.h>

using namespace std;
using namespace defstruct;
using namespace constant;

#define errprt(...) com.errorprint(__FILE__, __LINE__, __VA_ARGS__)

void generatewave(int *currT, int npnt, wfield wpoint, wfield wfake, cindx cdx)
{
	int i,j,k;
	int nx,ny,nz;
	int idx;
	Real value;
	
	(*currT)++;//simulate one step forward

	nx = cdx.nx; ny = cdx.ny; nz = cdx.nz;
	
	for(i=0;i<nx;i++)
		for(j=0;j<ny;j++)
			for(k=0;k<nz;k++)
			{
				idx = i*ny*nz + j*nz + k;
				value = (i+1)*1000*1000 + (j+1)*1000 + (k+1);
				value = value/1000000;
				wfake.Vx[idx] = 1*1000 + value;
				wfake.Vy[idx] = 2*1000 + value;
				wfake.Vz[idx] = 3*1000 + value;
				wfake.Txx[idx] = 4*1000 + value;
				wfake.Tyy[idx] = 5*1000 + value;
				wfake.Tzz[idx] = 6*1000 + value;
				wfake.Txy[idx] = 7*1000 + value;
				wfake.Txz[idx] = 8*1000 + value;
				wfake.Tyz[idx] = 9*1000 + value;
			}

	for(i=0;i<npnt;i++)
	{
		wpoint.Vx[i]= 1*1000 + (i+1);
		wpoint.Vy[i]= 2*1000 + (i+1);
		wpoint.Vz[i]= 3*1000 + (i+1);
		wpoint.Txx[i]= 4*1000 + (i+1);
		wpoint.Tyy[i]= 5*1000 + (i+1);
		wpoint.Tzz[i]= 6*1000 + (i+1);
		wpoint.Txy[i]= 7*1000 + (i+1);
		wpoint.Txz[i]= 8*1000 + (i+1);
		wpoint.Tyz[i]= 9*1000 + (i+1);
	}

}









