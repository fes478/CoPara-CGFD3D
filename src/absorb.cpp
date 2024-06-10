#include "typenew.h"
#include<math.h>
#include<typeinfo>

using namespace std;
using namespace defstruct;
using namespace constant;

#define errprt(...) com.errorprint(__FILE__, __LINE__, __VA_ARGS__)

//-------------------private--------------------------------
void absorb::getConf(const char *filename)
{
	char parpath[SeisStrLen];
	char name[SeisStrLen2];
	char errstr[SeisStrLen];
	FILE *fp;
	int i;

	fp = fopen(filename,"r");
	if(!fp)
	{
		sprintf(errstr,"Fail to open main par file %s in absorb.cpp", filename);
		errprt(Fail2Open,errstr);
	}
	com.get_conf(fp, "seispath", 3, parpath);
	com.get_conf(fp, "absorb_filename", 3, name);
	fclose(fp);
	
	sprintf(absfile,"%s/%s",parpath,name);

	fp = fopen(absfile,"r");
	if(!fp)
	{
		sprintf(errstr,"Fail to open absorb configure file %s in absorb.cpp", absfile);
		errprt(Fail2Open,errstr);
	}

	for(i=0;i<numblk;i++)
	{
		com.get_conf(fp,"abs_number",3+i,&nabs[i]);
		com.get_conf(fp,"abs_velocity",3+i,&velabs[i]);
		com.get_conf(fp,"bmax",3+i,&CFS_bmax[i]);
		com.get_conf(fp,"amax",3+i,&CFS_amax[i]);
	}
	fclose(fp);

}

Real absorb::CalExp(int loc, Real vs, Real L, int ni, Real stept)
{
	Real D;
	int m,n;
	D = 0.0;
	
	m = (int)( (ni*L*1.0)/(vs*stept) );//segment number
	
	for(n=1;n<=m;n++)
		D = D + pow(n*stept*vs,2)/pow(ni*L,2);

	D = 0.8/D*1.1;
	
	D = exp( -D*pow(loc*1.0/ni,2) );
	
	return D;
}

void absorb::GridCovariant(deriv drv, int i, int j, int k, int n, Real *vecg)
{
	Real vec1[3]={drv.xi_x[i][j][k], drv.xi_y[i][j][k], drv.xi_z[i][j][k]};
	Real vec2[3]={drv.eta_x[i][j][k], drv.eta_y[i][j][k], drv.eta_z[i][j][k]};
	Real vec3[3]={drv.zeta_x[i][j][k], drv.zeta_y[i][j][k], drv.zeta_z[i][j][k]};
	
	if(n==1)
		mathf.crossproduct(vec2,vec3,vecg);
	else if(n==2)
		mathf.crossproduct(vec3,vec1,vecg);
	else
		mathf.crossproduct(vec1,vec2,vecg);

	for(int m=0;m<3;m++)
		vecg[m] = vecg[m]*drv.jac[i][j][k];
	
}



//---------------------public-------------------------------------
absorb::absorb(const char *filename, cindx cdx, const int restart, const int Myid)
{
	myid = Myid;
	if(myid)
	{
		Mflag = false;//child procs
		printf("child procs %d doesn't paticipate into computing parameters (source)\n",myid);
		return;
	}
	else
		Mflag = true;//master procs

	if(restart==1)
		Rwork = true;
	else
		Rwork = false;

	fprintf(stdout,"***malloc and init Absorb Sponge Layer or Conplex Frequency Shift damping factor\n");
	
	this->numblk = SeisGeo*2;//2X 2Y 2Z
	this->pmlblk = (6+8+12)*SeisGeo*2;//6 surface 8 corner 12 edge

	nabs = new int[numblk]();
	velabs = new Real[numblk]();

	CFS_bmax = new Real[numblk]();
	CFS_amax = new Real[numblk]();

	CLoc = new int[pmlblk*6]();//include bounds
	ELoc = new int[numblk*6]();//include bounds
	
//------------------------------------------------------------------		
	apr.nabs = new int[numblk]();
	apr.CLoc = new int[pmlblk*6]();//include bounds
	apr.APDx = new Real[cdx.nx]();
	apr.APDy = new Real[cdx.ny]();
	apr.APDz = new Real[cdx.nz]();
	apr.Bx = new Real[cdx.nx]();
	apr.By = new Real[cdx.ny]();
	apr.Bz = new Real[cdx.nz]();
	apr.DBx = new Real[cdx.nx]();
	apr.DBy = new Real[cdx.ny]();
	apr.DBz = new Real[cdx.nz]();
	apr.ELoc = new int[numblk*6]();//include bounds
	apr.Ex = new Real[cdx.nx]();
	apr.Ey = new Real[cdx.ny]();
	apr.Ez = new Real[cdx.nz]();

//----------------------------------------------------------------	
	getConf(filename);

	//check absorbing layer number's plausibility
	char errstr[SeisStrLen];
#ifdef CondFree
	if(nabs[5]>0)
	{
		fprintf(stdout,"When apply the free surface condition, the absorbing layer number of top-Z-direction will be set to 0!\n");
		nabs[5]=0;
	}
#endif
	memcpy(apr.nabs,nabs,numblk*sizeof(int));
	
	for(int i=0;i<numblk;i++)
		if(nabs[i]<0)
		{
			sprintf(errstr,"the %d-th value %d for abs_number must be no negetive (check %s in absorb.cpp)", i,nabs[i],absfile);
			errprt(Fail2Check,errstr);
		}
	if(nabs[0]+nabs[1]>=cdx.ni)
		errprt(Fail2Check,"In X-direction, total absorbing layer number exceed the valid physical domain!");
	if(nabs[2]+nabs[3]>=cdx.nj)
		errprt(Fail2Check,"In Y-direction, total absorbing layer number exceed the valid physical domain!");
	if(nabs[4]+nabs[5]>=cdx.nk)
		errprt(Fail2Check,"In Z-direction, total absorbing layer number exceed the valid physical domain!");

}
absorb::~absorb()
{
	fprintf(stdout,"into data free at Procs[%d],in absorb.cpp\n",myid);
	if(Mflag)
	{
		delete [] ELoc;
		delete [] CLoc;

		delete [] CFS_amax;
		delete [] CFS_bmax;

		delete [] velabs;
		delete [] nabs;

		delete [] apr.DBz;
		delete [] apr.DBy;
		delete [] apr.DBx;
		delete [] apr.Bz;
		delete [] apr.By;
		delete [] apr.Bx;
		delete [] apr.APDz;
		delete [] apr.APDy;
		delete [] apr.APDx;
		delete [] apr.CLoc;
		delete [] apr.Ez;
		delete [] apr.Ey;
		delete [] apr.Ex;
		delete [] apr.ELoc;
		delete [] apr.nabs;
	}
	fprintf(stdout,"data free at Procs[%d],in absorb.cpp\n",myid);
}

void absorb::CalSLDamping(cindx cdx, deriv drv, Real steph, Real stept)
{
	fprintf(stdout,"***Start to calculate the damping factor of the Sponge Layer in absorb porgram\n");
	int i,j,k;
	//for lay thickness(point-type) calculation
	int ni1,ni2,ni;
	int nj1,nj2,nj;
	int nk1,nk2,nk;

	Real vecg[3];
	Real Length;
	Real *Ex,*Ey,*Ez;//damping factor(sponge layer)
	//from location index small to large in valid region,
	//in X1,X2,Y1,Y2,Z1,Z2 sequence,
	//for overlapped area, doesn't apply repeatedly,
	//
	//shoule work as this diagram
	//
	// X X X X X X X X X X
	// X X X X X X X X X X
	// Y Y Z Z Z Z Z Z Y Y
	// Y Y Z Z Z Z Z Z Y Y
	// Y Y Z Z Z Z Z Z Y Y
	// Y Y Z Z Z Z Z Z Y Y
	// Y Y Z Z Z Z Z Z Y Y
	// Y Y Z Z Z Z Z Z Y Y
	// X X X X X X X X X X
	// X X X X X X X X X X
	
	Ex = new Real[cdx.nx]();
	Ey = new Real[cdx.ny]();
	Ez = new Real[cdx.nz]();
	for(i=0;i<cdx.nx;i++)
		Ex[i] = 1.0;
	for(i=0;i<cdx.ny;i++)
		Ey[i] = 1.0;
	for(i=0;i<cdx.nz;i++)
		Ez[i] = 1.0;
	
	//X1
	ni = nabs[0];	
	ni2 = cdx.ni1+ni-1; 
	for(i=cdx.ni1;i<=ni2;i++)
	{
		j=cdx.nj1;
		k=cdx.nk1;
		GridCovariant(drv,i,j,k,1,vecg);
		Length=sqrt( mathf.dotproduct(vecg,vecg,3) )*steph;
		Ex[i] = CalExp( ni-(i-cdx.ni1), velabs[0], Length, ni ,stept);
	}
	
	ELoc[0*6+0] = cdx.ni1;	ELoc[0*6+1] = ni2;
	ELoc[0*6+2] = cdx.nj1;	ELoc[0*6+3] = cdx.nj2-1;
	ELoc[0*6+4] = cdx.nk1;	ELoc[0*6+5] = cdx.nk2-1;

	//X2
	ni = nabs[1];
	ni1 = cdx.ni2-ni;
	for(i=ni1; i<cdx.ni2;i++)
	{
		j=cdx.nj1;
		k=cdx.nk1;
		GridCovariant(drv,i,j,k,1,vecg);
		Length=sqrt( mathf.dotproduct(vecg,vecg,3) )*steph;
		Ex[i] = CalExp( i-ni1+1, velabs[1], Length, ni ,stept);
	}
	
	ELoc[1*6+0] = ni1;	ELoc[1*6+1] = cdx.ni2-1;
	ELoc[1*6+2] = cdx.nj1;	ELoc[1*6+3] = cdx.nj2-1;
	ELoc[1*6+4] = cdx.nk1;	ELoc[1*6+5] = cdx.nk2-1;

	//Y1
	nj = nabs[2];
	nj2 = cdx.nj1+nj-1;
	for(j=cdx.nj1;j<=nj2;j++)
	{
		i=cdx.ni1+nabs[0];
		k=cdx.nk1;
		GridCovariant(drv,i,j,k,2,vecg);
		Length=sqrt( mathf.dotproduct(vecg,vecg,3) )*steph;
		Ey[j] = CalExp( nj-(j-cdx.nj1), velabs[2], Length, nj ,stept);
	}
	
	ELoc[2*6+0] = cdx.ni1+nabs[0];	ELoc[2*6+1] = cdx.ni2-nabs[1]-1;
	ELoc[2*6+2] = cdx.nj1;		ELoc[2*6+3] = nj2;
	ELoc[2*6+4] = cdx.nk1;		ELoc[2*6+5] = cdx.nk2-1;

	//Y2
	nj = nabs[3];
	nj1 = cdx.nj2-nj;

	for(j=nj1;j<cdx.nj2;j++)
	{
		i=cdx.ni1+nabs[0];
		k=cdx.nk1;
		GridCovariant(drv,i,j,k,2,vecg);
		Length=sqrt( mathf.dotproduct(vecg,vecg,3) )*steph;
		Ey[j] = CalExp( j-nj1+1, velabs[3], Length, nj ,stept);
	}
	
	ELoc[3*6+0] = cdx.ni1+nabs[0];	ELoc[3*6+1] = cdx.ni2-nabs[1]-1;
	ELoc[3*6+2] = nj1;		ELoc[3*6+3] = cdx.nj2-1;
	ELoc[3*6+4] = cdx.nk1;		ELoc[3*6+5] = cdx.nk2-1;

	//Z1
	nk = nabs[4];
	nk2 = cdx.nk1+nk-1;
	for(k=cdx.nk1;k<=nk2;k++)
	{
		i=cdx.ni1+nabs[0];
		j=cdx.nj1+nabs[2];
		GridCovariant(drv,i,j,k,3,vecg);
		Length=sqrt( mathf.dotproduct(vecg,vecg,3) )*steph;
		Ez[k] = CalExp( nk-(k-cdx.nk1), velabs[4], Length, nk ,stept);
	}
	
	ELoc[4*6+0] = cdx.ni1+nabs[0];	ELoc[4*6+1] = cdx.ni2-nabs[1]-1;
	ELoc[4*6+2] = cdx.nj1+nabs[2];	ELoc[4*6+3] = cdx.nj2-nabs[3]-1;
	ELoc[4*6+4] = cdx.nk1;		ELoc[4*6+5] = nk2;

	//Z2
	nk = nabs[5];
	nk1 = cdx.nk2-nk;
	for(k=nk1;k<cdx.nk2;k++)
	{
		i=cdx.ni1+nabs[0];
		j=cdx.nj1+nabs[2];
		GridCovariant(drv,i,j,k,3,vecg);
		Length=sqrt( mathf.dotproduct(vecg,vecg,3) )*steph;
		Ez[k] = CalExp( k-nk1+1, velabs[5], Length, nk ,stept);
	}
	
	ELoc[5*6+0] = cdx.ni1+nabs[0];	ELoc[5*6+1] = cdx.ni2-nabs[1]-1;
	ELoc[5*6+2] = cdx.nj1+nabs[2];	ELoc[5*6+3] = cdx.nj2-nabs[3]-1;
	ELoc[5*6+4] = nk1;		ELoc[5*6+5] = cdx.nk2-1;

	memcpy(apr.ELoc, ELoc, numblk*6*sizeof(int));
	memcpy(apr.Ex, Ex, cdx.nx*sizeof(Real));
	memcpy(apr.Ey, Ey, cdx.ny*sizeof(Real));
	memcpy(apr.Ez, Ez, cdx.nz*sizeof(Real));

	delete [] Ez;
	delete [] Ey;
	delete [] Ex;

	fprintf(stdout,"---accomplished to calculate the damping factor of the Sponge Layer in absorb porgram\n");
}

void absorb::CalCFSfactors(cindx cdx, deriv drv, Real steph, Real stept)
{
	//Zhang 2010 ADE CFS PML
	fprintf(stdout,"***Start to calculate the related factor of the ADE CFS PML in absorb porgram\n");
	int i,j,k;
	//for lay thickness(point-type) calculation
	int ni1,ni2,ni;
	int nj1,nj2,nj;
	int nk1,nk2,nk;

	Real Rpp[this->numblk],dmax[this->numblk];

	Real vecg[3];
	Real L0,Lx[cdx.nx],Ly[cdx.ny],Lz[cdx.nz];
	Real PA,PB,PD;//constant of exponent part
	Real *Ax,*Ay,*Az, *Bx,*By,*Bz, *Dx,*Dy,*Dz;//shifting,scaling,damping factors(ADE CFS PML)

	//from location index small to large in valid region,
	//in X1,X2,Y1,Y2,Z1,Z2 sequence,
	//for overlapped area, doesn't apply repeatedly,
	//
	//shoule work as this diagram
	
	//                                
	//                             
	//                          
	//        Z-dir          O-----------------------------O
	//        |           x  |                          x  |
	//        |        x     |                       x     |
	//        |     x        |                    x        |
	//        |  x           |                 x           |
	//        O-----------------------------O              |
	//        |              |              |              |
	//        |              |              |              |
	//        |              |              |              |
	//        |              |        Y-dir |              |
	//        |              |     x        |              |
	//        |              |  x           |              |
	//        |              0--------------|--------------O
	//        |           x                 |           x
	//        |        x                    |        x
	//        |     x                       |     x
	//        |  x                          |  x
	//        O-----------------------------O-------------X-dir
	//
	//
	//       AbsLoc index change sequence X Y Z
	//	 Index contains bounds
	//	 The apply scope will not repeated
	//
	//
	//        O---------------------------------------------O
	//        |          |                       |          |  
	//        |          |                       |          |  
	//        |  corner  |         edge          |  corner  |  
	//        |          |                       |          |  
	//        |          |                       |          |  
	//        o----------o-----------------------o----------o				
	//        |          |                       |          |  
	//        |          |                       |          |  
	//        |          |                       |          |  
	//        |          |                       |          |  
	//        |   edge   |        surface        |   edge   |  
	//        |          |                       |          |  
	//        |          |                       |          |  
	//        |          |                       |          |  
	//        |          |                       |          |  
	//        o----------o-----------------------o----------o				
	//        |          |                       |          |
	//        |          |                       |          |
	//        |  corner  |         edge          |  corner  |
	//        |          |                       |          |
	//        |          |                       |          |
	//        O----------o-----------------------o----------O
	
	Ax = new Real[cdx.nx]();
	Ay = new Real[cdx.ny]();
	Az = new Real[cdx.nz]();
	Bx = new Real[cdx.nx]();
	By = new Real[cdx.ny]();
	Bz = new Real[cdx.nz]();
	Dx = new Real[cdx.nx]();
	Dy = new Real[cdx.ny]();
	Dz = new Real[cdx.nz]();
	
	for(i=0;i<cdx.nx;i++)
	{
		Ax[i] = 0.0;//min exterior
		Bx[i] = 1.0;//min interior
		Dx[i] = 0.0;//min interior
	}
	for(i=0;i<cdx.ny;i++)
	{
		Ay[i] = 0.0;//min exterior
		By[i] = 1.0;//min interior
		Dy[i] = 0.0;//min interior
	}
	for(i=0;i<cdx.nz;i++)
	{
		Az[i] = 0.0;//min exterior
		Bz[i] = 1.0;//min interior
		Dz[i] = 0.0;//min interior
	}

	PD = 2.0;	PB = 2.0;	PA = 1.0;

	for(i=0;i<this->numblk;i++)
	{
		Rpp[i] = 0.0;
		dmax[i] = 0.0;
	}

	for(i=0;i<cdx.nx;i++)
		Lx[i] = 0.0;
	for(j=0;j<cdx.ny;j++)
		Ly[j] = 0.0;
	for(k=0;k<cdx.nz;k++)
		Lz[k] = 0.0;

	//calculate theoritical Reflection coefficients
	for(i=0;i<this->numblk;i++)
		Rpp[i] = pow( 10.0 , -(log10(nabs[i]*1.0)-1)/log10(2.0)-3.0 );
	
	//X1
	ni = nabs[0];	
	ni2 = cdx.ni1+ni-1; 
	for(i=ni2;i>=cdx.ni1;i--)
	{
		j=cdx.nj1;
		k=cdx.nk1;
		vecg[0] = drv.xi_x[i][j][k];	vecg[1] = drv.xi_y[i][j][k];	vecg[2] = drv.xi_z[i][j][k];
		Lx[i] = Lx[i+1] + steph/sqrt( mathf.dotproduct(vecg,vecg,3) );
	}
	L0 = Lx[cdx.ni1];
	dmax[0] = -velabs[0]/2.0/L0*log(Rpp[0])*(PD+1.0); 
	for(i=cdx.ni1;i<=ni2;i++)
	{
		if(Lx[i]<0)
		{
			Dx[i]=0.0;
			Bx[i]=1.0;
			Ax[i]=0.0;
		}
		else
		{
			Dx[i]=dmax[0]*pow(Lx[i]/L0,PD);
			Bx[i]=1.0+(CFS_bmax[0]-1.0)*pow(Lx[i]/L0,PB);
			Ax[i]=CFS_amax[0]*(1.0-pow(Lx[i]/L0,PA));
		}
	}


	//X2
	ni = nabs[1];
	ni1 = cdx.ni2-ni;
	for(i=ni1; i<cdx.ni2;i++)
	{
		j=cdx.nj1;
		k=cdx.nk1;
		vecg[0] = drv.xi_x[i][j][k];	vecg[1] = drv.xi_y[i][j][k];	vecg[2] = drv.xi_z[i][j][k];
		Lx[i] = Lx[i-1] + steph/sqrt( mathf.dotproduct(vecg,vecg,3) );
	}
	L0 = Lx[cdx.ni2-1];
	dmax[1] = -velabs[1]/2.0/L0*log(Rpp[1])*(PD+1.0);
	for(i=ni1;i<cdx.ni2;i++)
	{
		if(Lx[i]<0)
		{
			Dx[i]=0.0;
			Bx[i]=1.0;
			Ax[i]=0.0;
		}
		else
		{
			Dx[i]=dmax[1]*pow(Lx[i]/L0,PD);
			Bx[i]=1.0+(CFS_bmax[1]-1.0)*pow(Lx[i]/L0,PB);
			Ax[i]=CFS_amax[1]*(1.0-pow(Lx[i]/L0,PA));
		}
	}
	

	//Y1
	nj = nabs[2];
	nj2 = cdx.nj1+nj-1;
	for(j=nj2;j>=cdx.nj1;j--)
	{
		i=cdx.ni1;
		k=cdx.nk1;
		vecg[0] = drv.eta_x[i][j][k];	vecg[1] = drv.eta_y[i][j][k];	vecg[2] = drv.eta_z[i][j][k];
		Ly[j] = Ly[j+1] + steph/sqrt( mathf.dotproduct(vecg,vecg,3) );
	}
	L0 = Ly[cdx.nj1];
	dmax[2] = -velabs[2]/2.0/L0*log(Rpp[2])*(PD+1.0);
	for(j=cdx.nj1;j<=nj2;j++)
	{
		if(Ly[j]<0)
		{
			Dy[j]=0.0;
			By[j]=1.0;
			Ay[j]=0.0;
		}
		else
		{
			Dy[j]=dmax[2]*pow(Ly[j]/L0,PD);
			By[j]=1.0+(CFS_bmax[2]-1.0)*pow(Ly[j]/L0,PB);
			Ay[j]=CFS_amax[2]*(1.0-pow(Ly[j]/L0,PA));
		}
	}
	

	//Y2
	nj = nabs[3];
	nj1 = cdx.nj2-nj;
	for(j=nj1;j<cdx.nj2;j++)
	{
		i=cdx.ni1;
		k=cdx.nk1;
		vecg[0] = drv.eta_x[i][j][k];	vecg[1] = drv.eta_y[i][j][k];	vecg[2] = drv.eta_z[i][j][k];
		Ly[j] = Ly[j-1] + steph/sqrt( mathf.dotproduct(vecg,vecg,3) );
	}
	L0 = Ly[cdx.nj2-1];
	dmax[3] = -velabs[3]/2.0/L0*log(Rpp[3])*(PD+1.0);
	for(j=nj1;j<cdx.nj2;j++)
	{
		if(Ly[j]<0)
		{
			Dy[j]=0.0;
			By[j]=1.0;
			Ay[j]=0.0;
		}
		else
		{
			Dy[j]=dmax[3]*pow(Ly[j]/L0,PD);
			By[j]=1.0+(CFS_bmax[3]-1.0)*pow(Ly[j]/L0,PB);
			Ay[j]=CFS_amax[3]*(1.0-pow(Ly[j]/L0,PA));
		}
	}
	

	//Z1
	nk = nabs[4];
	nk2 = cdx.nk1+nk-1;
	for(k=nk2;k>=cdx.nk1;k--)
	{
		i=cdx.ni1;
		j=cdx.nj1;
		vecg[0] = drv.zeta_x[i][j][k];	vecg[1] = drv.zeta_y[i][j][k];	vecg[2] = drv.zeta_z[i][j][k];
		Lz[k] = Lz[k+1] + steph/sqrt( mathf.dotproduct(vecg,vecg,3) );
	}
	L0 = Lz[cdx.nk1];
	dmax[4] = -velabs[4]/2.0/L0*log(Rpp[4])*(PD+1.0);
	for(k=cdx.nk1;k<=nk2;k++)
	{
		if(Lz[k]<0)
		{
			Dz[k]=0.0;
			Bz[k]=1.0;
			Az[k]=0.0;
		}
		else
		{
			Dz[k]=dmax[4]*pow(Lz[k]/L0,PD);
			Bz[k]=1.0+(CFS_bmax[4]-1.0)*pow(Lz[k]/L0,PB);
			Az[k]=CFS_amax[4]*(1.0-pow(Lz[k]/L0,PA));
		}
	}
	

	//Z2
	nk = nabs[5];
	nk1 = cdx.nk2-nk;
	for(k=nk1;k<cdx.nk2;k++)
	{
		i=cdx.ni1;
		j=cdx.nj1;
		vecg[0] = drv.zeta_x[i][j][k];	vecg[1] = drv.zeta_y[i][j][k];	vecg[2] = drv.zeta_z[i][j][k];
		Lz[k] = Lz[k-1] + steph/sqrt( mathf.dotproduct(vecg,vecg,3) );
	}
	L0 = Lz[cdx.nk2-1];
	dmax[5] = -velabs[5]/2.0/L0*log(Rpp[5])*(PD+1.0);
	for(k=nk1;k<cdx.nk2;k++)
	{
		if(Lz[k]<0)
		{
			Dz[k]=0.0;
			Bz[k]=1.0;
			Az[k]=0.0;
		}
		else
		{
			Dz[k]=dmax[5]*pow(Lz[k]/L0,PD);
			Bz[k]=1.0+(CFS_bmax[5]-1.0)*pow(Lz[k]/L0,PB);
			Az[k]=CFS_amax[5]*(1.0-pow(Lz[k]/L0,PA));
		}
	}

	//corner(XYZ,	8:0-7)
	//X1Y1Z1
	CLoc[0*6+0] = cdx.ni1;		CLoc[0*6+1] = cdx.ni1+nabs[0]-1;
	CLoc[0*6+2] = cdx.nj1;		CLoc[0*6+3] = cdx.nj1+nabs[2]-1;
	CLoc[0*6+4] = cdx.nk1;		CLoc[0*6+5] = cdx.nk1+nabs[4]-1;
	
	//X2Y1Z1
	CLoc[1*6+0] = cdx.ni2-nabs[1];	CLoc[1*6+1] = cdx.ni2-1;
	CLoc[1*6+2] = cdx.nj1;		CLoc[1*6+3] = cdx.nj1+nabs[2]-1;
	CLoc[1*6+4] = cdx.nk1;		CLoc[1*6+5] = cdx.nk1+nabs[4]-1;
	
	//X1Y2Z1
	CLoc[2*6+0] = cdx.ni1;		CLoc[2*6+1] = cdx.ni1+nabs[0]-1;
	CLoc[2*6+2] = cdx.nj2-nabs[3];	CLoc[2*6+3] = cdx.nj2-1;
	CLoc[2*6+4] = cdx.nk1;		CLoc[2*6+5] = cdx.nk1+nabs[4]-1;

	//X2Y2Z1
	CLoc[3*6+0] = cdx.ni2-nabs[1];	CLoc[3*6+1] = cdx.ni2-1;
	CLoc[3*6+2] = cdx.nj2-nabs[3];	CLoc[3*6+3] = cdx.nj2-1;
	CLoc[3*6+4] = cdx.nk1;		CLoc[3*6+5] = cdx.nk1+nabs[4]-1;

	//X1Y1Z2
	CLoc[4*6+0] = cdx.ni1;		CLoc[4*6+1] = cdx.ni1+nabs[0]-1;
	CLoc[4*6+2] = cdx.nj1;		CLoc[4*6+3] = cdx.nj1+nabs[2]-1;
	CLoc[4*6+4] = cdx.nk2-nabs[5];	CLoc[4*6+5] = cdx.nk2-1;

	//X2Y1Z2
	CLoc[5*6+0] = cdx.ni2-nabs[1];	CLoc[5*6+1] = cdx.ni2-1;
	CLoc[5*6+2] = cdx.nj1;		CLoc[5*6+3] = cdx.nj1+nabs[2]-1;
	CLoc[5*6+4] = cdx.nk2-nabs[5];	CLoc[5*6+5] = cdx.nk2-1;

	//X1Y2Z2
	CLoc[6*6+0] = cdx.ni1;		CLoc[6*6+1] = cdx.ni1+nabs[0]-1;
	CLoc[6*6+2] = cdx.nj2-nabs[3];	CLoc[6*6+3] = cdx.nj2-1;
	CLoc[6*6+4] = cdx.nk2-nabs[5];	CLoc[6*6+5] = cdx.nk2-1;

	//X2Y2Z2
	CLoc[7*6+0] = cdx.ni2-nabs[1];	CLoc[7*6+1] = cdx.ni2-1;
	CLoc[7*6+2] = cdx.nj2-nabs[3];	CLoc[7*6+3] = cdx.nj2-1;
	CLoc[7*6+4] = cdx.nk2-nabs[5];	CLoc[7*6+5] = cdx.nk2-1;


	//edge(XY XZ YZ,	12:8-19)
	//x1y1
	CLoc[8*6+0] = cdx.ni1;		CLoc[8*6+1] = cdx.ni1+nabs[0]-1;
	CLoc[8*6+2] = cdx.nj1;		CLoc[8*6+3] = cdx.nj1+nabs[2]-1;
	CLoc[8*6+4] = cdx.nk1+nabs[4];	CLoc[8*6+5] = cdx.nk2-nabs[5]-1;

	//x2y1
	CLoc[9*6+0] = cdx.ni2-nabs[1];	CLoc[9*6+1] = cdx.ni2-1;
	CLoc[9*6+2] = cdx.nj1;		CLoc[9*6+3] = cdx.nj1+nabs[2]-1;
	CLoc[9*6+4] = cdx.nk1+nabs[4];	CLoc[9*6+5] = cdx.nk2-nabs[5]-1;

	//x1y2
	CLoc[10*6+0] = cdx.ni1;		CLoc[10*6+1] = cdx.ni1+nabs[0]-1;
	CLoc[10*6+2] = cdx.nj2-nabs[3];	CLoc[10*6+3] = cdx.nj2-1;
	CLoc[10*6+4] = cdx.nk1+nabs[4];	CLoc[10*6+5] = cdx.nk2-nabs[5]-1;

	//x2y2
	CLoc[11*6+0] = cdx.ni2-nabs[1];	CLoc[11*6+1] = cdx.ni2-1;
	CLoc[11*6+2] = cdx.nj2-nabs[3];	CLoc[11*6+3] = cdx.nj2-1;
	CLoc[11*6+4] = cdx.nk1+nabs[4];	CLoc[11*6+5] = cdx.nk2-nabs[5]-1;

	//x1z1
	CLoc[12*6+0] = cdx.ni1;		CLoc[12*6+1] = cdx.ni1+nabs[0]-1;
	CLoc[12*6+2] = cdx.nj1+nabs[2];	CLoc[12*6+3] = cdx.nj2-nabs[3]-1;
	CLoc[12*6+4] = cdx.nk1;		CLoc[12*6+5] = cdx.nk1+nabs[4]-1;

	//x2z1
	CLoc[13*6+0] = cdx.ni2-nabs[1];	CLoc[13*6+1] = cdx.ni2-1;
	CLoc[13*6+2] = cdx.nj1+nabs[2];	CLoc[13*6+3] = cdx.nj2-nabs[3]-1;
	CLoc[13*6+4] = cdx.nk1;		CLoc[13*6+5] = cdx.nk1+nabs[4]-1;

	//x1z2
	CLoc[14*6+0] = cdx.ni1;		CLoc[14*6+1] = cdx.ni1+nabs[0]-1;
	CLoc[14*6+2] = cdx.nj1+nabs[2];	CLoc[14*6+3] = cdx.nj2-nabs[3]-1;
	CLoc[14*6+4] = cdx.nk2-nabs[5];	CLoc[14*6+5] = cdx.nk2-1;

	//x2z2
	CLoc[15*6+0] = cdx.ni2-nabs[1];	CLoc[15*6+1] = cdx.ni2-1;
	CLoc[15*6+2] = cdx.nj1+nabs[2];	CLoc[15*6+3] = cdx.nj2-nabs[3]-1;
	CLoc[15*6+4] = cdx.nk2-nabs[5];	CLoc[15*6+5] = cdx.nk2-1;

	//y1z1
	CLoc[16*6+0] = cdx.ni1+nabs[0];	CLoc[16*6+1] = cdx.ni2-nabs[1]-1;
	CLoc[16*6+2] = cdx.nj1;		CLoc[16*6+3] = cdx.nj1+nabs[2]-1;
	CLoc[16*6+4] = cdx.nk1;		CLoc[16*6+5] = cdx.nk1+nabs[4]-1;

	//y2z1
	CLoc[17*6+0] = cdx.ni1+nabs[0];	CLoc[17*6+1] = cdx.ni2-nabs[1]-1;
	CLoc[17*6+2] = cdx.nj2-nabs[3];	CLoc[17*6+3] = cdx.nj2-1;
	CLoc[17*6+4] = cdx.nk1;		CLoc[17*6+5] = cdx.nk1+nabs[4]-1;

	//y1z2
	CLoc[18*6+0] = cdx.ni1+nabs[0];	CLoc[18*6+1] = cdx.ni2-nabs[1]-1;
	CLoc[18*6+2] = cdx.nj1;		CLoc[18*6+3] = cdx.nj1+nabs[2]-1;
	CLoc[18*6+4] = cdx.nk2-nabs[5];	CLoc[18*6+5] = cdx.nk2-1;

	//y2z2
	CLoc[19*6+0] = cdx.ni1+nabs[0];	CLoc[19*6+1] = cdx.ni2-nabs[1]-1;
	CLoc[19*6+2] = cdx.nj2-nabs[3];	CLoc[19*6+3] = cdx.nj2-1;
	CLoc[19*6+4] = cdx.nk2-nabs[5];	CLoc[19*6+5] = cdx.nk2-1;

	//surface(XYZ,	26:20-25)
	//x1
	CLoc[20*6+0] = cdx.ni1;		CLoc[20*6+1] = cdx.ni1+nabs[0]-1;
	CLoc[20*6+2] = cdx.nj1+nabs[2];	CLoc[20*6+3] = cdx.nj2-nabs[3]-1;
	CLoc[20*6+4] = cdx.nk1+nabs[4];	CLoc[20*6+5] = cdx.nk2-nabs[5]-1;

	//x2
	CLoc[21*6+0] = cdx.ni2-nabs[1];	CLoc[21*6+1] = cdx.ni2-1;
	CLoc[21*6+2] = cdx.nj1+nabs[2];	CLoc[21*6+3] = cdx.nj2-nabs[3]-1;
	CLoc[21*6+4] = cdx.nk1+nabs[4];	CLoc[21*6+5] = cdx.nk2-nabs[5]-1;

	//y1
	CLoc[22*6+0] = cdx.ni1+nabs[0];	CLoc[22*6+1] = cdx.ni2-nabs[1]-1;
	CLoc[22*6+2] = cdx.nj1;		CLoc[22*6+3] = cdx.nj1+nabs[2]-1;
	CLoc[22*6+4] = cdx.nk1+nabs[4];	CLoc[22*6+5] = cdx.nk2-nabs[5]-1;

	//y2
	CLoc[23*6+0] = cdx.ni1+nabs[0];	CLoc[23*6+1] = cdx.ni2-nabs[1]-1;
	CLoc[23*6+2] = cdx.nj2-nabs[3];	CLoc[23*6+3] = cdx.nj2-1;
	CLoc[23*6+4] = cdx.nk1+nabs[4];	CLoc[23*6+5] = cdx.nk2-nabs[5]-1;
	
	//z1
	CLoc[24*6+0] = cdx.ni1+nabs[0];	CLoc[24*6+1] = cdx.ni2-nabs[1]-1;
	CLoc[24*6+2] = cdx.nj1+nabs[2];	CLoc[24*6+3] = cdx.nj2-nabs[3]-1;
	CLoc[24*6+4] = cdx.nk1;		CLoc[24*6+5] = cdx.nk1+nabs[4]-1;

	//y2
	CLoc[25*6+0] = cdx.ni1+nabs[0];	CLoc[25*6+1] = cdx.ni2-nabs[1]-1;
	CLoc[25*6+2] = cdx.nj1+nabs[2];	CLoc[25*6+3] = cdx.nj2-nabs[3]-1;
	CLoc[25*6+4] = cdx.nk2-nabs[5];	CLoc[25*6+5] = cdx.nk2-1;

	memcpy(apr.CLoc, CLoc, pmlblk*6*sizeof(int));
	//original alpha beta and damping factor, used as initialization
	memcpy(apr.Bx, Bx, cdx.nx*sizeof(Real));
	memcpy(apr.By, By, cdx.ny*sizeof(Real));
	memcpy(apr.Bz, Bz, cdx.nz*sizeof(Real));
	
	//function type
	for(i=0;i<cdx.nx;i++)
	{
		apr.APDx[i] = Ax[i] + Dx[i]/Bx[i];
		apr.DBx[i] = Dx[i]/Bx[i];
	}
	
	for(j=0;j<cdx.ny;j++)
	{
		apr.APDy[j] = Ay[j] + Dy[j]/By[j];
		apr.DBy[j] = Dy[j]/By[j];
	}
	for(k=0;k<cdx.nz;k++)
	{
		apr.APDz[k] = Az[k] + Dz[k]/Bz[k];
		apr.DBz[k] = Dz[k]/Bz[k];
	}
	
	delete [] Dz;
	delete [] Dy;
	delete [] Dx;
	delete [] Bz;
	delete [] By;
	delete [] Bx;
	delete [] Az;
	delete [] Ay;
	delete [] Ax;

	fprintf(stdout,"---accomplished to calculate the related factor of the ADE CFS PML in absorb porgram\n");
}

