#include<mpi.h>
#include<math.h>
#include "typedata.h"
#include<string.h>
#include<unistd.h>

using namespace constant;
using namespace defstruct;
using namespace flatstruct;
using namespace std;

#define errprt(...) com.errorprint(__FILE__, __LINE__, __VA_ARGS__)

//-------------------------------------------Tag Flag---------------------------------------------------
//
//	1E6	--->	cdx, nfrc, nmnt, nstf	|	1-18, 20-22	|	simulation.cc
//			restart, steph, stept		23,24,25
//			nt 				26
//			ConIndex,HyGrid			27,28
//			nabs,Mid.np		        29,30
//			spt.Snp,Mid.fp			31,32
//			src.Rmnt.nt,src.Rmnt.dt         33,34
//			pvflag				35
//	2E6	--->	drv                     |       1-10            |	datadis.cc
//	3E6	--->	mpa                     |       1-3             |	datadis.cc
//	4E6	--->	frc                     |       1-8             |	datadis.cc
//	5E6	--->	moment,Rmom             |       1-11,20+(1-10)  |	datadis.cc
//	6E6	--->	currT,absEXP,absPML     |       2,3-7,8-18      |	RestartS/R,datadis.cc
//	7E6	--->	fullwave                |       1-9             |	datadis.cc
//	7E6	--->	restartwave             |       20+(1-9)        |	simulation.cc
//	7E6	--->	peak velocity           |       50+(1-3)        |	simulation.cc
//	8E6	--->	bounds                  |       18(9*2)         |	datadis.cc
//			boundscomm(GA,SC)	|	20+18,40+18	|	datadis.cc
//	0E6	--->    CPNum,CopySize,Xsize    |	1K,2K,3K	|	datadis.cc(SplitM2C)
//      9E6     --->    Hpt,HPW                 |       1-5,20-9+Gsn    |       datadis.cc
//      1E6     --->    HSpt,HSW(1E7)           |       1-7,nsnap*npnt  |       datadis.cc
//      9E6     --->    Fpt                     |       51-55,          |       datadis.cc

//------------------------------------snap transfer---------------------------------------------
//transfer snap buffer
void mpiSWSend(wfield *HSW, int myid, int currT, int nt, int nsnap, int *CSpn, SnapIndexBufferF *HSpt)
{
	int i,j;
	int Tlen,nTime;
	int tag;
	int Dtag;
	int SmallFlag=50;//below for nsnap,upper for VT and myid
	Dtag = 1E7;

	for(j=0;j<nsnap;j++)
	{
		Tlen = ceil(1.0*currT/HSpt[j].tinv);
		nTime = ceil(1.0*nt/HSpt[j].tinv);
		
		/*
		//single point temporal slice
		for(i=0;i<CSpn[j];i++)
		{
			//   (myid+50, NO.snap)1E7 + (VT)1E6 + Gsn 	
			tag = Dtag*((SmallFlag+myid) + j) + HSpt[j].Gsn[i] + (1E6)*1;//1E7  point type
			MPI_Send( HSW[j].Vx+i*nTime, Tlen, MpiType, 0, tag, MPI_COMM_WORLD);
		}
		*/
		
		//full child procs temporal block
		if(HSpt[j].cmp==1 || HSpt[j].cmp==3)
		{
			//                         snap          VT
			tag = Dtag + myid +(1E3)*((SmallFlag+j) + 1);
			MPI_Send( HSW[j].Vx, nTime*CSpn[j], MpiType, 0, tag, MPI_COMM_WORLD);

			tag = Dtag + myid +(1E3)*((SmallFlag+j) + 2);
			MPI_Send( HSW[j].Vy, nTime*CSpn[j], MpiType, 0, tag, MPI_COMM_WORLD);

			tag = Dtag + myid +(1E3)*((SmallFlag+j) + 3);
			MPI_Send( HSW[j].Vz, nTime*CSpn[j], MpiType, 0, tag, MPI_COMM_WORLD);

		}
			
		if(HSpt[j].cmp==2 || HSpt[j].cmp==3)
		{
			tag = Dtag + myid +(1E3)*((SmallFlag+j) + 4);
			MPI_Send( HSW[j].Txx, nTime*CSpn[j], MpiType, 0, tag, MPI_COMM_WORLD);

			tag = Dtag + myid +(1E3)*((SmallFlag+j) + 5);
			MPI_Send( HSW[j].Tyy, nTime*CSpn[j], MpiType, 0, tag, MPI_COMM_WORLD);

			tag = Dtag + myid +(1E3)*((SmallFlag+j) + 6);
			MPI_Send( HSW[j].Tzz, nTime*CSpn[j], MpiType, 0, tag, MPI_COMM_WORLD);

			tag = Dtag + myid +(1E3)*((SmallFlag+j) + 7);
			MPI_Send( HSW[j].Txy, nTime*CSpn[j], MpiType, 0, tag, MPI_COMM_WORLD);

			tag = Dtag + myid +(1E3)*((SmallFlag+j) + 8);
			MPI_Send( HSW[j].Txz, nTime*CSpn[j], MpiType, 0, tag, MPI_COMM_WORLD);

			tag = Dtag + myid +(1E3)*((SmallFlag+j) + 9);
			MPI_Send( HSW[j].Tyz, nTime*CSpn[j], MpiType, 0, tag, MPI_COMM_WORLD);
		}
		
	}

}
void mpiSWRecv(wfield *MSW, int cpn, int currT, int nt, int nsnap, int *Snp, SnapIndexBuffer *HSpt, MPI_Status status)
{
	int i,j;
	int Tlen,nTime;
	int size;
	int tag;
	int Dtag;
	int SmallFlag=50;
	Dtag = 1E7;
	
	for(j=0;j<nsnap;j++)
	{
		Tlen = ceil(1.0*currT/HSpt[j].tinv);
		nTime = ceil(1.0*nt/HSpt[j].tinv);
		
		for(i=0;i<cpn;i++)
		{
			size = Snp[i*nsnap+j]*nTime;

			if(HSpt[j].cmp==1 || HSpt[j].cmp==3)
			{
				tag = Dtag + i+1 + (1E3)*((SmallFlag+j) +1);
				MPI_Recv( MSW[j].Vx+HSpt[j].Gsn[i][0]*nTime, size, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

				tag = Dtag + i+1 + (1E3)*((SmallFlag+j) +2);
				MPI_Recv( MSW[j].Vy+HSpt[j].Gsn[i][0]*nTime, size, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

				tag = Dtag + i+1 + (1E3)*((SmallFlag+j) +3);
				MPI_Recv( MSW[j].Vz+HSpt[j].Gsn[i][0]*nTime, size, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
			}
			
			if(HSpt[j].cmp==2 || HSpt[j].cmp==3)
			{
				tag = Dtag + i+1 + (1E3)*((SmallFlag+j) +4);
				MPI_Recv( MSW[j].Txx+HSpt[j].Gsn[i][0]*nTime, size, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

				tag = Dtag + i+1 + (1E3)*((SmallFlag+j) +5);
				MPI_Recv( MSW[j].Tyy+HSpt[j].Gsn[i][0]*nTime, size, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

				tag = Dtag + i+1 + (1E3)*((SmallFlag+j) +6);
				MPI_Recv( MSW[j].Tzz+HSpt[j].Gsn[i][0]*nTime, size, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

				tag = Dtag + i+1 + (1E3)*((SmallFlag+j) +7);
				MPI_Recv( MSW[j].Txy+HSpt[j].Gsn[i][0]*nTime, size, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

				tag = Dtag + i+1 + (1E3)*((SmallFlag+j) +8);
				MPI_Recv( MSW[j].Txz+HSpt[j].Gsn[i][0]*nTime, size, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

				tag = Dtag + i+1 + (1E3)*((SmallFlag+j) +9);
				MPI_Recv( MSW[j].Tyz+HSpt[j].Gsn[i][0]*nTime, size, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
			}
			
		}
	}
}
//transfer snap Par
void mpiSendSP(SnapIndexBuffer *MHSpt, int nsnap, int cpn, int *Snp)
{//1E6
	int i,j;
	int tag;
	int Dtag;
	Dtag = 1E6; 

	for(j=0;j<nsnap;j++)
		for(i=0;i<cpn;i++)
		{
			tag = Dtag + (i+1)*nsnap+j + (1E3)*1;
			MPI_Send(MHSpt[j].Rsn[i], Snp[i*nsnap+j], MPI_INT, i+1, tag, MPI_COMM_WORLD); 

			tag = Dtag + (i+1)*nsnap+j + (1E3)*2;
			MPI_Send(MHSpt[j].Gsn[i], Snp[i*nsnap+j], MPI_INT, i+1, tag, MPI_COMM_WORLD); 

			tag = Dtag + (i+1)*nsnap+j + (1E3)*3;
			MPI_Send(MHSpt[j].locx[i], Snp[i*nsnap+j], MPI_INT, i+1, tag, MPI_COMM_WORLD); 

			tag = Dtag + (i+1)*nsnap+j + (1E3)*4;
			MPI_Send(MHSpt[j].locy[i], Snp[i*nsnap+j], MPI_INT, i+1, tag, MPI_COMM_WORLD); 

			tag = Dtag + (i+1)*nsnap+j + (1E3)*5;
			MPI_Send(MHSpt[j].locz[i], Snp[i*nsnap+j], MPI_INT, i+1, tag, MPI_COMM_WORLD); 
			
			tag = Dtag + (i+1)*nsnap+j + (1E3)*6;
			MPI_Send(&MHSpt[j].tinv, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD); 
			
			tag = Dtag + (i+1)*nsnap+j + (1E3)*7;
			MPI_Send(&MHSpt[j].cmp, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD); 
		}

}
void mpiRecvSP(SnapIndexBufferF *CHSpt, int nsnap, int myid, int *CSpn, MPI_Status status)
{//1E6
	int i;
	int tag;
	int Dtag;
	Dtag = 1E6; 
	
	for(i=0;i<nsnap;i++)
	{
		tag = Dtag + myid*nsnap+i + (1E3)*1;
		MPI_Recv(CHSpt[i].Rsn, CSpn[i], MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + myid*nsnap+i + (1E3)*2;
		MPI_Recv(CHSpt[i].Gsn, CSpn[i], MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + myid*nsnap+i + (1E3)*3;
		MPI_Recv(CHSpt[i].locx, CSpn[i], MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + myid*nsnap+i + (1E3)*4;
		MPI_Recv(CHSpt[i].locy, CSpn[i], MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + myid*nsnap+i + (1E3)*5;
		MPI_Recv(CHSpt[i].locz, CSpn[i], MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
		
		tag = Dtag + myid*nsnap+i + (1E3)*6;
		MPI_Recv(&CHSpt[i].tinv, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
		
		tag = Dtag + myid*nsnap+i + (1E3)*7;
		MPI_Recv(&CHSpt[i].cmp, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	}
}

//------------------------------------------PP transfer--------------------------------------------------
void mpiPWSend(wfield HPW, int myid, int currT, int nt, int ppn, PointIndexBufferF Hpt)
{
	int i;
	int tag;
	int Dtag;
	int SmallFlag=20;
	Dtag = 9E6;
	for(i=0;i<ppn;i++)
	{
		tag = Dtag + myid +(1E3)*(SmallFlag+1) + Hpt.Gsn[i];
		MPI_Send( HPW.Vx+i*nt, currT, MpiType, 0, tag, MPI_COMM_WORLD);

		tag = Dtag + myid +(1E3)*(SmallFlag+2) + Hpt.Gsn[i];
		MPI_Send( HPW.Vy+i*nt, currT, MpiType, 0, tag, MPI_COMM_WORLD);

		tag = Dtag + myid +(1E3)*(SmallFlag+3) + Hpt.Gsn[i];
		MPI_Send( HPW.Vz+i*nt, currT, MpiType, 0, tag, MPI_COMM_WORLD);

		tag = Dtag + myid +(1E3)*(SmallFlag+4) + Hpt.Gsn[i];
		MPI_Send( HPW.Txx+i*nt, currT, MpiType, 0, tag, MPI_COMM_WORLD);

		tag = Dtag + myid +(1E3)*(SmallFlag+5) + Hpt.Gsn[i];
		MPI_Send( HPW.Tyy+i*nt, currT, MpiType, 0, tag, MPI_COMM_WORLD);

		tag = Dtag + myid +(1E3)*(SmallFlag+6) + Hpt.Gsn[i];
		MPI_Send( HPW.Tzz+i*nt, currT, MpiType, 0, tag, MPI_COMM_WORLD);

		tag = Dtag + myid +(1E3)*(SmallFlag+7) + Hpt.Gsn[i];
		MPI_Send( HPW.Txy+i*nt, currT, MpiType, 0, tag, MPI_COMM_WORLD);

		tag = Dtag + myid +(1E3)*(SmallFlag+8) + Hpt.Gsn[i];
		MPI_Send( HPW.Txz+i*nt, currT, MpiType, 0, tag, MPI_COMM_WORLD);

		tag = Dtag + myid +(1E3)*(SmallFlag+9) + Hpt.Gsn[i];
		MPI_Send( HPW.Tyz+i*nt, currT, MpiType, 0, tag, MPI_COMM_WORLD);

	}
}
void mpiPWRecv(wfield MPW, int cpn, int currT, int nt, int *np, PointIndexBuffer Hpt, MPI_Status status)
{
	int i,j;
	int tag;
	int Dtag;
	int SmallFlag=20;
	Dtag = 9E6;
	for(i=0;i<cpn;i++)
	{
		for(j=0;j<np[i];j++)
		{
			tag = Dtag + i+1 + (1E3)*(SmallFlag+1) + Hpt.Gsn[i][j];
			MPI_Recv( MPW.Vx+Hpt.Gsn[i][j]*nt, currT, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

			tag = Dtag + i+1 + (1E3)*(SmallFlag+2) + Hpt.Gsn[i][j];
			MPI_Recv( MPW.Vy+Hpt.Gsn[i][j]*nt, currT, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

			tag = Dtag + i+1 + (1E3)*(SmallFlag+3) + Hpt.Gsn[i][j];
			MPI_Recv( MPW.Vz+Hpt.Gsn[i][j]*nt, currT, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

			tag = Dtag + i+1 + (1E3)*(SmallFlag+4) + Hpt.Gsn[i][j];
			MPI_Recv( MPW.Txx+Hpt.Gsn[i][j]*nt, currT, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

			tag = Dtag + i+1 + (1E3)*(SmallFlag+5) + Hpt.Gsn[i][j];
			MPI_Recv( MPW.Tyy+Hpt.Gsn[i][j]*nt, currT, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

			tag = Dtag + i+1 + (1E3)*(SmallFlag+6) + Hpt.Gsn[i][j];
			MPI_Recv( MPW.Tzz+Hpt.Gsn[i][j]*nt, currT, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

			tag = Dtag + i+1 + (1E3)*(SmallFlag+7) + Hpt.Gsn[i][j];
			MPI_Recv( MPW.Txy+Hpt.Gsn[i][j]*nt, currT, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

			tag = Dtag + i+1 + (1E3)*(SmallFlag+8) + Hpt.Gsn[i][j];
			MPI_Recv( MPW.Txz+Hpt.Gsn[i][j]*nt, currT, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

			tag = Dtag + i+1 + (1E3)*(SmallFlag+9) + Hpt.Gsn[i][j];
			MPI_Recv( MPW.Tyz+Hpt.Gsn[i][j]*nt, currT, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		}
	}
}
//transfer Point Par
void mpiSendPP(PointIndexBuffer *MHpt, int cpn, int *np)
{//9E6
	int i;
	int tag;
	int Dtag;
	Dtag = 9E6; 
	
	for(i=0;i<cpn;i++)
	{
		tag = Dtag + i+1 + (1E3)*1;
		MPI_Send(MHpt->Rsn[i], np[i], MPI_INT, i+1, tag, MPI_COMM_WORLD); 
	
		tag = Dtag + i+1 + (1E3)*2;
		MPI_Send(MHpt->Gsn[i], np[i], MPI_INT, i+1, tag, MPI_COMM_WORLD); 
	
		tag = Dtag + i+1 + (1E3)*3;
		MPI_Send(MHpt->locx[i], np[i], MPI_INT, i+1, tag, MPI_COMM_WORLD); 
	
		tag = Dtag + i+1 + (1E3)*4;
		MPI_Send(MHpt->locy[i], np[i], MPI_INT, i+1, tag, MPI_COMM_WORLD); 
	
		tag = Dtag + i+1 + (1E3)*5;
		MPI_Send(MHpt->locz[i], np[i], MPI_INT, i+1, tag, MPI_COMM_WORLD); 
	}

}
void mpiRecvPP(PointIndexBufferF *CHpt, int myid, int ppn, MPI_Status status)
{//9E6
	int i;
	int tag;
	int Dtag;
	Dtag = 9E6; 
	
	tag = Dtag + myid + (1E3)*1;
	MPI_Recv(CHpt->Rsn, ppn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid + (1E3)*2;
	MPI_Recv(CHpt->Gsn, ppn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid + (1E3)*3;
	MPI_Recv(CHpt->locx, ppn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid + (1E3)*4;
	MPI_Recv(CHpt->locy, ppn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid + (1E3)*5;
	MPI_Recv(CHpt->locz, ppn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
}

//transfer Focus Par
void mpiSendFP(FocalIndexBuffer *MFpt, int cpn, int *fp)
{//9E6
	int i;
	int tag;
	int Dtag;
	Dtag = 9E6; 
	
	for(i=0;i<cpn;i++)
	{
		tag = Dtag + i+1 + (1E3)*51;
		MPI_Send(MFpt->Rsn[i], fp[i], MPI_INT, i+1, tag, MPI_COMM_WORLD); 
	
		tag = Dtag + i+1 + (1E3)*52;
		MPI_Send(MFpt->Gsn[i], fp[i], MPI_INT, i+1, tag, MPI_COMM_WORLD); 
	
		tag = Dtag + i+1 + (1E3)*53;
		MPI_Send(MFpt->locx[i], fp[i], MPI_INT, i+1, tag, MPI_COMM_WORLD); 
	
		tag = Dtag + i+1 + (1E3)*54;
		MPI_Send(MFpt->locy[i], fp[i], MPI_INT, i+1, tag, MPI_COMM_WORLD); 
	
		tag = Dtag + i+1 + (1E3)*55;
		MPI_Send(MFpt->locz[i], fp[i], MPI_INT, i+1, tag, MPI_COMM_WORLD); 
	}

}
void mpiRecvFP(FocalIndexBufferF *CHFpt, int myid, int fpn, MPI_Status status)
{//9E6
	int i;
	int tag;
	int Dtag;
	Dtag = 9E6; 
	
	tag = Dtag + myid + (1E3)*51;
	MPI_Recv(CHFpt->Rsn, fpn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid + (1E3)*52;
	MPI_Recv(CHFpt->Gsn, fpn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid + (1E3)*53;
	MPI_Recv(CHFpt->locx, fpn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid + (1E3)*54;
	MPI_Recv(CHFpt->locy, fpn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid + (1E3)*55;
	MPI_Recv(CHFpt->locz, fpn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
}
//transfer Focal data
void mpiFDSend(Rmom MFD, int cpn, int *fp , int NT, FocalIndexBuffer Fpt)
{
	//repackage MFD to CFD and flatten
	int i,j,k,m;
	int fpn,size;
	int Gindex,Src,Dst;

	int **locx,**locy,**locz;//[cp,fp]
	Real **Mxx,**Myy,**Mzz,**Mxy,**Mxz,**Myz;//[cp, fp*nt]
	
	//allocation
#ifdef SrcSmooth
	Real **Dn,*MDn;
	Dn = new Real*[cpn];
	MDn = new Real [MFD.np*LenNorm*LenNorm*LenNorm]();
	for(i=0;i<cpn;i++)
	{
		fpn = fp[i];	size = fpn*LenNorm*LenNorm*LenNorm;
		Dn[i] = new Real[size]();
	}
	for(i=0;i<MFD.np;i++)
		for(j=0;j<LenNorm;j++)
			for(k=0;k<LenNorm;k++)
				for(m=0;m<LenNorm;m++)
					MDn[i*LenNorm*LenNorm*LenNorm + j*LenNorm*LenNorm + k*LenNorm + m] = MFD.dnorm[i][j][k][m];
#endif

	locx = new int*[cpn]; locy = new int*[cpn]; locz = new int*[cpn];
	Mxx  = new Real*[cpn]; Myy  = new Real*[cpn]; Mzz  = new Real*[cpn];
	Mxy  = new Real*[cpn]; Mxz  = new Real*[cpn]; Myz  = new Real*[cpn];
	for(i=0;i<cpn;i++)
	{
		fpn = fp[i];	size = fpn*NT;
		locx[i] = new int[fpn](); locy[i] = new int[fpn](); locz[i] = new int[fpn]();
		Mxx[i] = new Real[size](); Myy[i] = new Real[size](); Mzz[i] = new Real[size]();
		Mxy[i] = new Real[size](); Mxz[i] = new Real[size](); Myz[i] = new Real[size]();
	}
	
	//distribution
	for(i=0;i<cpn;i++)
	{
		fpn = fp[i];
		for(j=0;j<fpn;j++)
		{
			Gindex = Fpt.Gsn[i][j];
			
			locx[i][j] = MFD.locx[Gindex];
			locy[i][j] = MFD.locy[Gindex];
			locz[i][j] = MFD.locz[Gindex];
			
			Dst = j*NT;
			Src = 0;
			memcpy( Mxx[i]+Dst, MFD.mxx[Gindex]+Src, NT*sizeof(Real) );
			//memcpy( &Mxx[i][Dst], &MFD.mxx[Gindex][0], NT*sizeof(Real) );
			memcpy( Myy[i]+Dst, MFD.myy[Gindex]+Src, NT*sizeof(Real) );
			memcpy( Mzz[i]+Dst, MFD.mzz[Gindex]+Src, NT*sizeof(Real) );
			memcpy( Mxy[i]+Dst, MFD.mxy[Gindex]+Src, NT*sizeof(Real) );
			memcpy( Mxz[i]+Dst, MFD.mxz[Gindex]+Src, NT*sizeof(Real) );
			memcpy( Myz[i]+Dst, MFD.myz[Gindex]+Src, NT*sizeof(Real) );

#ifdef SrcSmooth
			size = LenNorm*LenNorm*LenNorm;
			Dst = j*size;
			Src = Gindex*size;
			memcpy( Dn[i]+Dst, MDn+Src, size*sizeof(Real) );
#endif

		}
	}
	
	//transfer
	int tag;
	int Dtag;
	int SmallFlag=20;//(1-10)
	Dtag = 5E6;
	
	for(i=0;i<cpn;i++)
	{
		fpn = fp[i];
		size = fpn*NT;

		tag = Dtag + i+1 + (1E3)*(SmallFlag+1);
		MPI_Send( locx[i], fpn, MPI_INT, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+2);
		MPI_Send( locy[i], fpn, MPI_INT, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+3);
		MPI_Send( locz[i], fpn, MPI_INT, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+4);
		MPI_Send( Mxx[i], size, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+5);
		MPI_Send( Myy[i], size, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+6);
		MPI_Send( Mzz[i], size, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+7);
		MPI_Send( Mxy[i], size, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+8);
		MPI_Send( Mxz[i], size, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+9);
		MPI_Send( Myz[i], size, MpiType, i+1, tag, MPI_COMM_WORLD);

#ifdef SrcSmooth
		size = fpn*LenNorm*LenNorm*LenNorm;
		tag = Dtag + i+1 + (1E3)*(SmallFlag+10);
		MPI_Send( Dn[i], size, MpiType, i+1, tag, MPI_COMM_WORLD);
#endif

	}

	for(i=0;i<cpn;i++)
	{
		delete [] Myz[i]; delete [] Mxz[i]; delete [] Mxy[i];
		delete [] Mzz[i]; delete [] Myy[i]; delete [] Mxx[i];
		delete [] locz[i]; delete [] locy[i]; delete [] locx[i];
	}
	delete [] Myz; delete [] Mxz; delete [] Mxy;
	delete [] Mzz; delete [] Myy; delete [] Mxx;
	delete [] locz; delete [] locy; delete [] locx;
#ifdef SrcSmooth
	for(i=0;i<cpn;i++) 
		delete [] Dn[i];	
	delete [] Dn;
	delete [] MDn;
#endif

}
void mpiFDRecv(RmomF CFD, int myid, int fpn, int NT, FocalIndexBufferF HFpt, MPI_Status status)
{
	int i;
	int size;
	int tag;
	int Dtag;
	int SmallFlag=20;//1-10
	Dtag = 5E6;

	size = fpn*NT;

	tag = Dtag + myid +(1E3)*(SmallFlag+1);
	MPI_Recv(CFD.locx, fpn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*(SmallFlag+2);
	MPI_Recv(CFD.locy, fpn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*(SmallFlag+3);
	MPI_Recv(CFD.locz, fpn, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*(SmallFlag+4);
	MPI_Recv(CFD.mxx, size, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*(SmallFlag+5);
	MPI_Recv(CFD.myy, size, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*(SmallFlag+6);
	MPI_Recv(CFD.mzz, size, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*(SmallFlag+7);
	MPI_Recv(CFD.mxy, size, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*(SmallFlag+8);
	MPI_Recv(CFD.mxz, size, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*(SmallFlag+9);
	MPI_Recv(CFD.myz, size, MpiType, 0, tag, MPI_COMM_WORLD, &status);

#ifdef SrcSmooth
	size = fpn*LenNorm*LenNorm*LenNorm;
	tag = Dtag + myid +(1E3)*(SmallFlag+10);
	MPI_Recv(CFD.dnorm, size, MpiType, 0, tag, MPI_COMM_WORLD, &status);
#endif
	
}

//transfer topo peak velocity
void mpiPVSend(PeakVel CP, int myid, int Cxn, cindx cdx)
{//7E6
	int tag;
	int Syz;
	int Dtag;
	int SmallFlag=50;//1-3
	Dtag = 7E6;
	Syz = cdx.ny;

	tag = Dtag + myid + (1E3)*(SmallFlag+1);
	MPI_Send(CP.Vx+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD);

	tag = Dtag + myid + (1E3)*(SmallFlag+2);
	MPI_Send(CP.Vy+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD);

	tag = Dtag + myid + (1E3)*(SmallFlag+3);
	MPI_Send(CP.Vz+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD);

}
void mpiPVRecv(PeakVel MP, int cpn, cindx cdx, M2Csplit Mid, MPI_Status status)
{//7E6
	int i;
	int tag;
	int Syz;
	int Dtag;
	int SmallFlag=50;//1-3
	Dtag = 7E6;
	Syz = cdx.ny;
	
	for(i=0;i<cpn;i++)
	{
		tag = Dtag + i+1 + (1E3)*(SmallFlag+1);
		MPI_Recv(MP.Vx+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	
		tag = Dtag + i+1 + (1E3)*(SmallFlag+2);
		MPI_Recv(MP.Vy+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	
		tag = Dtag + i+1 + (1E3)*(SmallFlag+3);
		MPI_Recv(MP.Vz+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	}

}

//-------------------------------------------Data transfer----------------------------------------------
void mpiDataGatherSend(wfield Cfullwave, int myid, int Cxn, cindx cdx)
{
	int tag;
	tag = 7E6;
	mpisend(Cfullwave, tag, myid, Cxn, cdx);

}
void mpiDataGatherRecv(wfield Mfullwave, int cpn, cindx cdx, M2Csplit Mid, MPI_Status status)
{
	int tag;
	tag = 7E6;
	mpirecv(Mfullwave, tag, cpn, cdx, Mid, status);

}

//transfer wave field data
void mpisend(wfield CF, int Dtag, int myid, int Cxn, cindx cdx)
{//7E6
	int tag;
	int Syz;
	Syz = cdx.ny*cdx.nz;
	
	tag = Dtag + myid + (1E3)*1;
	MPI_Send(CF.Txx+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD);

	tag = Dtag + myid + (1E3)*2;
	MPI_Send(CF.Tyy+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD);

	tag = Dtag + myid + (1E3)*3;
	MPI_Send(CF.Tzz+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD);

	tag = Dtag + myid + (1E3)*4;
	MPI_Send(CF.Txy+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD);

	tag = Dtag + myid + (1E3)*5;
	MPI_Send(CF.Txz+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD);

	tag = Dtag + myid + (1E3)*6;
	MPI_Send(CF.Tyz+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD);

	tag = Dtag + myid + (1E3)*7;
	MPI_Send(CF.Vx+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD);

	tag = Dtag + myid + (1E3)*8;
	MPI_Send(CF.Vy+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD);

	tag = Dtag + myid + (1E3)*9;
	MPI_Send(CF.Vz+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD);

}
void mpirecv(wfield MF, int Dtag, int cpn, cindx cdx, M2Csplit Mid, MPI_Status status)
{//7E6
	int i;
	int tag;
	int Syz;
	Syz = cdx.ny*cdx.nz;
	for(i=0;i<cpn;i++)
	{
		tag = Dtag + i+1 + (1E3)*1;
		MPI_Recv(MF.Txx+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	
		tag = Dtag + i+1 + (1E3)*2;
		MPI_Recv(MF.Tyy+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	
		tag = Dtag + i+1 + (1E3)*3;
		MPI_Recv(MF.Tzz+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	
		tag = Dtag + i+1 + (1E3)*4;
		MPI_Recv(MF.Txy+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	
		tag = Dtag + i+1 + (1E3)*5;
		MPI_Recv(MF.Txz+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	
		tag = Dtag + i+1 + (1E3)*6;
		MPI_Recv(MF.Tyz+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	
		tag = Dtag + i+1 + (1E3)*7;
		MPI_Recv(MF.Vx+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	
		tag = Dtag + i+1 + (1E3)*8;
		MPI_Recv(MF.Vy+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	
		tag = Dtag + i+1 + (1E3)*9;
		MPI_Recv(MF.Vz+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	}

}

void RestartSend(wfield MF, int Mtime, int Dtag, int cpn, cindx cdx, M2Csplit Mid)
{//7E6   SmallFlag20
 //6E6   2
	int i;
	int tag,SmallFlag;
	int Syz;
	Syz = cdx.ny*cdx.nz;
	SmallFlag = 20;
	for(i=0;i<cpn;i++)
	{
		//send restart wave
		tag = Dtag + i+1 + (1E3)*(SmallFlag+1);
		MPI_Send(MF.Txx+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+2);
		MPI_Send(MF.Tyy+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+3);
		MPI_Send(MF.Tzz+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+4);
		MPI_Send(MF.Txy+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+5);
		MPI_Send(MF.Txz+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+6);
		MPI_Send(MF.Tyz+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+7);
		MPI_Send(MF.Vx+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+8);
		MPI_Send(MF.Vy+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+9);
		MPI_Send(MF.Vz+Mid.Xstart[i]*Syz, Mid.Xsize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);
		
		//send time
		tag = 6E6 + 1 + (1E3)*2;
		MPI_Send(&Mtime, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
	}


}
void RestartRecv(wfield CF, int *Ctime, int Dtag, int myid, int Cxn, cindx cdx, MPI_Status status)
{//7E6   SmallFlag20
 //6E6   2
	int tag,SmallFlag;
	int Syz;
	SmallFlag=20;
	Syz = cdx.ny*cdx.nz;

	//recv restart wave
	tag = Dtag + myid + (1E3)*(SmallFlag+1);
	MPI_Recv(CF.Txx+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+2);
	MPI_Recv(CF.Tyy+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+3);
	MPI_Recv(CF.Tzz+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+4);
	MPI_Recv(CF.Txy+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+5);
	MPI_Recv(CF.Txz+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+6);
	MPI_Recv(CF.Tyz+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+7);
	MPI_Recv(CF.Vx+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+8);
	MPI_Recv(CF.Vy+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+9);
	MPI_Recv(CF.Vz+LenFD*Syz, Cxn*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	//recv time
	tag = 6E6 + 1 + (1E3)*2;
	MPI_Recv(Ctime, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

}


//---------------------------------------boundary transfer---------------------------------------------
//transfer wave field boundary data, After a full RK step
void BoundSend(wfield MF, cindx cdx, M2Csplit Mid, int cpn, int SmallFlag)
{//8E6
	int i;
	int tag,Dtag;
	int Syz;
	Dtag = 8E6;
	Syz = cdx.ny*cdx.nz;
	for(i=0;i<cpn;i++)
	{
		tag = Dtag + i+1 + (1E3)*(SmallFlag+1);
		MPI_Send(MF.Txx + Mid.CopyS[i]*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//head sending
		
		tag = Dtag + i+1 + (1E3)*(SmallFlag+2);
		MPI_Send(MF.Txx + (Mid.Xend[i]+1)*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//tail sending

		tag = Dtag + i+1 + (1E3)*(SmallFlag+3);
		MPI_Send(MF.Tyy + Mid.CopyS[i]*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//head sending
		
		tag = Dtag + i+1 + (1E3)*(SmallFlag+4);
		MPI_Send(MF.Tyy + (Mid.Xend[i]+1)*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//tail sending

		tag = Dtag + i+1 + (1E3)*(SmallFlag+5);
		MPI_Send(MF.Tzz + Mid.CopyS[i]*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//head sending
		
		tag = Dtag + i+1 + (1E3)*(SmallFlag+6);
		MPI_Send(MF.Tzz + (Mid.Xend[i]+1)*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//tail sending

		tag = Dtag + i+1 + (1E3)*(SmallFlag+7);
		MPI_Send(MF.Txy + Mid.CopyS[i]*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//head sending
		
		tag = Dtag + i+1 + (1E3)*(SmallFlag+8);
		MPI_Send(MF.Txy + (Mid.Xend[i]+1)*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//tail sending

		tag = Dtag + i+1 + (1E3)*(SmallFlag+9);
		MPI_Send(MF.Txz + Mid.CopyS[i]*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//head sending
		
		tag = Dtag + i+1 + (1E3)*(SmallFlag+10);
		MPI_Send(MF.Txz + (Mid.Xend[i]+1)*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//tail sending

		tag = Dtag + i+1 + (1E3)*(SmallFlag+11);
		MPI_Send(MF.Tyz + Mid.CopyS[i]*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//head sending
		
		tag = Dtag + i+1 + (1E3)*(SmallFlag+12);
		MPI_Send(MF.Tyz + (Mid.Xend[i]+1)*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//tail sending

		tag = Dtag + i+1 + (1E3)*(SmallFlag+13);
		MPI_Send(MF.Vx + Mid.CopyS[i]*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//head sending
		
		tag = Dtag + i+1 + (1E3)*(SmallFlag+14);
		MPI_Send(MF.Vx + (Mid.Xend[i]+1)*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//tail sending

		tag = Dtag + i+1 + (1E3)*(SmallFlag+15);
		MPI_Send(MF.Vy + Mid.CopyS[i]*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//head sending
		
		tag = Dtag + i+1 + (1E3)*(SmallFlag+16);
		MPI_Send(MF.Vy + (Mid.Xend[i]+1)*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//tail sending

		tag = Dtag + i+1 + (1E3)*(SmallFlag+17);
		MPI_Send(MF.Vz + Mid.CopyS[i]*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//head sending
		
		tag = Dtag + i+1 + (1E3)*(SmallFlag+18);
		MPI_Send(MF.Vz + (Mid.Xend[i]+1)*Syz, LenFD*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);//tail sending

	}

}
void BoundRecv(wfield CF, cindx cdx, int Cxn, int myid, int SmallFlag, MPI_Status status)
{//8E6
	int tag,Dtag;
	int Syz;
	Dtag = 8E6;
	Syz = cdx.ny*cdx.nz;
	
	tag = Dtag + myid + (1E3)*(SmallFlag+1);
	MPI_Recv(CF.Txx, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//head recving

	tag = Dtag + myid + (1E3)*(SmallFlag+2);
	MPI_Recv(CF.Txx + (LenFD+Cxn)*Syz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//tail recving

	tag = Dtag + myid + (1E3)*(SmallFlag+3);
	MPI_Recv(CF.Tyy, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//head recving

	tag = Dtag + myid + (1E3)*(SmallFlag+4);
	MPI_Recv(CF.Tyy + (LenFD+Cxn)*Syz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//tail recving

	tag = Dtag + myid + (1E3)*(SmallFlag+5);
	MPI_Recv(CF.Tzz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//head recving

	tag = Dtag + myid + (1E3)*(SmallFlag+6);
	MPI_Recv(CF.Tzz + (LenFD+Cxn)*Syz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//tail recving

	tag = Dtag + myid + (1E3)*(SmallFlag+7);
	MPI_Recv(CF.Txy, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//head recving

	tag = Dtag + myid + (1E3)*(SmallFlag+8);
	MPI_Recv(CF.Txy + (LenFD+Cxn)*Syz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//tail recving

	tag = Dtag + myid + (1E3)*(SmallFlag+9);
	MPI_Recv(CF.Txz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//head recving

	tag = Dtag + myid + (1E3)*(SmallFlag+10);
	MPI_Recv(CF.Txz + (LenFD+Cxn)*Syz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//tail recving

	tag = Dtag + myid + (1E3)*(SmallFlag+11);
	MPI_Recv(CF.Tyz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//head recving

	tag = Dtag + myid + (1E3)*(SmallFlag+12);
	MPI_Recv(CF.Tyz + (LenFD+Cxn)*Syz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//tail recving

	tag = Dtag + myid + (1E3)*(SmallFlag+13);
	MPI_Recv(CF.Vx, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//head recving

	tag = Dtag + myid + (1E3)*(SmallFlag+14);
	MPI_Recv(CF.Vx + (LenFD+Cxn)*Syz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//tail recving

	tag = Dtag + myid + (1E3)*(SmallFlag+15);
	MPI_Recv(CF.Vy, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//head recving

	tag = Dtag + myid + (1E3)*(SmallFlag+16);
	MPI_Recv(CF.Vy + (LenFD+Cxn)*Syz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//tail recving

	tag = Dtag + myid + (1E3)*(SmallFlag+17);
	MPI_Recv(CF.Vz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//head recving

	tag = Dtag + myid + (1E3)*(SmallFlag+18);
	MPI_Recv(CF.Vz + (LenFD+Cxn)*Syz, LenFD*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);//tail recving

}

//Bounds interaction main program
void BoundGS(wfield W, cindx cdx, int cpn, int myid, int cxn, MPI_Status status)
{
	int SmallFlag;
	int Syz;
	int BCsize;
	Syz = cdx.ny*cdx.nz;
	BCsize = LenFD*Syz;

	//gather from Child procs to Master procs
	SmallFlag = 20;
	if(myid)
	{
		if(cxn)
			BoundGatherSend(W, Syz, cxn, myid, SmallFlag);//send from inner
		else
		{
			//printf("at PCS[%d], node-level boundary send in only bound data type\n",myid);
			BoundGatherSendOB(W, Syz, cxn, myid, SmallFlag);//only boundary data send
		}
	}
	else
	{
		BoundGatherRecv(W, Syz, cpn, SmallFlag, status);
		//BoundExtend(W, BCsize, cpn);wrong, should be mirror extend; orginal fortran version has none
	}
	//MPI_Barrier(MPI_COMM_WORLD);

	
	//Scatter from Master procs to Child procs
	SmallFlag = 40;
	if(!myid)
		BoundScatterSend(W, Syz, cpn, SmallFlag);
	else
		BoundScatterRecv(W, Syz, cxn, myid, SmallFlag, status);
	
}
void BoundGatherSendOB(wfield W, int Syz, int cxn, int myid, int SmallFlag)
{//8E6   |   SmallFlag=20

	int tag,Dtag;
	int BCsize;
	BCsize = LenFD*Syz;
	Dtag = 8E6;

	tag = Dtag + myid + (1E3)*(SmallFlag+1);
	MPI_Send(W.Txx, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+2);
	MPI_Send(W.Txx+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+3);
	MPI_Send(W.Tyy, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+4);
	MPI_Send(W.Tyy+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+5);
	MPI_Send(W.Tzz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+6);
	MPI_Send(W.Tzz+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+7);
	MPI_Send(W.Txy, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+8);
	MPI_Send(W.Txy+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+9);
	MPI_Send(W.Txz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+10);
	MPI_Send(W.Txz+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+11);
	MPI_Send(W.Tyz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+12);
	MPI_Send(W.Tyz+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+13);
	MPI_Send(W.Vx, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+14);
	MPI_Send(W.Vx+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+15);
	MPI_Send(W.Vy, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+16);
	MPI_Send(W.Vy+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+17);
	MPI_Send(W.Vz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+18);
	MPI_Send(W.Vz+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

}
void BoundGatherSend(wfield W, int Syz, int cxn, int myid, int SmallFlag)
{//8E6   |   SmallFlag=20

	int tag,Dtag;
	int BCsize;
	BCsize = LenFD*Syz;
	Dtag = 8E6;

	tag = Dtag + myid + (1E3)*(SmallFlag+1);
	MPI_Send(W.Txx+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+2);
	MPI_Send(W.Txx+cxn*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+3);
	MPI_Send(W.Tyy+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+4);
	MPI_Send(W.Tyy+cxn*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+5);
	MPI_Send(W.Tzz+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+6);
	MPI_Send(W.Tzz+cxn*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+7);
	MPI_Send(W.Txy+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+8);
	MPI_Send(W.Txy+cxn*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+9);
	MPI_Send(W.Txz+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+10);
	MPI_Send(W.Txz+cxn*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+11);
	MPI_Send(W.Tyz+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+12);
	MPI_Send(W.Tyz+cxn*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+13);
	MPI_Send(W.Vx+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+14);
	MPI_Send(W.Vx+cxn*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+15);
	MPI_Send(W.Vy+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+16);
	MPI_Send(W.Vy+cxn*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+17);
	MPI_Send(W.Vz+BCsize, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//head GA sends

	tag = Dtag + myid + (1E3)*(SmallFlag+18);
	MPI_Send(W.Vz+cxn*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD);//tail GA sends

}
void BoundGatherRecv(wfield W, int Syz, int cpn, int SmallFlag, MPI_Status status)
{//8E6     |  SmallFlag=20
	int i;
	int tag,Dtag;
	int BCsize;
	Dtag = 8E6;
	BCsize = LenFD*Syz;
	for(i=0;i<cpn;i++)
	{
		//put
		// relative | absolute
		// (myid-1)*2+0	| (myid-1)*2+1  |  i*2+1
		// (myid-1)*2+1 | (myid-1)*2+2  |  i*2+2
		// myid=i+1 ---> myid-1=i
		// Here use absolute location, head and tail are useless
		
		tag = Dtag + i+1 + (1E3)*(SmallFlag+1);
		MPI_Recv(W.Txx+(i*2+1)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+2);
		MPI_Recv(W.Txx+(i*2+2)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+3);
		MPI_Recv(W.Tyy+(i*2+1)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+4);
		MPI_Recv(W.Tyy+(i*2+2)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+5);
		MPI_Recv(W.Tzz+(i*2+1)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+6);
		MPI_Recv(W.Tzz+(i*2+2)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+7);
		MPI_Recv(W.Txy+(i*2+1)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+8);
		MPI_Recv(W.Txy+(i*2+2)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+9);
		MPI_Recv(W.Txz+(i*2+1)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+10);
		MPI_Recv(W.Txz+(i*2+2)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+11);
		MPI_Recv(W.Tyz+(i*2+1)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+12);
		MPI_Recv(W.Tyz+(i*2+2)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+13);
		MPI_Recv(W.Vx+(i*2+1)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+14);
		MPI_Recv(W.Vx+(i*2+2)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+15);
		MPI_Recv(W.Vy+(i*2+1)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+16);
		MPI_Recv(W.Vy+(i*2+2)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+17);
		MPI_Recv(W.Vz+(i*2+1)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+18);
		MPI_Recv(W.Vz+(i*2+2)*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD, &status);
	
	}

}
void BoundScatterSend(wfield W, int Syz, int cpn, int SmallFlag)
{//8E6		|	SmallFlag=40
	int i;
	int tag,Dtag;
	int BCsize;
	Dtag = 8E6;
	BCsize = LenFD*Syz;
	for(i=0;i<cpn;i++)
	{
		//extract
		// relative | absolute
		// (myid-1)*2+1-1 | (myid-1)*2  |  i*2
		// (myid-1)*2+2+1 | myid*2+1    |  i*2+3
		// myid=i+1 ---> myid-1=i

		tag = Dtag + i+1 + (1E3)*(SmallFlag+1);
		MPI_Send(W.Txx+i*2*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+2);
		MPI_Send(W.Txx+( i*2+3 )*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+3);
		MPI_Send(W.Tyy+i*2*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+4);
		MPI_Send(W.Tyy+( i*2+3 )*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+5);
		MPI_Send(W.Tzz+i*2*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+6);
		MPI_Send(W.Tzz+( i*2+3 )*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+7);
		MPI_Send(W.Txy+i*2*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+8);
		MPI_Send(W.Txy+( i*2+3 )*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+9);
		MPI_Send(W.Txz+i*2*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+10);
		MPI_Send(W.Txz+( i*2+3 )*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+11);
		MPI_Send(W.Tyz+i*2*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+12);
		MPI_Send(W.Tyz+( i*2+3 )*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+13);
		MPI_Send(W.Vx+i*2*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+14);
		MPI_Send(W.Vx+( i*2+3 )*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+15);
		MPI_Send(W.Vy+i*2*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+16);
		MPI_Send(W.Vy+( i*2+3 )*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+17);
		MPI_Send(W.Vz+i*2*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*(SmallFlag+18);
		MPI_Send(W.Vz+( i*2+3 )*BCsize, BCsize, MpiType, i+1, tag, MPI_COMM_WORLD);

	}

}
void BoundScatterRecv(wfield W, int Syz, int cxn, int myid, int SmallFlag, MPI_Status status)
{//8E6 		| 	SmallFlag=40
	int tag,Dtag;
	int BCsize;
	BCsize = LenFD*Syz;
	Dtag = 8E6;
	tag = Dtag + myid + (1E3)*(SmallFlag+1);
	MPI_Recv(W.Txx, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+2);
	MPI_Recv(W.Txx+(LenFD+cxn)*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+3);
	MPI_Recv(W.Tyy, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+4);
	MPI_Recv(W.Tyy+(LenFD+cxn)*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+5);
	MPI_Recv(W.Tzz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+6);
	MPI_Recv(W.Tzz+(LenFD+cxn)*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+7);
	MPI_Recv(W.Txy, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+8);
	MPI_Recv(W.Txy+(LenFD+cxn)*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+9);
	MPI_Recv(W.Txz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+10);
	MPI_Recv(W.Txz+(LenFD+cxn)*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+11);
	MPI_Recv(W.Tyz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+12);
	MPI_Recv(W.Tyz+(LenFD+cxn)*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+13);
	MPI_Recv(W.Vx, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+14);
	MPI_Recv(W.Vx+(LenFD+cxn)*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+15);
	MPI_Recv(W.Vy, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+16);
	MPI_Recv(W.Vy+(LenFD+cxn)*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+17);
	MPI_Recv(W.Vz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*(SmallFlag+18);
	MPI_Recv(W.Vz+(LenFD+cxn)*Syz, BCsize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

}
void BoundExtend(wfield W, int BCsize, int cpn)
{
	memcpy(W.Txx, W.Txx+BCsize, BCsize*sizeof(Real));
	memcpy(W.Txx+(cpn*2+1)*BCsize, W.Txx+cpn*2*BCsize, BCsize*sizeof(Real));

	memcpy(W.Tyy, W.Tyy+BCsize, BCsize*sizeof(Real));
	memcpy(W.Tyy+(cpn*2+1)*BCsize, W.Tyy+cpn*2*BCsize, BCsize*sizeof(Real));

	memcpy(W.Tzz, W.Tzz+BCsize, BCsize*sizeof(Real));
	memcpy(W.Tzz+(cpn*2+1)*BCsize, W.Tzz+cpn*2*BCsize, BCsize*sizeof(Real));

	memcpy(W.Txy, W.Txy+BCsize, BCsize*sizeof(Real));
	memcpy(W.Txy+(cpn*2+1)*BCsize, W.Txy+cpn*2*BCsize, BCsize*sizeof(Real));

	memcpy(W.Txz, W.Txz+BCsize, BCsize*sizeof(Real));
	memcpy(W.Txz+(cpn*2+1)*BCsize, W.Txz+cpn*2*BCsize, BCsize*sizeof(Real));

	memcpy(W.Tyz, W.Tyz+BCsize, BCsize*sizeof(Real));
	memcpy(W.Tyz+(cpn*2+1)*BCsize, W.Tyz+cpn*2*BCsize, BCsize*sizeof(Real));

	memcpy(W.Vx, W.Vx+BCsize, BCsize*sizeof(Real));
	memcpy(W.Vx+(cpn*2+1)*BCsize, W.Vx+cpn*2*BCsize, BCsize*sizeof(Real));

	memcpy(W.Vy, W.Vy+BCsize, BCsize*sizeof(Real));
	memcpy(W.Vy+(cpn*2+1)*BCsize, W.Vy+cpn*2*BCsize, BCsize*sizeof(Real));

	memcpy(W.Vz, W.Vz+BCsize, BCsize*sizeof(Real));
	memcpy(W.Vz+(cpn*2+1)*BCsize, W.Vz+cpn*2*BCsize, BCsize*sizeof(Real));

}

//--------------------------------------------Par transfer---------------------------------------------
void mpiSendPars(deriv *Mdrv, mdpar *Mmpa, force *Mfrc, moment *Mmnt, 
		int cpn, int nfrc, int nmnt, int nstf, cindx cdx, M2Csplit Mid)
{
	int tag;
	tag = 2E6;
	mpisend(Mdrv, tag, cpn, cdx, Mid);
	
	tag = 3E6;
	mpisend(Mmpa, tag, cpn, cdx, Mid);
	
	if(nfrc)
	{
		tag = 4E6;
		mpisend(Mfrc, tag, cpn, nfrc, nstf);
	}

	if(nmnt)
	{
		tag = 5E6;
		mpisend(Mmnt, tag, cpn, nmnt, nstf);
	}

}
void mpiRecvPars(derivF *Cdrv, mdparF *Cmpa, forceF *Cfrc, momentF *Cmnt, 
		int myid, int Csize, int nfrc, int nmnt, int nstf, cindx cdx, MPI_Status status) 
{
	int tag,Dtag;
	tag = 2E6;
	mpirecv(Cdrv, tag, myid, Csize, cdx, status);
	
	tag = 3E6;
	mpirecv(Cmpa, tag, myid, Csize, cdx, status);

	if(nfrc)
	{
		tag = 4E6;
		mpirecv(Cfrc, tag, myid, nfrc, nstf, status);
	}

	if(nmnt)
	{
		tag = 5E6;
		mpirecv(Cmnt, tag, myid, nmnt, nstf, status);
	}
	
}

//transfer derivative data
void mpisend(deriv *Mdrv, int Dtag, int cpn, cindx cdx, M2Csplit Mid)
{//2E6
	common com;
	int tag;
	int i;
	int Syz;//Size of Y*Z
	Real *h1d;
	h1d = new Real[cdx.nx*cdx.ny*cdx.nz];
	Syz = cdx.ny*cdx.nz;
	
	for(i=0;i<cpn;i++)
	{
		tag = Dtag + i+1 + (1E3)*1;
		com.flatten21D(Mdrv->xi_x, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*2;
		com.flatten21D(Mdrv->xi_y, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*3;
		com.flatten21D(Mdrv->xi_z, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 + (1E3)*4;
		com.flatten21D(Mdrv->eta_x, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*5;
		com.flatten21D(Mdrv->eta_y, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*6;
		com.flatten21D(Mdrv->eta_z, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*7;
		com.flatten21D(Mdrv->zeta_x, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*8;
		com.flatten21D(Mdrv->zeta_y, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*9;
		com.flatten21D(Mdrv->zeta_z, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*10;
		com.flatten21D(Mdrv->jac, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);
	}

	delete [] h1d;
}
void mpirecv(derivF *Cdrv, int Dtag, int myid, int Csize, cindx cdx, MPI_Status status)
{//2E6
	int tag;
	int Syz;

	Syz = cdx.ny*cdx.nz;
	
	tag = Dtag + myid + (1E3)*1;
	MPI_Recv(Cdrv->xix, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*2;
	MPI_Recv(Cdrv->xiy, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*3;
	MPI_Recv(Cdrv->xiz, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*4;
	MPI_Recv(Cdrv->etax, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*5;
	MPI_Recv(Cdrv->etay, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*6;
	MPI_Recv(Cdrv->etaz, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*7;
	MPI_Recv(Cdrv->zetax, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*8;
	MPI_Recv(Cdrv->zetay, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*9;
	MPI_Recv(Cdrv->zetaz, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*10;
	MPI_Recv(Cdrv->jac, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

}

//transfer media parameters and ConIndex
void mpisend(mdpar *Mmpa, int Dtag, int cpn, cindx cdx, M2Csplit Mid)
{//3E6
	common com;
	int tag;
	int i;
	int Syz;//Size of Y*Z
	Real *h1d;
	h1d = new Real[cdx.nx*cdx.ny*cdx.nz];
	Syz = cdx.ny*cdx.nz;
	
	for(i=0;i<cpn;i++)
	{
		tag = Dtag + i+1 + (1E3)*1;
		com.flatten21D(Mmpa->alpha, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*2;
		com.flatten21D(Mmpa->beta, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 + (1E3)*3;
		com.flatten21D(Mmpa->rho, h1d, cdx, 0);//full size flatten
		MPI_Send(h1d+Mid.CopyS[i]*Syz, Mid.CopySize[i]*Syz, MpiType, i+1, tag, MPI_COMM_WORLD);
	}

	delete [] h1d;
}
void mpirecv(mdparF *Cmpa, int Dtag, int myid, int Csize, cindx cdx, MPI_Status status)
{//3E6
	int tag;
	int Syz;

	Syz = cdx.ny*cdx.nz;
	
	tag = Dtag + myid + (1E3)*1;
	MPI_Recv(Cmpa->alpha, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*2;
	MPI_Recv(Cmpa->beta, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*3;
	MPI_Recv(Cmpa->rho, Csize*Syz, MpiType, 0, tag, MPI_COMM_WORLD, &status);
}

//transfer force data
void mpisend(force *Mfrc, int Dtag, int cpn, int nfrc, int nstf)
{//4E6
	int tag;
	int i,j;
	int S1;//Size of source time function
	Real *h1d;
	h1d = new Real[nfrc*nstf];
	S1 = nfrc*nstf;

	for(i=0;i<nfrc;i++)
		for(j=0;j<nstf;j++)
			h1d[i*nstf+j] = Mfrc->stf[i][j];

#ifdef SrcSmooth
	int k,m;
	Real *h2d;
	int S2;

	h2d = new Real [nfrc*LenNorm*LenNorm*LenNorm];
	S2 = nfrc*LenNorm*LenNorm*LenNorm;
	
	for(i=0;i<nfrc;i++)
		for(j=0;j<LenNorm;j++)
			for(k=0;k<LenNorm;k++)
				for(m=0;m<LenNorm;m++)
					h2d[i*LenNorm*LenNorm*LenNorm + j*LenNorm*LenNorm + k*LenNorm + m] = Mfrc->dnorm[i][j][k][m];
#endif

	for(i=0;i<cpn;i++)
	{
		tag = Dtag + i+1 + (1E3)*1;
		MPI_Send(Mfrc->locx, nfrc, MPI_INT, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*2;
		MPI_Send(Mfrc->locy, nfrc, MPI_INT, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*3;
		MPI_Send(Mfrc->locz, nfrc, MPI_INT, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*4;
		MPI_Send(Mfrc->fx, nfrc, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*5;
		MPI_Send(Mfrc->fy, nfrc, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*6;
		MPI_Send(Mfrc->fz, nfrc, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*7;
		MPI_Send(h1d, S1, MpiType, i+1, tag, MPI_COMM_WORLD);

#ifdef SrcSmooth
		tag = Dtag + i+1 + (1E3)*8;
		MPI_Send(h2d, S2, MpiType, i+1, tag, MPI_COMM_WORLD);
#endif
	}

#ifdef SrcSmooth
	delete [] h2d;
#endif
	delete [] h1d;
}
void mpirecv(forceF *Cfrc, int Dtag, int myid, int nfrc, int nstf, MPI_Status status)
{//4E6
	int tag;
	int S1;
	S1 = nfrc*nstf;

#ifdef SrcSmooth
	int S2 = nfrc*LenNorm*LenNorm*LenNorm;
#endif
	
	tag = Dtag + myid + (1E3)*1;
	MPI_Recv(Cfrc->locx, nfrc, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*2;
	MPI_Recv(Cfrc->locy, nfrc, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*3;
	MPI_Recv(Cfrc->locz, nfrc, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*4;
	MPI_Recv(Cfrc->fx, nfrc, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*5;
	MPI_Recv(Cfrc->fy, nfrc, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*6;
	MPI_Recv(Cfrc->fz, nfrc, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*7;
	MPI_Recv(Cfrc->stf, S1, MpiType, 0, tag, MPI_COMM_WORLD, &status);

#ifdef SrcSmooth
	tag = Dtag + myid + (1E3)*8;
	MPI_Recv(Cfrc->dnorm, S2, MpiType, 0, tag, MPI_COMM_WORLD, &status);
#endif

}

//transfer moment data
void mpisend(moment *Mmnt, int Dtag, int cpn, int nmnt, int nstf)
{//5E6
	int tag;
	int i,j;
	int S1;//Size of source time function
	Real *h1d;
	h1d = new Real[nmnt*nstf];
	S1 = nmnt*nstf;

	for(i=0;i<nmnt;i++)
		for(j=0;j<nstf;j++)
			h1d[i*nstf+j] = Mmnt->stf[i][j];

#ifdef SrcSmooth
	int k,m;
	Real *h2d;
	int S2;

	h2d = new Real [nmnt*LenNorm*LenNorm*LenNorm];
	S2 = nmnt*LenNorm*LenNorm*LenNorm;
	
	for(i=0;i<nmnt;i++)
		for(j=0;j<LenNorm;j++)
			for(k=0;k<LenNorm;k++)
				for(m=0;m<LenNorm;m++)
					h2d[i*LenNorm*LenNorm*LenNorm + j*LenNorm*LenNorm + k*LenNorm + m] = Mmnt->dnorm[i][j][k][m];
#endif
	
	for(i=0;i<cpn;i++)
	{
		tag = Dtag + i+1 + (1E3)*1;
		MPI_Send(Mmnt->locx, nmnt, MPI_INT, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*2;
		MPI_Send(Mmnt->locy, nmnt, MPI_INT, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*3;
		MPI_Send(Mmnt->locz, nmnt, MPI_INT, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*4;
		MPI_Send(Mmnt->mxx, nmnt, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*5;
		MPI_Send(Mmnt->myy, nmnt, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*6;
		MPI_Send(Mmnt->mzz, nmnt, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*7;
		MPI_Send(Mmnt->mxy, nmnt, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*8;
		MPI_Send(Mmnt->mxz, nmnt, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*9;
		MPI_Send(Mmnt->myz, nmnt, MpiType, i+1, tag, MPI_COMM_WORLD);
	
		tag = Dtag + i+1 + (1E3)*10;
		MPI_Send(h1d, S1, MpiType, i+1, tag, MPI_COMM_WORLD);
	
#ifdef SrcSmooth
		tag = Dtag + i+1 + (1E3)*11;
		MPI_Send(h2d, S2, MpiType, i+1, tag, MPI_COMM_WORLD);
#endif
	}

#ifdef SrcSmooth
	delete [] h2d;
#endif
	delete [] h1d;
}
void mpirecv(momentF *Cmnt, int Dtag, int myid, int nmnt, int nstf, MPI_Status status)
{//5E6
	int tag;
	int S1;
	S1 = nmnt*nstf;

#ifdef SrcSmooth
	int S2 = nmnt*LenNorm*LenNorm*LenNorm;
#endif
	
	tag = Dtag + myid + (1E3)*1;
	MPI_Recv(Cmnt->locx, nmnt, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*2;
	MPI_Recv(Cmnt->locy, nmnt, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*3;
	MPI_Recv(Cmnt->locz, nmnt, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*4;
	MPI_Recv(Cmnt->mxx, nmnt, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*5;
	MPI_Recv(Cmnt->myy, nmnt, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*6;
	MPI_Recv(Cmnt->mzz, nmnt, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*7;
	MPI_Recv(Cmnt->mxy, nmnt, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*8;
	MPI_Recv(Cmnt->mxz, nmnt, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*9;
	MPI_Recv(Cmnt->myz, nmnt, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*10;
	MPI_Recv(Cmnt->stf, S1, MpiType, 0, tag, MPI_COMM_WORLD, &status);

#ifdef SrcSmooth
	tag = Dtag + myid + (1E3)*11;
	MPI_Recv(Cmnt->dnorm, S2, MpiType, 0, tag, MPI_COMM_WORLD, &status);
#endif

}

//--------------------------------------------Par transfer---------------------------------------------
void mpiSendABC(apara *Mapr, int cpn, cindx cdx, M2Csplit Mid)
{
	int tag,Dtag;
	int i;
	tag = 6E6;
	//Dtag = 6E6;
#ifdef CFSPML
	mpisend(Mapr->nabs, Mapr->APDx, Mapr->APDy, Mapr->APDz, Mapr->Bx, Mapr->By, Mapr->Bz, 
		Mapr->DBx, Mapr->DBy, Mapr->DBz, Mapr->CLoc, tag, cpn, cdx, Mid);
#else	
	mpisend(Mapr->nabs, Mapr->Ex, Mapr->Ey, Mapr->Ez, Mapr->ELoc, tag, cpn, cdx, Mid);
#endif	
}
void mpiRecvABC(apara *Capr, int myid, int Csize, cindx cdx, MPI_Status status) 
{
	int tag,Dtag;
	tag = 6E6;
	//Dtag = 6E6;
#ifdef CFSPML
	mpirecv(Capr->nabs, Capr->APDx, Capr->APDy, Capr->APDz, Capr->Bx, Capr->By, Capr->Bz, 
		Capr->DBx, Capr->DBy, Capr->DBz, Capr->CLoc, tag, myid, Csize, cdx, status);
#else	
	mpirecv(Capr->nabs, Capr->Ex, Capr->Ey, Capr->Ez, Capr->ELoc, tag, myid, Csize, cdx, status);
#endif
}

void mpiSendABC(int *Mnabs, Real *MEx, Real *MEy, Real *MEz, int *MAbsLoc, int cpn, cindx cdx, M2Csplit Mid)
{//deprecated
	int tag;
	tag = 6E6;
	mpisend(Mnabs, MEx, MEy, MEz, MAbsLoc, tag, cpn, cdx, Mid);
}
void mpiRecvABC(int *Cnabs, Real *CEx, Real *CEy, Real *CEz, int *CAbsLoc, int myid, int Csize, cindx cdx, MPI_Status status) 
{//deprecated
	int tag;
	tag = 6E6;
	mpirecv(Cnabs, CEx, CEy, CEz, CAbsLoc, tag, myid, Csize, cdx, status);
}

//transfer AbsExp pars;
void mpisend(int *Mnabs, Real *MEx, Real *MEy, Real *MEz, int *MAbsLoc, int Dtag, int cpn, cindx cdx, M2Csplit Mid)
{//6E6,3-7
	int tag;
	int i;

	for(i=0;i<cpn;i++)
	{
		tag = Dtag + i+1 + (1E3)*3;
		MPI_Send(Mnabs, SeisGeo*2, MPI_INT, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*4;
		MPI_Send(MEx+Mid.CopyS[i], Mid.CopySize[i], MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*5;
		MPI_Send(MEy, cdx.ny, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*6;
		MPI_Send(MEz, cdx.nz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*7;
		MPI_Send(MAbsLoc, SeisGeo*2*SeisGeo*2, MPI_INT, i+1, tag, MPI_COMM_WORLD);
	}

}
void mpirecv(int *Cnabs, Real *CEx, Real *CEy, Real *CEz, int *CAbsLoc, int Dtag, int myid, int Csize, cindx cdx, MPI_Status status)
{//6E6,3-7
	int tag;

	tag = Dtag + myid + (1E3)*3;
	MPI_Recv(Cnabs, SeisGeo*2, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*4;
	MPI_Recv(CEx, Csize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*5;
	MPI_Recv(CEy, cdx.ny, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*6;
	MPI_Recv(CEz, cdx.nz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*7;
	MPI_Recv(CAbsLoc, SeisGeo*2*SeisGeo*2, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

}

void mpiSendABC(int *Mnabs, Real *MAx, Real *MAy, Real *MAz, Real *MBx, Real *MBy, Real *MBz, Real *MDx, Real *MDy, Real *MDz, 
		int *MAbsLoc, int cpn, cindx cdx, M2Csplit Mid)
{//deprecated
	int tag;
	tag = 6E6;
	mpisend(Mnabs, MAx, MAy, MAz, MBx, MBy, MBz, MDx, MDy, MDz, MAbsLoc, tag, cpn, cdx, Mid);
}
void mpiRecvABC(int *Cnabs, Real *CAx, Real *CAy, Real *CAz, Real *CBx, Real *CBy, Real *CBz, Real *CDx, Real *CDy, Real *CDz, 
		int *CAbsLoc, int myid, int Csize, cindx cdx, MPI_Status status) 
{//deprecated
	int tag;
	tag = 6E6;
	mpirecv(Cnabs, CAx, CAy, CAz, CBx, CBy, CBz, CDx, CDy, CDz, CAbsLoc, tag, myid, Csize, cdx, status);
}

//transfer AbsPML pars;
void mpisend(int *Mnabs, Real *MAx, Real *MAy, Real *MAz, Real *MBx, Real *MBy, Real *MBz, Real *MDx, Real *MDy, Real *MDz,
	     int *MAbsLoc, int Dtag, int cpn, cindx cdx, M2Csplit Mid)
{//6E6,8-18
	int tag;
	int i;

	for(i=0;i<cpn;i++)
	{
		tag = Dtag + i+1 + (1E3)*8;
		MPI_Send(Mnabs, SeisGeo*2, MPI_INT, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*9;
		MPI_Send(MAx+Mid.CopyS[i], Mid.CopySize[i], MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*10;
		MPI_Send(MAy, cdx.ny, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*11;
		MPI_Send(MAz, cdx.nz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*12;
		MPI_Send(MBx+Mid.CopyS[i], Mid.CopySize[i], MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*13;
		MPI_Send(MBy, cdx.ny, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*14;
		MPI_Send(MBz, cdx.nz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*15;
		MPI_Send(MDx+Mid.CopyS[i], Mid.CopySize[i], MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*16;
		MPI_Send(MDy, cdx.ny, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*17;
		MPI_Send(MDz, cdx.nz, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 + (1E3)*18;
		MPI_Send(MAbsLoc, 26*SeisGeo*2, MPI_INT, i+1, tag, MPI_COMM_WORLD);

	}

}
void mpirecv(int *Cnabs, Real *CAx, Real *CAy, Real *CAz, Real *CBx, Real *CBy, Real *CBz, Real *CDx, Real *CDy, Real *CDz,
	     int *CAbsLoc, int Dtag, int myid, int Csize, cindx cdx, MPI_Status status)
{//6E6,8-18
	int tag;

	tag = Dtag + myid + (1E3)*8;
	MPI_Recv(Cnabs, SeisGeo*2, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*9;
	MPI_Recv(CAx, Csize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*10;
	MPI_Recv(CAy, cdx.ny, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*11;
	MPI_Recv(CAz, cdx.nz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*12;
	MPI_Recv(CBx, Csize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*13;
	MPI_Recv(CBy, cdx.ny, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*14;
	MPI_Recv(CBz, cdx.nz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*15;
	MPI_Recv(CDx, Csize, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*16;
	MPI_Recv(CDy, cdx.ny, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*17;
	MPI_Recv(CDz, cdx.nz, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid + (1E3)*18;
	MPI_Recv(CAbsLoc, 26*SeisGeo*2, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

}

//transfer index parameters;
void mpisend(cindx *cdx, int cpn, int Dtag)
{//1E6
	//cpn=child procs number=procsNum-1
	int tag;
	int i;
	
	for(i=0;i<cpn;i++)
	{
		//tag = type 	procs	 variable
		tag = Dtag + i+1 +(1E3)*1;
		MPI_Send(&cdx->nx, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*2;
		MPI_Send(&cdx->ny, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*3;
		MPI_Send(&cdx->nz, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*4;
		MPI_Send(&cdx->ni, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*5;
		MPI_Send(&cdx->nj, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*6;
		MPI_Send(&cdx->nk, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*7;
		MPI_Send(&cdx->nx1, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*8;
		MPI_Send(&cdx->nx2, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*9;
		MPI_Send(&cdx->ny1, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*10;
		MPI_Send(&cdx->ny2, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*11;
		MPI_Send(&cdx->nz1, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*12;
		MPI_Send(&cdx->nz2, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*13;
		MPI_Send(&cdx->ni1, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*14;
		MPI_Send(&cdx->ni2, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*15;
		MPI_Send(&cdx->nj1, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*16;
		MPI_Send(&cdx->nj2, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*17;
		MPI_Send(&cdx->nk1, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*18;
		MPI_Send(&cdx->nk2, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
	}
}
void mpirecv(cindx *cdx, int myid, int Dtag, MPI_Status status)
{//1E6
	int tag;
	
	tag = Dtag + myid +(1E3)*1;
	MPI_Recv(&cdx->nx, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*2;
	MPI_Recv(&cdx->ny, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*3;
	MPI_Recv(&cdx->nz, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*4;
	MPI_Recv(&cdx->ni, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*5;
	MPI_Recv(&cdx->nj, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*6;
	MPI_Recv(&cdx->nk, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*7;
	MPI_Recv(&cdx->nx1, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*8;
	MPI_Recv(&cdx->nx2, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*9;
	MPI_Recv(&cdx->ny1, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*10;
	MPI_Recv(&cdx->ny2, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*11;
	MPI_Recv(&cdx->nz1, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*12;
	MPI_Recv(&cdx->nz2, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*13;
	MPI_Recv(&cdx->ni1, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*14;
	MPI_Recv(&cdx->ni2, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*15;
	MPI_Recv(&cdx->nj1, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*16;
	MPI_Recv(&cdx->nj2, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*17;
	MPI_Recv(&cdx->nk1, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*18;
	MPI_Recv(&cdx->nk2, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

}

//transfer CP initialization parameters
void mpisend(int nt, Real steph, Real stept, int nfrc, int nmnt, int nstf, int restart, int ConIndex, int HyGrid, int *nabs, int *np, int *Snp,
	     int *fp, int fnt, Real fdt, int nsnap, int pvflag, int cpn, int Dtag)
{//1E6
	//cpn=child procs number=procsNum-1
	int tag;//continue with cdx vars
	int i;
	
	for(i=0;i<cpn;i++)
	{
		//tag = type 	procs	 variable
		tag = Dtag + i+1 +(1E3)*20;
		MPI_Send(&nfrc, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*21;
		MPI_Send(&nmnt, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 +(1E3)*22;
		MPI_Send(&nstf, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 +(1E3)*23;
		MPI_Send(&restart, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 +(1E3)*24;
		MPI_Send(&steph, 1, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 +(1E3)*25;
		MPI_Send(&stept, 1, MpiType, i+1, tag, MPI_COMM_WORLD);

		tag = Dtag + i+1 +(1E3)*26;
		MPI_Send(&nt, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 + (1E3)*27;///sepcial for conindex
		MPI_Send(&ConIndex, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 + (1E3)*28;///sepcial for HyGrid
		MPI_Send(&HyGrid, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 + (1E3)*29;
		MPI_Send(nabs, 6, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 + (1E3)*30;//special for piont number
		MPI_Send(np+i, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 + (1E3)*31;//special for Snap piont number
		MPI_Send(Snp+i*nsnap, nsnap, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 + (1E3)*32;//special for focal point number
		MPI_Send(fp+i, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 + (1E3)*33;//special for focal time number
		MPI_Send(&fnt, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 + (1E3)*34;//special for focal time interval
		MPI_Send(&fdt, 1, MpiType, i+1, tag, MPI_COMM_WORLD);
		
		tag = Dtag + i+1 + (1E3)*35;//special for peak vel flag
		MPI_Send(&pvflag, 1, MPI_INT, i+1, tag, MPI_COMM_WORLD);
	}
}
void mpirecv(int *nt, Real *steph, Real *stept, int *nfrc, int *nmnt, int *nstf, int *restart, int *ConIndex, int *HyGrid, int *Mnabs, int *ppn, int *CSpn,
	     int *fpn, int *FNT, Real *FDT, int nsnap, int *pvflag, int myid, int Dtag, MPI_Status status)
{//1E6
	int tag;

	tag = Dtag + myid +(1E3)*20;
	MPI_Recv(nfrc, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*21;
	MPI_Recv(nmnt, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*22;
	MPI_Recv(nstf, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*23;
	MPI_Recv(restart, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*24;
	MPI_Recv(steph, 1, MpiType, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*25;
	MPI_Recv(stept, 1, MpiType, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*26;
	MPI_Recv(nt, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	tag = Dtag + myid +(1E3)*27;//special for ConIndex
	MPI_Recv(ConIndex, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*28;//special for HyGrid
	MPI_Recv(HyGrid, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*29;
	MPI_Recv(Mnabs, 6, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*30;
	MPI_Recv(ppn, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*31;
	MPI_Recv(CSpn, nsnap, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*32;
	MPI_Recv(fpn, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*33;
	MPI_Recv(FNT, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*34;
	MPI_Recv(FDT, 1, MpiType, 0, tag, MPI_COMM_WORLD, &status);
	
	tag = Dtag + myid +(1E3)*35;
	MPI_Recv(pvflag, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
}


//-------------------------------------------index compute--------------------------------------------
void dataalloc(M2Csplit *Mid, int cp)
{
	Mid->CPNum  = new int [cp];
	Mid->np  = new int [cp];//receiver number
	Mid->fp  = new int [cp]();//focal number
	Mid->Xstart = new int [cp];
	Mid->Xend = new int [cp];
	Mid->Xsize = new int [cp];
	Mid->CopyS = new int [cp];
	Mid->CopyE = new int [cp];
	Mid->CopySize = new int [cp];
}
void datafree(M2Csplit *Mid)
{
	delete [] Mid->CPNum;
	delete [] Mid->np;
	delete [] Mid->fp;
	delete [] Mid->Xstart;
	delete [] Mid->Xend;
	delete [] Mid->Xsize;
	delete [] Mid->CopyS;
	delete [] Mid->CopyE;
	delete [] Mid->CopySize;
}
void splitM2C(const char *filename, const int myid, int cp, int nx, int *Csize, int *Cxn, int *Cstart, M2Csplit *Mid, MPI_Status status)
{
	common com;
	char parpath[SeisStrLen];
	char name[SeisStrLen2];
	char errstr[SeisStrLen];
	char devfile[SeisStrLen];
	FILE *fp;
	int i;
	if(myid)
	{
		fp = fopen(filename,"r");
		if(!fp)
		{
			sprintf(errstr,"Fail to open main par file %s in splitM2C", filename);
			errprt(Fail2Open,errstr);
		}
		com.get_conf(fp, "seispath", 3, parpath);
		com.get_conf(fp, "device_filename", 3, name);
		fclose(fp);
		
		sprintf(devfile,"%s/%s",parpath,name);

		char hostname[256];
		char str[256];
		int dn;

		fp = fopen(devfile,"r");

		gethostname(hostname,256);
		sprintf(str,"used_device_number_%s",hostname);
		com.get_conf(fp, str, 3, &dn);

		fclose(fp);

		MPI_Send(&dn, 1, MPI_INT, 0, 1000+myid, MPI_COMM_WORLD);
	}
	else
	{
		for(i=0;i<cp;i++)
			MPI_Recv(&Mid->CPNum[i], 1, MPI_INT, i+1, 1000+i+1, MPI_COMM_WORLD, &status);

		CPdataIdx(nx, cp, Mid->CPNum, Mid->Xstart, Mid->Xend, Mid->Xsize, Mid->CopyS, Mid->CopyE, Mid->CopySize);
		
		for(i=0;i<cp;i++)
		printf("for CP-%d, physical data(length %d) start from column[%d] and finish at column[%d],\n"
		       "	   virtual data(length %d) ranges from [%d] to [%d]\n",
		i+1,Mid->Xsize[i],Mid->Xstart[i],Mid->Xend[i],Mid->CopySize[i],Mid->CopyS[i],Mid->CopyE[i]);
	}

	if(!myid)
	{
		for(i=0;i<cp;i++)
		{
			MPI_Send(&Mid->CopySize[i], 1, MPI_INT, i+1, 2000+i+1, MPI_COMM_WORLD);
			MPI_Send(&Mid->Xsize[i], 1, MPI_INT, i+1, 3000+i+1, MPI_COMM_WORLD);
			MPI_Send(&Mid->CopyS[i], 1, MPI_INT, i+1, 4000+i+1, MPI_COMM_WORLD);
		}
	}
	else
	{
		MPI_Recv(Csize, 1, MPI_INT, 0, 2000+myid, MPI_COMM_WORLD, &status);
		MPI_Recv(Cxn, 1, MPI_INT, 0, 3000+myid, MPI_COMM_WORLD, &status);
		MPI_Recv(Cstart, 1, MPI_INT, 0, 4000+myid, MPI_COMM_WORLD, &status);
	}


}

void CPdataIdx(int nx, int numCP, int *devN, int *Xstart, int *Xend, int *size, int *Cs, int *Ce, int *Csize)
{
	if(numCP<1)
	{
		printf("There is no Child Procs(=%d)!!!!!\n",numCP);
		return;
	}

	int i,j;
	int totaldev;
	int tempacc;
	
	totaldev=0;
	
	for(i=0;i<numCP;i++)
		totaldev = totaldev+devN[i];
	if(numCP>1)
	{
		tempacc = 0;
		for(i=0;i<numCP-1;i++)
		{
			if( (nx*devN[i])%totaldev != 0 )
				size[i] = int(nx*devN[i]/totaldev) + 1;
			else
				size[i] = int(nx*devN[i]/totaldev);
			tempacc = tempacc + size[i];
		}
		size[numCP-1] = nx - tempacc;
	}
	else
		size[numCP-1] = nx;

	for(i=0;i<numCP;i++)
	{
		if(i!=0)
			Xstart[i] = Xend[i-1]+1;
		else
			Xstart[i] = LenFD;//cause use the total physical number, so add a shift to index
		Xend[i] = Xstart[i] + size[i]-1 ;
	}

	for(i=0;i<numCP;i++)
	{
		//from M to C copy, should add virtual boundary length of LenFD
		Cs[i] = Xstart[i] - LenFD;
		Ce[i] = Xend[i] + LenFD;
		Csize[i] = size[i] + 2*LenFD;
	}
}

int idxcom(int nx, int ny, int deviceNum, int Ydims, int *xl, int *xr, int *yd, int *yu)
{
	//start and finish index
	//BxxxxxxxxxxB 
	//y          y
	//y          y
	//y          y
	//y          y
	//BxxxxxxxxxxB
	//  start <=  INDEX  <= end
	
	int Xdims;
	int Xnorsize,Xremsize;
	int Ynorsize,Yremsize;
	int i,j;
	int idx;
	int temp;

	int verbose=0;//display (1) working processing.


	// index calculation
	Xdims = int(deviceNum/Ydims);


	if(deviceNum%Ydims != 0)
	{
		Xdims = Xdims + 1;
		temp = deviceNum%Ydims;//Ydims in last X column
		if(verbose) 
			printf("There are %d devices remained in last column, so adjust Xdims to %d\n",temp,Xdims);
	}
	else
		temp = Ydims;

	if(nx%Xdims != 0)
	{
		bimax(nx, 1.0*nx/Xdims, &Xnorsize);
		Xremsize = nx - Xnorsize*(Xdims-1);
	}
	else
	{
		Xnorsize = nx/Xdims;
		Xremsize = Xnorsize;
	}
	if(verbose) 
		printf("Xdims=%d,nx=%d,Xremain=%d, Xnormal=%d, Xremain=%d\n",Xdims,nx,nx%Xdims,Xnorsize,Xremsize);

	if(ny%Ydims != 0)
	{
		bimax(ny, 1.0*ny/Ydims,&Ynorsize);
		Yremsize = ny - Ynorsize*(Ydims-1);
	}
	else
	{
		Ynorsize = ny/Ydims;
		Yremsize = Ynorsize;
	}
	if(verbose) 
		printf("Ydims=%d,ny=%d,Yremain=%d, Ynormal=%d, Yremain=%d\n",Ydims,ny,ny%Ydims,Ynorsize,Yremsize);

	if(Ydims != 1)
	{
		for(i=0;i<Xdims-1;i++)
		{
			for(j=0;j<Ydims-1;j++)
			{
				//normal region
				idx = i*Ydims + j;

				xl[idx] = i*Xnorsize;
				xr[idx] = (i+1)*Xnorsize - 1;

				yd[idx] = j*Ynorsize;
				yu[idx] = (j+1)*Ynorsize - 1;

				if(verbose)
					printf("idx->%d: xl=%2d,xr=%2d,yd=%2d,yu=%2d\n",idx,xl[idx],xr[idx],yd[idx],yu[idx]);
			}
			//Y remain
			xl[idx+1] = i*Xnorsize;
			xr[idx+1] = (i+1)*Xnorsize - 1;
			yd[idx+1] = (Ydims-1)*Ynorsize;
			yu[idx+1] = yd[idx+1] + Yremsize -1;
			if(verbose)
				printf("idx->%d: xl=%2d,xr=%2d,yd=%2d,yu=%2d\n",idx+1,xl[idx+1],xr[idx+1],yd[idx+1],yu[idx+1]);

		}
	}
	else
	{
		for(i=0;i<Xdims-1;i++)
		{
			idx = i;
			xl[idx] = i*Xnorsize;
			xr[idx] = (i+1)*Xnorsize-1;
			yd[idx] = 0;
			yu[idx] = ny-1;
			if(verbose)
				printf("Ydims==1----idx->%d: xl=%2d,xr=%2d,yd=%2d,yu=%2d\n",idx,xl[idx],xr[idx],yd[idx],yu[idx]);
		}
	}


	//last X column
	//whether Y is full
	if(temp != Ydims)
	{
		if(ny%temp != 0)
		{
			bimax(ny, 1.0*ny/temp,&Ynorsize);
			Yremsize = ny - Ynorsize*(temp-1);
		}
		else
		{
			Ynorsize = ny/temp;
			Yremsize = Ynorsize;
		}
		if(verbose) 
			printf("Last Xcol: Remaindims=%d,ny=%d,Yflag=%d, Ynormal=%d, Yremain=%d\n",temp,ny,ny%Ydims,Ynorsize,Yremsize);
	}

	idx = (Xdims-1)*Ydims;
	if(temp != 1)
	{
		for(i=0;i<temp-1;i++)
		{
			idx = (Xdims-1)*Ydims + i;

			xl[idx] = (Xdims-1)*Xnorsize;
			xr[idx] = xl[idx] + Xremsize -1;

			yd[idx] = i*Ynorsize;
			yu[idx] = (i+1)*Ynorsize - 1;
			if(verbose)
				printf("temp!=1-----idx->%d: xl=%2d,xr=%2d,yd=%2d,yu=%2d\n",idx,xl[idx],xr[idx],yd[idx],yu[idx]);
		}
		idx = idx + 1;
		xl[idx] = (Xdims-1)*Xnorsize;
		xr[idx] = xl[idx] + Xremsize -1;
		yd[idx] = (temp-1)*Ynorsize;
		yu[idx] = yd[idx] + Yremsize -1;
		if(verbose)
			printf("temp!=1-----idx->%d: xl=%2d,xr=%2d,yd=%2d,yu=%2d\n",idx,xl[idx],xr[idx],yd[idx],yu[idx]);
	}
	else
	{
		idx = (Xdims-1)*Ydims;
		xl[idx] = (Xdims-1)*Xnorsize;
		xr[idx] = xl[idx] + Xremsize -1;
		yd[idx] = (temp-1)*Ynorsize;
		yu[idx] = yd[idx] + Yremsize -1;
		if(verbose)
			printf("temp==1----idx->%d: xl=%2d,xr=%2d,yd=%2d,yu=%2d\n",idx,xl[idx],xr[idx],yd[idx],yu[idx]);
	}

	for(i=0;i<Xdims*Ydims;i++)
	{
		xl[i] = xl[i] + LenFD;
		xr[i] = xr[i] + LenFD;
		yd[i] = yd[i] + LenFD;
		yu[i] = yu[i] + LenFD;
	}
	
	return Xdims;
}

void bimax(int limit, float A, int *B)
{
	int i;
	double left,right;
	float temp=A;

	i=0;
	do
	{
		i++;
		A = A/2;
	}while(A/2>1);

	left = pow(2,i);
	right = pow(2,i+1);
	
	if(right < limit)
		(temp - left) >= (right - temp) ? *B = (int)right : *B = (int)left;
	else
		*B = left;

}

void loadfixedarray(int *ipam, int C_id, int Cstart, int Vni, int ni, int nj, int nk, C2Dsplit Cid)
{
	int i;
	
	for(i=0;i<Cid.DNum;i++)
	{
		ipam[i*11+0] = Cid.Rank[i];//device rank
		ipam[i*11+1] = C_id;//child procs rank
		ipam[i*11+2] = Cid.xl[i];//xstart
		ipam[i*11+3] = Cid.xr[i];//xend
		ipam[i*11+4] = Cid.yd[i];//ystart
		ipam[i*11+5] = Cid.yu[i];//yend
		ipam[i*11+6] = Vni;//valid ni---node-size
		ipam[i*11+7] = nj;//nj
		ipam[i*11+8] = nk;//nk
		ipam[i*11+9] = Cstart;//Process X start
		ipam[i*11+10] = ni;//ni
		//from 6 to 10, in node perspective
		//from 2 to 5, in device perspective
	}
}

//---------------------------------------------Error function---------------------------------------
void MpiErrorPrint(const char *file, int line, int errnum, const char *errstr)
{
	if(errnum)//check for main procs
	{
		cout<<errstr<<endl;
		MPI_Abort(MPI_COMM_WORLD,errnum);
	}
}


//---------------------------------------------Timing function---------------------------------------
double Tsecond()
{
	struct timeval tp;
	int i = gettimeofday(&tp,NULL);
	return( (double)tp.tv_sec+(double)tp.tv_usec*1.e-6 );
}














