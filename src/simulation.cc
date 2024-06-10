#include "typedata.h"
#include<sys/stat.h> //mkdir
#include<sys/types.h>
#include<unistd.h> //access

#include<iomanip>

using namespace constant;
using namespace defstruct;
using namespace std;

#define errprt(...) com.errorprint(__FILE__,__LINE__,__VA_ARGS__)

common com;
mathfunc mathf;
void get_conf_file(int,char *[],int);
void get_conf();

int currT;
Real stept,steph;
int nt;//total time step
char seisfile[SeisStrLen],seisOutpath[SeisStrLen],seisInpath[SeisStrLen];
cindx cdx;
flatstruct::M2Csplit Mid;
int restart,pausetime;



int main(int argc, char *argv[])
{
	
	int myid,procsNum,Dtag;
	int ChildProcsNum;
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Comm_size(MPI_COMM_WORLD,&procsNum);
  	ChildProcsNum = procsNum-1;
	time_t Time;
	char *actime;
	if(!myid)
	{
		time(&Time);
		actime = asctime(localtime(&Time));
		fprintf(stdout,"***FDTD3D Program Starting at %s \n",actime);
	}
	currT = 0;
	get_conf_file(argc,argv,myid);

	Dtag = 1E6;
	if(!myid)
	{
		get_conf();
		mpisend(&cdx, ChildProcsNum, Dtag);
	}
	else
	{
		mpirecv(&cdx, myid, Dtag, status);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	int Csepsize,Cxlength,Cstart;
	Csepsize=0; Cxlength=0; Cstart=0;
	dataalloc(&Mid, ChildProcsNum);
	splitM2C(seisfile, myid, ChildProcsNum, cdx.ni, &Csepsize, &Cxlength, &Cstart, &Mid, status);
	MPI_Barrier(MPI_COMM_WORLD);

	gridmesh grd(seisfile, cdx, restart, myid);
	absorb abc(seisfile, cdx, restart, myid);
	mediapar mdp(seisfile, cdx, restart, myid);
	source src(seisfile, nt, restart, myid, ChildProcsNum);
	seisplot spt(seisfile, cdx, restart, myid, ChildProcsNum, nt);//temproal canceled because allocation
	MPI_Barrier(MPI_COMM_WORLD);

//return 0;
	if(!myid)
	{
		cout<<"\nstart to reading grid\n";
		grd.readdata(cdx, seisfile, steph);
		grd.calmetric(steph, cdx);
		grd.export_data(seisInpath, cdx);
		
		cout<<"\nstart to reading media\n";
		mdp.readdata(cdx, grd.crd);
		mdp.timecheck(stept, cdx, grd.crd);
		mdp.export_data(seisInpath, cdx, grd.crd);
		
		cout<<"\nstart to reading source\n";
		src.readdata(stept);
		src.cal_index(cdx, grd.crd, mdp.mpa);
		src.export_data(seisInpath, src.frc, src.mnt);
		//src.BoundJug(cdx, grd.crd, abc.nabs);
		src.M2CPointPick(cdx, Mid.Xstart, Mid.Xend, Mid.fp, ChildProcsNum);//only use for focus

		cout<<"\nstart to reading station\n";
		spt.readdata(seisOutpath);
		spt.locpoint(cdx, grd.crd);
		spt.M2CPointPick(cdx, Mid.Xstart, Mid.Xend, Mid.np, ChildProcsNum);
		spt.M2CSnapPick(cdx, Mid.Xstart, Mid.Xend, ChildProcsNum);
		spt.data_export_def(&currT, stept, cdx, &spt.pnt, spt.snp, &spt.wbuffer, spt.wsnap);
		
		cout<<"\nstart to compute absorb damping profiles\n";
		abc.CalCFSfactors(cdx, grd.drv, steph, stept);
		abc.CalSLDamping(cdx, grd.drv, steph, stept);
	}
	MPI_Barrier(MPI_COMM_WORLD);

//return 0;
	
	float i_Cfdt;
	i_Cfdt=0.0;
	int i_nfrc,i_nmnt,i_nstf,i_ConIndex,i_HyGrid,i_Cppn, i_Cfpn,i_Cfnt, i_pvflag;
	i_nfrc=0; i_nmnt=0; i_nstf=0; i_ConIndex=cdx.nk1; i_HyGrid=0; i_Cppn=0; i_Cfpn=0; i_Cfnt=0; i_pvflag=0;
	int *i_nabs,*i_CSpn;
	i_nabs = new int[6]();	i_CSpn = new int[spt.nsnap]();
	//transfer public parameters
	Dtag = 1E6;
	if(!myid)
	{
		i_nfrc = src.nfrc; i_nmnt = src.nmnt; i_nstf = src.nstf; i_ConIndex=grd.ConIndex; i_HyGrid=grd.HyGrid;
		//i_ConIndex=cp.cdx.nk2;  
		//i_ConIndex=cp.cdx.nk1;  
		//CI == nk2 ---> all SL
		//CI == nk1 ---> all CL
		//i_ConIndex = cdx.nk1;
		//i_HyGrid = 0;
		mpisend(nt, steph, stept, i_nfrc, i_nmnt, i_nstf, restart, i_ConIndex, i_HyGrid, abc.nabs, Mid.np, spt.Snp, 
			Mid.fp, src.Rmnt.nt, src.Rmnt.dt, spt.nsnap, spt.PVflag, ChildProcsNum, Dtag);
	}
	else
	{
		mpirecv(&nt, &steph, &stept, &i_nfrc, &i_nmnt, &i_nstf, &restart, &i_ConIndex, &i_HyGrid, i_nabs, &i_Cppn, i_CSpn, 
			&i_Cfpn, &i_Cfnt, &i_Cfdt, spt.nsnap, &i_pvflag, myid, Dtag, status);
	}
	MPI_Barrier(MPI_COMM_WORLD);

//return 0;

	ChildProcs cp(seisfile, cdx, steph, stept, i_nfrc, i_nmnt, i_nstf, Csepsize, Cxlength, Cstart, 
		      i_ConIndex, i_HyGrid, i_nabs, i_Cppn, nt, i_CSpn, spt.nsnap, i_Cfpn, i_Cfnt, i_Cfdt, i_pvflag,
		      restart, myid, ChildProcsNum);
	
	//transfer compute needs data
	if(!myid)
	{
		mpiSendPars(&grd.drv, &mdp.mpa, &src.frc, &src.mnt, ChildProcsNum, src.nfrc, src.nmnt, src.nstf, cdx, Mid);
#ifdef withABS	
		mpiSendABC(&abc.apr, ChildProcsNum, cdx, Mid);
#endif		
		mpiSendPP(&spt.Hpt, ChildProcsNum, Mid.np);
		mpiSendSP(spt.HSpt, spt.nsnap, ChildProcsNum, spt.Snp);
		
		if(src.Rmnt.np) mpiSendFP(&src.Fpt, ChildProcsNum, Mid.fp);
	}
	else
	{
		mpiRecvPars(&cp.H_drv, &cp.H_mpa, &cp.H_frc, &cp.H_mnt, myid, cp.Csize, cp.nfrc, cp.nmnt, cp.nstf, cdx, status);
#ifdef withABS		
		mpiRecvABC(&cp.H_apr, myid, cp.Csize, cdx, status);
#endif
		mpiRecvPP(&cp.Hpt, myid, cp.ppn, status);
		mpiRecvSP(cp.HSpt, cp.nsnap, myid, cp.CSpn, status);
		
		if(cp.fpn) mpiRecvFP(&cp.HFpt, myid, cp.fpn, status);

		//check HFpt get 
		//for(int i=0;i<cp.fpn;i++) 
		//{ 
		//	printf("in PCS[%d] HFpt has %d point---->Rsn[%d],Gsn[%d]:(%d,%d,%d)\n",
		//			myid, cp.fpn, cp.HFpt.Rsn[i], cp.HFpt.Gsn[i], cp.HFpt.locx[i], cp.HFpt.locy[i], cp.HFpt.locz[i]); 
		//}


	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	if(!myid)
	{
		if(src.Rmnt.np) mpiFDSend(src.Rmnt, ChildProcsNum, Mid.fp, src.Rmnt.nt, src.Fpt);
	}
	else
	{
		if(cp.fpn) mpiFDRecv(cp.H_Rmnt, myid, cp.fpn, cp.FNT, cp.HFpt, status);
		//cout<<cp.H_Rmnt.locx[0]<<endl; cout<<cp.H_Rmnt.locy[0]<<endl; cout<<cp.H_Rmnt.locz[0]<<endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);

//return 0;

	if(myid)
	{
		printf("CP.ConIndex=%d, Hygrid=%d, PVF=%d\n",cp.ConIndex,cp.HyGrid,cp.PVF);
		//for(int i=0;i<6;i++)
		//	printf("nabs[%d]=%d\t",i,cp.apr.nabs[i]);
		//cout<<endl;
		//for(int i=0;i<cdx.nx;i++)
		//	printf("apdx[%d]=%g\n",i,cp.apr.APDx[i]);
		//for(int jjj=0;jjj<cp.fpn;jjj++)
		//	for(int kkk=0;kkk<10;kkk++)
		//		printf("at PCS[%d],FNT=%d,cp.H_Rmnt.mxx[%d][%d]=%g\n",myid,cp.FNT,jjj,kkk,cp.H_Rmnt.mxx[jjj*cp.FNT+kkk]);


	}

	
//return 0;

	if(myid)
	{
		cp.C2DPointPick();
#ifndef PointOnly		
		cp.C2DSnapPick();
#endif
		
		if(cp.fpn) cp.C2DFocalPick();
		
		cp.ParH2D();
		
		printf("PCS[%d], pass all Pick and ParH2D\n",myid);	
		
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
//return 0;

	if(myid)
	{
		cp.VelCoeff();
		//cp.SynTopo();
		//cp.SynPV();
		printf("PCS[%d], pass all GPU side preprocess\n",myid);	
	}
	MPI_Barrier(MPI_COMM_WORLD);

	
//return 0;



	double Tstart,Tend;//time counts
	int SmallFlag, BCsize;
	BCsize = LenFD*cdx.ny*cdx.nz;//boundary copy size
	if(!myid)
	{
		fprintf(stdout,"\n***Start to do wave-field computing work, from time step %d on %d ChildProcses\n\n", currT, ChildProcsNum);
		Tstart = Tsecond();
	}
//return 0;

	//nt=20;///AAA
	
	while(1)
	{

		if( currT%4==0 )
		{
			//1st
			//-------------------------------FFF-----------------------------------------
			//begin
			if(myid)
			{
				cp.wavesyn(cp.mW, cp.FW);//input
				cp.abssyn(1);//FW to mW
				cp.RKite(0, currT, 1, 1, 1);//RK step1
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);//inter-node communication

			//inn1
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.mW, cp.W);//reflush temporal wave filed
				cp.abssyn(2);//W to mW
				cp.RKite(1, currT, -1, -1, -1);//RK step2
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);

			//inn2
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.mW, cp.W);//reflush temporal wave filed
				cp.abssyn(2);//W to mW
				cp.RKite(2, currT, 1, 1, 1);//RK step3
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);

			//final
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.mW, cp.W);//reflush temporal wave filed
				cp.abssyn(2);//W to mW
				cp.RKite(3, currT, -1, -1, -1);//RK step4
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);
			
			//output and reflush FW(both w and abs)
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.FW, cp.W);//output
				cp.abssyn(3);//W to FW
				cp.PWpick(cp.FW, currT);
#ifndef PointOnly
				cp.SWpick(cp.FW, currT);
#endif			
			}
			MPI_Barrier(MPI_COMM_WORLD);
			
			currT++;//time iterates up when a full RK loop ends
			if(currT>=nt)
			{
				if(myid)
				{
					cp.PWgather(currT);
					mpiPWSend(cp.HPW, myid, currT, cp.nt, cp.ppn, cp.Hpt);
				}
				else
				{
					mpiPWRecv(spt.MPW, ChildProcsNum, currT, spt.nt, Mid.np, spt.Hpt, status);
					spt.point_export(currT, pausetime, stept);
				}
				MPI_Barrier(MPI_COMM_WORLD);
#ifndef PointOnly
				if(myid)
				{
					cp.SWgather(currT);
					mpiSWSend(cp.HSW, myid, currT, cp.nt, cp.nsnap, cp.CSpn, cp.HSpt);
				}
				else
				{
					mpiSWRecv(spt.MSW, ChildProcsNum, currT, spt.nt, spt.nsnap, spt.Snp, spt.HSpt, status);
					spt.snap_export(currT, pausetime, stept);
				}
				MPI_Barrier(MPI_COMM_WORLD);
#endif			
				if(myid)
				{
					if(cp.PVF)
					{
						cp.SynPV();
						mpiPVSend(cp.Hpv, myid, Cxlength, cdx);
					}
				}
				else
				{
					if(spt.PVflag)
					{
						mpiPVRecv(spt.pv, ChildProcsNum, cdx, Mid, status);
						spt.export_pv(seisOutpath, cdx);
					}
				}
				MPI_Barrier(MPI_COMM_WORLD);
				break;//store the previous time wavefield
			}
		}


		if( currT%4==1 )
		{
			//2nd
			//-------------------------------FFB-----------------------------------------
			//begin
			if(myid)
			{
				cp.wavesyn(cp.mW, cp.FW);//input
				cp.abssyn(1);//FW to mW
				cp.RKite(0, currT, 1, 1, -1);//RK step1
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);

			//inn1
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.mW, cp.W);//reflush temporal wave filed
				cp.abssyn(2);//W to mW
				cp.RKite(1, currT, -1, -1, 1);//RK step2
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);

			//inn2
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.mW, cp.W);//reflush temporal wave filed
				cp.abssyn(2);//W to mW
				cp.RKite(2, currT, 1, 1, -1);//RK step3
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);

			//final
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.mW, cp.W);//reflush temporal wave filed
				cp.abssyn(2);//W to mW
				cp.RKite(3, currT, -1, -1, 1);//RK step4
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);
			
			//output and reflush FW(both w and abs)
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.FW, cp.W);//output
				cp.abssyn(3);//W to FW
				cp.PWpick(cp.FW, currT);
#ifndef PointOnly
				cp.SWpick(cp.FW, currT);
#endif
			}
			MPI_Barrier(MPI_COMM_WORLD);

			currT++;//time iterates up when a full RK loop ends
			if(currT>=nt)
			{
				if(myid)
				{
					cp.PWgather(currT);
					mpiPWSend(cp.HPW, myid, currT, cp.nt, cp.ppn, cp.Hpt);
				}
				else
				{
					mpiPWRecv(spt.MPW, ChildProcsNum, currT, spt.nt, Mid.np, spt.Hpt, status);
					spt.point_export(currT, pausetime, stept);
				}
				MPI_Barrier(MPI_COMM_WORLD);
#ifndef PointOnly
				if(myid)
				{
					cp.SWgather(currT);
					mpiSWSend(cp.HSW, myid, currT, cp.nt, cp.nsnap, cp.CSpn, cp.HSpt);
				}
				else
				{
					mpiSWRecv(spt.MSW, ChildProcsNum, currT, spt.nt, spt.nsnap, spt.Snp, spt.HSpt, status);
					spt.snap_export(currT, pausetime, stept);
				}
				MPI_Barrier(MPI_COMM_WORLD);
#endif			
				if(myid)
				{
					if(cp.PVF)
					{
						cp.SynPV();
						mpiPVSend(cp.Hpv, myid, Cxlength, cdx);
					}
				}
				else
				{
					if(spt.PVflag)
					{
						mpiPVRecv(spt.pv, ChildProcsNum, cdx, Mid, status);
						spt.export_pv(seisOutpath, cdx);
					}
				}
				MPI_Barrier(MPI_COMM_WORLD);
				break;//store the previous time wavefield
			}

		}

	
		if( currT%4==2 )
		{
			//3rd
			//-------------------------------BBB-----------------------------------------
			//begin
			if(myid)
			{
				cp.wavesyn(cp.mW, cp.FW);//input
				cp.abssyn(1);//FW to mW
				cp.RKite(0, currT, -1, -1, -1);//RK step1
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);

			//inn1
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.mW, cp.W);//reflush temporal wave filed
				cp.abssyn(2);//W to mW
				cp.RKite(1, currT, 1, 1, 1);//RK step2
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);

			//inn2
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.mW, cp.W);//reflush temporal wave filed
				cp.abssyn(2);//W to mW
				cp.RKite(2, currT, -1, -1, -1);//RK step3
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);

			//final
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.mW, cp.W);//reflush temporal wave filed
				cp.abssyn(2);//W to mW
				cp.RKite(3, currT, 1, 1, 1);//RK step4
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);
			
			//output and reflush FW(both w and abs)
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.FW, cp.W);//output
				cp.abssyn(3);//W to FW
				cp.PWpick(cp.FW, currT);
#ifndef PointOnly
				cp.SWpick(cp.FW, currT);
#endif	
			}
			MPI_Barrier(MPI_COMM_WORLD);

			currT++;//time iterates up when a full RK loop ends
			if(currT>=nt)
			{
				if(myid)
				{
					cp.PWgather(currT);
					mpiPWSend(cp.HPW, myid, currT, cp.nt, cp.ppn, cp.Hpt);
				}
				else
				{
					mpiPWRecv(spt.MPW, ChildProcsNum, currT, spt.nt, Mid.np, spt.Hpt, status);
					spt.point_export(currT, pausetime, stept);
				}
				MPI_Barrier(MPI_COMM_WORLD);
#ifndef PointOnly
				if(myid)
				{
					cp.SWgather(currT);
					mpiSWSend(cp.HSW, myid, currT, cp.nt, cp.nsnap, cp.CSpn, cp.HSpt);
				}
				else
				{
					mpiSWRecv(spt.MSW, ChildProcsNum, currT, spt.nt, spt.nsnap, spt.Snp, spt.HSpt, status);
					spt.snap_export(currT, pausetime, stept);
				}
				MPI_Barrier(MPI_COMM_WORLD);
#endif			
				if(myid)
				{
					if(cp.PVF)
					{
						cp.SynPV();
						mpiPVSend(cp.Hpv, myid, Cxlength, cdx);
					}
				}
				else
				{
					if(spt.PVflag)
					{
						mpiPVRecv(spt.pv, ChildProcsNum, cdx, Mid, status);
						spt.export_pv(seisOutpath, cdx);
					}
				}
				MPI_Barrier(MPI_COMM_WORLD);
				break;//store the previous time wavefield
			}

		}


		if( currT%4==3 )
		{
			//4th
			//-------------------------------BBF-----------------------------------------
			//begin
			if(myid)
			{
				cp.wavesyn(cp.mW, cp.FW);//input
				cp.abssyn(1);//FW to mW
				cp.RKite(0, currT, -1, -1, 1);//RK step1
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);

			//inn1
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.mW, cp.W);//reflush temporal wave filed
				cp.abssyn(2);//W to mW
				cp.RKite(1, currT, 1, 1, -1);//RK step2
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);

			//inn2
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.mW, cp.W);//reflush temporal wave filed
				cp.abssyn(2);//W to mW
				cp.RKite(2, currT, -1, -1, 1);//RK step3
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);

			//final
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.mW, cp.W);//reflush temporal wave filed
				cp.abssyn(2);//W to mW
				cp.RKite(3, currT, 1, 1, -1);//RK step4
				cp.ShareData();//intra-node, inter-device communication
				cp.IntraBoundGS(1);//gather bound into a node, for inter-node communication use 
			}
			BoundGS(cp.IraB, cdx, ChildProcsNum, myid, 0, status);
			
			//output and reflush FW(both w and abs)
			if(myid)
			{
				cp.IntraBoundGS(0);//scatter bound into a node, from inter-node communication
				cp.wavesyn(cp.FW, cp.W);//output
				cp.abssyn(3);//W to FW
				cp.PWpick(cp.FW, currT);
#ifndef PointOnly
				cp.SWpick(cp.FW, currT);
#endif	
			}
			MPI_Barrier(MPI_COMM_WORLD);

			currT++;//time iterates up when a full RK loop ends
			if(currT>=nt)
			{
				if(myid)
				{
					cp.PWgather(currT);
					mpiPWSend(cp.HPW, myid, currT, cp.nt, cp.ppn, cp.Hpt);
				}
				else
				{
					mpiPWRecv(spt.MPW, ChildProcsNum, currT, spt.nt, Mid.np, spt.Hpt, status);
					spt.point_export(currT, pausetime, stept);
				}
				MPI_Barrier(MPI_COMM_WORLD);
#ifndef PointOnly
				if(myid)
				{
					cp.SWgather(currT);
					mpiSWSend(cp.HSW, myid, currT, cp.nt, cp.nsnap, cp.CSpn, cp.HSpt);
				}
				else
				{
					mpiSWRecv(spt.MSW, ChildProcsNum, currT, spt.nt, spt.nsnap, spt.Snp, spt.HSpt, status);
					spt.snap_export(currT, pausetime, stept);
				}
				MPI_Barrier(MPI_COMM_WORLD);
#endif			
				if(myid)
				{
					if(cp.PVF)
					{
						cp.SynPV();
						mpiPVSend(cp.Hpv, myid, Cxlength, cdx);
					}
				}
				else
				{
					if(spt.PVflag)
					{
						mpiPVRecv(spt.pv, ChildProcsNum, cdx, Mid, status);
						spt.export_pv(seisOutpath, cdx);
					}
				}
				MPI_Barrier(MPI_COMM_WORLD);
				break;//store the previous time wavefield
			}
	
		}
		

		if(!myid && currT && currT%200==0)
		{
			Tend = Tsecond();
			fprintf(stdout,"Processed %d steps (totally %d steps) and %lf seconds passed (still need %lf seconds)\n", 
				currT,nt, Tend-Tstart, (Tend-Tstart)*(nt-currT)/currT);
		}

	}
	MPI_Barrier(MPI_COMM_WORLD);

	if(!myid)
	{
		Tend = Tsecond();
		time(&Time);
		actime = asctime(localtime(&Time));
		fprintf(stdout,"\nWork done at %s totally duration is %lf seconds ( equals to %lf hours )\n",actime,Tend-Tstart,(Tend-Tstart)/60.0/60.0);
	}


	if(!myid)
	{
		spt.data_export_end(spt.pnt, spt.snp, spt.wbuffer);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	delete [] i_CSpn;	delete [] i_nabs;
	datafree(&Mid);
	MPI_Barrier(MPI_COMM_WORLD);

	if(!myid)
		fprintf(stdout,"\n***Program finished\n");
	MPI_Finalize();

	fprintf(stdout,"after finalze()====Procs[%d]\n",myid);

	return 0;
}

//-------------------------------------------------
void get_conf_file(int argc, char*argv[], int myid)
{
	if(argc==2)
		strcpy(seisfile,argv[1]);
	else
		strcpy(seisfile,"SeisFD3D.conf");
	if(!myid)
		printf("the input configure file is %s\n",seisfile);
}

void get_conf()
{
	char errstr[SeisStrLen];
	char seispath[SeisStrLen];
	//char valstr[SeisStrLen];    //for restart
	FILE *fp;
	fp=fopen(seisfile,"r");
	if(!fp)
	{
		sprintf(errstr,"fail to open %s\n",seisfile);
		errprt(Fail2Open,errstr);
	}
	printf("***Start to read the master configure file %s in the main program!\n",seisfile);

	com.get_conf(fp,"ni",3,&cdx.ni);//valid number of x
	com.get_conf(fp,"nj",3,&cdx.nj);//valid number of y
	com.get_conf(fp,"nk",3,&cdx.nk);//valid number of z
	com.get_conf(fp,"stept",3,&stept);
	com.get_conf(fp,"steph",3,&steph);
	com.get_conf(fp,"nt",3,&nt);
	com.get_conf(fp,"seispath",3,seispath);
	com.get_conf(fp,"restart_reading",3,&restart);
	com.get_conf(fp,"last_pause_time",3,&pausetime);

	if(!restart)
		pausetime = 0;

	fclose(fp);

	if(restart) 
		printf("+++***---NOTICE: the work is reastart, it will read the already exist GRID DERIV MEDIA SOURCE STATION!!!!\n");

	strcpy(seisInpath,seispath);
	strcpy(seisOutpath,seispath);
	strcat(seisInpath,"/input");
	strcat(seisOutpath,"/output");

	if(access(seisInpath,F_OK)!=0)
	{
		printf("there is no input path, so we make it as %s\n",seisInpath);
		mkdir(seisInpath,0777);
	}
	if(access(seisOutpath,F_OK)!=0)
	{
		printf("there is no output path, so we make it as %s\n",seisOutpath);
		mkdir(seisOutpath,0777);
	}
	//   nx1            ni1                                    ni2             nx2 
	//   0    1    2    3    4    5    6    7    8    9   10   11   12   13    14
	//   F    F    F    A    A    A    A    A    A    A    A    F    F    F
	//   valid point = 8 = ni
	//   all point = ni + 2*LenFD = 8 + 6 = 14
	//   nx1 = 0
	//   ni1 = nx1 + LenFD = 0 + 3 = 3
	//   ni2 = ni1 + ni = 3 + 8 = 11
	//   nx2 = ni2 + LenFD = 14


	cdx.nx=cdx.ni+2*LenFD;
	cdx.ny=cdx.nj+2*LenFD;
	cdx.nz=cdx.nk+2*LenFD;

	cdx.nx1=0;
	cdx.ni1=cdx.nx1+LenFD;
	cdx.ni2=cdx.ni1+cdx.ni;
	cdx.nx2=cdx.ni2+LenFD;

	cdx.ny1=0;
	cdx.nj1=cdx.ny1+LenFD;
	cdx.nj2=cdx.nj1+cdx.nj;
	cdx.ny2=cdx.nj2+LenFD;

	cdx.nz1=0;
	cdx.nk1=cdx.nz1+LenFD;
	cdx.nk2=cdx.nk1+cdx.nk;
	cdx.nz2=cdx.nk2+LenFD;
}

