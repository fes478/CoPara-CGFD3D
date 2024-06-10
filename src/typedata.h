#include "typenew.h"
#include<mpi.h>
#include<cuda.h>
#include<vector_types.h>//dim3
#include<unistd.h>
#include<sys/time.h>

using namespace constant;
using namespace defstruct;
using namespace std;


namespace flatstruct
{
	typedef struct
	{
		int *CPNum;
		int *fp;//focal number in each CP
		int *np;//receiver number in each CP
		int *Xstart,*Xend,*Xsize;//valid data
		int *CopyS,*CopyE,*CopySize;//valid data with virtual boundary
	}M2Csplit; //allocated in MainProgram

	typedef struct
	{
		int DNum;
		int *fp;//receiver number in each device
		int *np;//receiver number in each device
		int **Snp;//snapshot receiver point in [each Dev * each snapshot]
		int ydim,xdim;
		int *Size;
		int *Rank;//device rank
		int *xl, *xr;//start and end index location of valid region
		int *yd, *yu;// V V V | xl[0] A A A A A xr[0] xl[1] A A A xr[1] xl[2] A A A xr[2] | V V V
	}C2Dsplit; //allocated in ChildProcs

	typedef struct
	{
		int *Rsn,*Gsn;
		int *locx,*locy,*locz;
	}FocalIndexBufferF;

	typedef struct
	{
		int *Rsn,*Gsn;
		int *locx,*locy,*locz;
	}PointIndexBufferF;
	
	typedef struct
	{
		int tinv,cmp;
		int *Rsn,*Gsn;
		int *locx,*locy,*locz;
	}SnapIndexBufferF;

	//flattened and reduced array
	typedef struct
	{
		Real *xix,*xiy,*xiz;
		Real *etax,*etay,*etaz;
		Real *zetax,*zetay,*zetaz;
		Real *jac;
	}derivF;

	typedef struct
	{
		Real *alpha,*beta,*rho;
	}mdparF;

	typedef struct
	{
		int *locx,*locy,*locz;//index
		Real *fx,*fy,*fz;
		Real *stf;
#ifdef SrcSmooth
		Real *dnorm;
#endif
	}forceF;

	typedef struct
	{
		int *locx,*locy,*locz;
		Real *mxx,*myy,*mzz,*mxy,*mxz,*myz;
		Real *stf;
#ifdef SrcSmooth
		Real *dnorm;
#endif
	}momentF;
	
	typedef struct
	{
		int *locx,*locy,*locz;
		Real *mxx,*myy,*mzz,*mxy,*mxz,*myz;
#ifdef SrcSmooth
		Real *dnorm;
#endif
	}RmomF;
	
	typedef struct
	{
		Real *mxx,*myy,*mzz,*mxy,*mxz,*myz;
	}InterpMom;

	typedef struct
	{
		Real *DxTxx,*DxTyy,*DxTzz,*DxTxy,*DxTxz,*DxTyz,*DxVx,*DxVy,*DxVz;
		Real *DyTxx,*DyTyy,*DyTzz,*DyTxy,*DyTxz,*DyTyz,*DyVx,*DyVy,*DyVz;
		Real *DzTxx,*DzTyy,*DzTzz,*DzTxy,*DzTxz,*DzTyz,*DzVx,*DzVy,*DzVz;
	}PartialD;

}


class ChildProcs
{
	private:
		common com;
		dim3 BlockPerGrid,ThreadPerBlock;
		dim3 *BPG;
		bool Rwork,Mflag;
		static const Real RK4A[3], RK4B[4];
		int fullsize,hysize;//hysize only store curvilinear parts
		int axsize,aysize,azsize;//ade size in absorbtion area
		Real stept,steph;
		Real InterpTime;
		int HostMpiRank;
		
		Real **matVx2Vz,**matVy2Vz;

		void GpuAbility(const char*);
		Real ExtractValue(Real*, Real, Real, Real, Real);
		void InterpFocus(Real);

	public:
		int Csize;//with virtual bounds, only X (have not mutiply Y)
		int Cxn;//valid size of X
		int Cstart;//relative starting location
		int ppn,nt;
		int *CSpn,nsnap;
		int fpn,FNT;
		Real FDT;
		int nfrc,nmnt,nstf;
		int ConIndex,HyGrid;//Convert depth index(to bottom)
		int PVF;//peak vel output flag
		flatstruct::C2Dsplit Cid;
		defstruct::cindx cdx;
		
		flatstruct::derivF *drv, H_drv;
		flatstruct::mdparF *mpa, H_mpa;
		
		defstruct::apara *apr, H_apr;
		flatstruct::forceF *frc, H_frc;
		flatstruct::momentF *mnt, H_mnt;
		flatstruct::RmomF *Rmnt, H_Rmnt;
		flatstruct::InterpMom *IM;
		
		flatstruct::FocalIndexBufferF HFpt,*DFpt;
		flatstruct::PointIndexBufferF Hpt,*Dpt,*D_Dpt;
		defstruct::wfield HPW,*DPW;
		flatstruct::SnapIndexBufferF *HSpt, **DSpt, **D_DSpt;
		defstruct::wfield *HSW,**DSW;//use marco DevicePick
		defstruct::PeakVel *Dpv, Hpv;//peak vel

		defstruct::wfield GD,IraB;//HOST, GD for data output; IraB for ChildPCS is node bounds(2*LenFD), for HostPCS is full node bounds(cpn+1)*2*LenFD
		defstruct::wfield *FW, *h_FW;//fullwave data	DEVICE
		defstruct::wfield *W, *mW, *hW, *tW;//wave field in RK
		flatstruct::PartialD *pd;//partial derivative
		
#ifdef CFSPML		
		defstruct::wfield *Ax,*mAx,*hAx,*tAx,*FAx;//absorbtion wave field of ADE 
		defstruct::wfield *Ay,*mAy,*hAy,*tAy,*FAy;//absorbtion wave field of ADE 
		defstruct::wfield *Az,*mAz,*hAz,*tAz,*FAz;//absorbtion wave field of ADE 
#endif
		
		ChildProcs(const char*, defstruct::cindx, Real, Real, 
			   int, int, int, int, int, int, 
			   int, int, int*, 
			   const int, const int, int*, int, const int, const int, Real, int,
			   const int, const int, const int);
		~ChildProcs();
		
		//syn for computation
		void wavesyn(defstruct::wfield*, defstruct::wfield*);//previous and temporal wave cache
		void abssyn(int);

		//syn for boundary  operate at W
		void ShareData();//D2D, device boundary
		void IntraBoundGS(int);//D2H2D, node boundary gather and scatter, need cooperating with BoundGS
		
		//HD bidirection data transfer   operate at FW
		void GatherData(defstruct::wfield, defstruct::wfield*, int);// D2H/H2D H(node-size) D(device-size), gather/scatter
		void SynData();//D2H device-size, for display
		void SynTopo();//D2H,to display topo matVx2Vz and Vy2Vz
		void SynPV();//D2H,to transfer PeakVel
		
		void RKite(int, int, int, int, int);
		void VelCoeff();
		void ParH2D();

		//output point and snapshot, default use host pick(directly from FW to HPW&HSW
		void C2DPointPick();
		void PWpick(defstruct::wfield*, int);
		void PWgather(int);//valid under DevicePick
		void C2DSnapPick();
		void SWpick(defstruct::wfield*, int);
		void SWgather(int);//valid under DevicePick

		//input focus
		void C2DFocalPick();


	//NOTE:
		//defstruct::wfield W      for master procs store boundary
		//                             child procs store wave field


};

//-------------Send & Recv Pars-------------------
void mpisend(defstruct::cindx*, int, int);
void mpirecv(defstruct::cindx*, int, int, MPI_Status);
void mpisend(int, Real, Real, int, int, int, int, int, int, int*, int*, int*, int*, int, Real, int, int, int, int);
	   //nt, steph, stept, nfrc,nmnt,nstf,restart,ConIndex,HyGrid,nabs,Mid.np,spt.Snp,Mid.fp,spt.nsnap,pvflag,cpn,Dtag
void mpirecv(int*, Real*, Real*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*, Real*, int, int*, int, int, MPI_Status);
void mpiSendPars(defstruct::deriv*, defstruct::mdpar*, defstruct::force*, defstruct::moment*, 
		 int, int, int, int, defstruct::cindx, flatstruct::M2Csplit);
void mpiRecvPars(flatstruct::derivF*, flatstruct::mdparF*, flatstruct::forceF*, flatstruct::momentF*, 
		 int, int, int, int, int, defstruct::cindx, MPI_Status);
void mpisend(defstruct::deriv*, int, int, defstruct::cindx, flatstruct::M2Csplit);
void mpisend(defstruct::mdpar*, int, int, defstruct::cindx, flatstruct::M2Csplit);
void mpisend(defstruct::force*, int, int, int, int);
void mpisend(defstruct::moment*, int, int, int, int);
void mpirecv(flatstruct::derivF*, int, int, int, defstruct::cindx, MPI_Status);
void mpirecv(flatstruct::mdparF*, int, int, int, defstruct::cindx, MPI_Status);
void mpirecv(flatstruct::forceF*, int, int, int, int, MPI_Status);
void mpirecv(flatstruct::momentF*, int, int, int, int, MPI_Status);
//------------Send & Recv ABC--------------------
void mpiSendABC(defstruct::apara*, int, defstruct::cindx, flatstruct::M2Csplit);
void mpiRecvABC(defstruct::apara*, int, int, defstruct::cindx, MPI_Status);
void mpiSendABC(int*, Real*, Real*, Real*, int*, int, defstruct::cindx, flatstruct::M2Csplit);//deprecated
void mpiRecvABC(int*, Real*, Real*, Real*, int*, int, int, defstruct::cindx, MPI_Status);//deprecated
void mpisend(int*, Real*, Real*, Real*, int*, int, int, defstruct::cindx, flatstruct::M2Csplit);//AbsExp
void mpirecv(int*, Real*, Real*, Real*, int*, int, int, int, defstruct::cindx, MPI_Status);
void mpiSendABC(int*, Real*, Real*, Real*, Real*, Real*, Real*, Real*, Real*, Real*, int*, int, defstruct::cindx, flatstruct::M2Csplit);//deprecated
void mpiRecvABC(int*, Real*, Real*, Real*, Real*, Real*, Real*, Real*, Real*, Real*, int*, int, int, defstruct::cindx, MPI_Status);//deprecated
void mpisend(int*, Real*, Real*, Real*, Real*, Real*, Real*, Real*, Real*, Real*, int*, int, int, defstruct::cindx, flatstruct::M2Csplit);//AbsPML
void mpirecv(int*, Real*, Real*, Real*, Real*, Real*, Real*, Real*, Real*, Real*, int*, int, int, int, defstruct::cindx, MPI_Status);
//------------Send & Recv Datas------------------
void mpiDataGatherSend(defstruct::wfield, int, int, defstruct::cindx);
void mpiDataGatherRecv(defstruct::wfield, int, defstruct::cindx, flatstruct::M2Csplit, MPI_Status);
void RestartSend(defstruct::wfield, int, int, int, defstruct::cindx, flatstruct::M2Csplit);
void RestartRecv(defstruct::wfield, int*, int, int, int, defstruct::cindx, MPI_Status);
void mpisend(defstruct::wfield, int, int, int, defstruct::cindx);
void mpirecv(defstruct::wfield, int, int, defstruct::cindx, flatstruct::M2Csplit, MPI_Status);
void mpiPVSend(defstruct::PeakVel, int, int, defstruct::cindx);
void mpiPVRecv(defstruct::PeakVel, int, defstruct::cindx, flatstruct::M2Csplit, MPI_Status);
//------------Send & Recv Bounds-----------------------
void BoundSend(defstruct::wfield, defstruct::cindx, flatstruct::M2Csplit, int, int);
void BoundRecv(defstruct::wfield, defstruct::cindx, int, int, int, MPI_Status);
void BoundGS(defstruct::wfield, defstruct::cindx, int, int, int, MPI_Status);
void BoundGatherSendOB(defstruct::wfield, int, int, int, int);//only boundary data use
void BoundGatherSend(defstruct::wfield, int, int, int, int);
void BoundGatherRecv(defstruct::wfield, int, int, int, MPI_Status);
void BoundScatterSend(defstruct::wfield, int, int, int);
void BoundScatterRecv(defstruct::wfield, int, int, int, int, MPI_Status);
void BoundExtend(defstruct::wfield, int, int);
//------------Send & Recv PP and Snap-----------------------
void mpiSendPP(defstruct::PointIndexBuffer*, int, int*);
void mpiRecvPP(flatstruct::PointIndexBufferF*, int, int, MPI_Status);
void mpiPWSend(defstruct::wfield, int, int, int, int, flatstruct::PointIndexBufferF);
void mpiPWRecv(defstruct::wfield, int, int, int, int*, defstruct::PointIndexBuffer, MPI_Status);
void mpiSendSP(defstruct::SnapIndexBuffer*, int, int, int*);
void mpiRecvSP(flatstruct::SnapIndexBufferF*, int, int, int*, MPI_Status);
void mpiSWSend(defstruct::wfield*, int, int, int, int, int*, flatstruct::SnapIndexBufferF*);
void mpiSWRecv(defstruct::wfield*, int, int, int, int, int*, defstruct::SnapIndexBuffer*, MPI_Status);
//------------Send Focus------------------------------------
void mpiSendFP(defstruct::FocalIndexBuffer*, int, int*);
void mpiRecvFP(flatstruct::FocalIndexBufferF*, int, int, MPI_Status);
void mpiFDSend(defstruct::Rmom, int, int*, int, defstruct::FocalIndexBuffer);
void mpiFDRecv(flatstruct::RmomF, int, int, int, flatstruct::FocalIndexBufferF, MPI_Status);

void dataalloc(flatstruct::M2Csplit*, int);
void datafree(flatstruct::M2Csplit*);
void splitM2C(const char*, const int, int, int, int*, int*, int*, flatstruct::M2Csplit*, MPI_Status);

void bimax(int, float, int *);
int idxcom(int, int, int, int, int*, int*, int*, int*);//for C2D
void CPdataIdx(int ,int, int*, int*, int*, int*, int*, int*, int*);//for M2C
void loadfixedarray(int*, int, int, int, int, int, int, flatstruct::C2Dsplit);



double Tsecond();//unit--->second


