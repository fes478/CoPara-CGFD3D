#include "typedata.h"
#include<cuda_runtime.h>
#include<helper_cuda.h>
#include<helper_functions.h>
#include<unistd.h>

using namespace std;
using namespace defstruct;
using namespace flatstruct;
using namespace constant;

const Real ChildProcs::RK4A[3] = {0.5, 0.5, 1.0};
const Real ChildProcs::RK4B[4] = {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0};

#define errprt(...) com.errorprint(__FILE__, __LINE__, __VA_ARGS__)

#define CC(call) CheckCuda(__FILE__, __LINE__, call)
inline void CheckCuda(const char *file, const int line, cudaError_t errS)
{//check at special cuda calls
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != errS )
	{
		char errstr[256];
		sprintf(errstr,"CUDA Error at %s line %d, error string is %s, error definition and numberis %s <%d> \n",
				file, line, errS, cudaGetErrorString(errS), (int)err);
		cudaDeviceReset();
		MpiErrorPrint(file,line,Fail2Cuda,errstr);
	}
}

#define Kcheck(call) KernelCheck(__FILE__,__LINE__,call)
inline void KernelCheck(const char *file, const int line, const char* call)
{//check at special position
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess!=err)
	{
		char errstr[256];
		sprintf(errstr,"%s behind line %d in file %s, error information is %s\n",call,line,file,cudaGetErrorString(err));
		cudaDeviceReset();
		MpiErrorPrint(file,line,Fail2Kernel,errstr);
	}
}

//---------------------------------------device kernel and function declaration---------------
cudaError_t errmessage;
__constant__ int ipam[11];
__device__ void matinv(Real *A);//3*3 matrix invertion
__device__ void matmul(Real *A, Real *B, Real *C);//3*3 matrix mutiply

__global__ void perform();//display ipam
__global__ void generatewave(defstruct::wfield, int, int);//check wave and index

__global__ void WavefieldPick(defstruct::wfield, defstruct::wfield, flatstruct::PointIndexBufferF, int, int, int);
__global__ void SnapWavefieldPick(defstruct::wfield, defstruct::wfield, flatstruct::SnapIndexBufferF, int, int, int);//Abandoned

__global__ void VelPDcoeff(flatstruct::derivF, flatstruct::mdparF, defstruct::apara, Real*, Real*);// velocity partial derivative conversion coeffients

__global__ void CalDiff(int, int, int, int, Real, Real*, Real*, defstruct::wfield, flatstruct::PartialD);// space-domain stress and velocity partial derivative
__global__ void CalWave(int, flatstruct::derivF, flatstruct::mdparF, flatstruct::PartialD, defstruct::apara, Real*, Real*, defstruct::wfield,
			  defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield);
__global__ void CalDiffCL(int, int, int, int, Real, Real*, Real*, defstruct::wfield, flatstruct::PartialD);// space-domain stress and velocity partial derivative
__global__ void CalWaveCL(int, flatstruct::derivF, flatstruct::mdparF, flatstruct::PartialD, defstruct::apara, Real*, Real*, defstruct::wfield,
			  defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield);
__global__ void CalDiffSL(int, int, int, int, Real, Real*, Real*, defstruct::wfield, flatstruct::PartialD);// space-domain stress and velocity partial derivative
__global__ void CalWaveSL(int, flatstruct::derivF, flatstruct::mdparF, defstruct::apara, flatstruct::PartialD, defstruct::wfield,
			  defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield);
__global__ void CalTIMG(int, int, int, Real, Real*, flatstruct::derivF, defstruct::wfield, defstruct::wfield,
		        defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::apara);// Z-direction traction image free surface condition
__global__ void CalVUCD(int, int, int, Real, Real*, Real*, flatstruct::mdparF, flatstruct::derivF, defstruct::wfield, defstruct::wfield,
			defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::apara);// velocity unilater compact difference
__global__ void LoadForce(int, defstruct::cindx, Real, int, int, flatstruct::forceF, Real*, Real*, defstruct::wfield);//load force
__global__ void LoadMoment(int, defstruct::cindx, Real, int, int, flatstruct::momentF, Real*, defstruct::wfield);//load moment
__global__ void LoadRmom(defstruct::cindx, Real, int, flatstruct::RmomF, Real*, defstruct::wfield);//load focus
	
__global__ void IterationBegin(Real, Real, Real, defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield,
			       int*, defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield,
			       defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield,
			       defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield);//RK iteration begin
__global__ void IterationInner(Real, Real, Real, defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield,
			       int*, defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield,
			       defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield,
			       defstruct::wfield, defstruct::wfield, defstruct::wfield, defstruct::wfield);//RK iteration inner, excute twice
__global__ void IterationFinal(Real, Real, defstruct::wfield, defstruct::wfield, defstruct::wfield,
			       int*, defstruct::wfield, defstruct::wfield, defstruct::wfield,
			       defstruct::wfield, defstruct::wfield, defstruct::wfield,
			       defstruct::wfield, defstruct::wfield, defstruct::wfield);//RK iteration final
__global__ void IterationFinalPV(Real, Real, defstruct::PeakVel, defstruct::wfield, defstruct::wfield, defstruct::wfield,
			       int*, defstruct::wfield, defstruct::wfield, defstruct::wfield,
			       defstruct::wfield, defstruct::wfield, defstruct::wfield,
			       defstruct::wfield, defstruct::wfield, defstruct::wfield);//RK iteration final
__global__ void ErrorSta(defstruct::wfield , int*);
__global__ void AbsExp(Real*, Real*, Real*, int*, defstruct::wfield);//expotional absorbing condition


//--------------------------------public--------------------------------------------
ChildProcs::ChildProcs(const char *filename, cindx i_cdx, Real i_steph, Real i_stept, 
			int i_nfrc, int i_nmnt, int i_nstf, int sepsize, int cxn, int cstart, 
			int i_ConIndex, int i_HyGrid, int *i_nabs,
			const int Cppn, const int i_nt, int *i_CSpn, int i_nsnap, const int Cfpn, const int Cfnt, Real Cfdt, int PVflag,
			const int restart, const int myid, const int cpn)
{
	int i;
	//pars init
	HostMpiRank = myid;
	cdx = i_cdx;
	steph = i_steph;
	stept = i_stept;
	nfrc = i_nfrc;
	nmnt = i_nmnt;
	nstf = i_nstf;
	Csize = sepsize;//valid size with 2 bounds
	Cxn = cxn;//valid size
	Cstart = cstart;//start index in absolute location
	ConIndex = i_ConIndex;//default at bottom
	HyGrid = i_HyGrid;
	ppn = Cppn;
	nt = i_nt;
	nsnap = i_nsnap;
	fpn = Cfpn;
	FNT = Cfnt;
	FDT = Cfdt;
	InterpTime = -1.0;//time to interp focus
	PVF = PVflag;
	
	CSpn = new int[nsnap]();
	for(i=0;i<nsnap;i++)
		CSpn[i] = i_CSpn[i];
	
	//host side boundary GS buffer, X-dir two bounds length
	fullsize = 2*LenFD*i_cdx.ny*i_cdx.nz;
	if(!myid) fullsize = (cpn+1)*fullsize;
	IraB.Txx = new Real [fullsize]();
	IraB.Tyy = new Real [fullsize]();
	IraB.Tzz = new Real [fullsize]();
	IraB.Txy = new Real [fullsize]();
	IraB.Txz = new Real [fullsize]();
	IraB.Tyz = new Real [fullsize]();
	IraB.Vx = new Real  [fullsize]();
	IraB.Vy = new Real  [fullsize]();
	IraB.Vz = new Real  [fullsize]();
	
	if(!myid)
	{
		Mflag = true;//master procs   //only malloc needed variables // 	W     for boundary
		printf("---accomplished GPU device boundary allocation work at Process[%d]\n",myid);
		return;
	}
	else
		Mflag = false;//child procs
	
	if(restart==1)
		Rwork = true;//restart work, reading the exists
	else
		Rwork = false;
	
	GpuAbility(filename);
	
	Cid.xdim = idxcom(Cxn, cdx.nj, Cid.DNum, Cid.ydim, Cid.xl, Cid.xr, Cid.yd, Cid.yu);
	
	printf("\n\n***Start to do GPU device initialization, parameter transfer, data array allocation and prepare\n");
	printf("On Rank %d node, full_X_size is %d, vaild_X_size = %d, start at %d, data index seires is :\n",myid,Csize,Cxn,Cstart);
	for(i=0;i<Cid.DNum;i++)
		printf("in device ID[%d]: xl=%d xr=%d yd=%d yu=%d\n",Cid.Rank[i],Cid.xl[i],Cid.xr[i],Cid.yd[i],Cid.yu[i]);
	printf("\n");

	//struct malloc
	int *ipamcache;
	ipamcache = new int[Cid.DNum*11];
	loadfixedarray(ipamcache, HostMpiRank, Cstart, Cxn, cdx.ni, cdx.nj, cdx.nk, Cid);
	for(i=0;i<Cid.DNum;i++)
	{
		printf("at PCS[%d], ipamcache[No.%d]=",HostMpiRank,i);
		for(int j=0;j<11;j++)
			printf("%d ",ipamcache[i*11+j]);
		printf("\n");
	}

	for(i=0;i<Cid.DNum;i++)
	{
		cudaSetDevice( Cid.Rank[i] );
		cudaMemcpyToSymbol(ipam,ipamcache+i*11,11*sizeof(int),0,cudaMemcpyHostToDevice);
		//perform<<<1,1>>>();
	}
	delete [] ipamcache;

	//data struct allocation
	HSW = new wfield[nsnap]();	DSW = new wfield*[nsnap];
	HSpt = new SnapIndexBufferF[nsnap]();
	DSpt = new SnapIndexBufferF*[nsnap];	D_DSpt = new SnapIndexBufferF*[nsnap];
	for(i=0;i<nsnap;i++)
	{	
		DSpt[i] = new SnapIndexBufferF[Cid.DNum]();	D_DSpt[i] = new SnapIndexBufferF[Cid.DNum]();
		DSW[i] = new wfield[Cid.DNum]();
	}

	if(PVF) Dpv = new PeakVel[Cid.DNum]();
	if(fpn) DFpt = new FocalIndexBufferF[Cid.DNum]();//only valid under focal source
	Dpt = new PointIndexBufferF[Cid.DNum]();	D_Dpt = new PointIndexBufferF[Cid.DNum]();
	DPW = new wfield[Cid.DNum]();
	FW = new wfield[Cid.DNum]();	//h_FW = new wfield[Cid.DNum]();	
	W = new wfield[Cid.DNum]();	mW = new wfield[Cid.DNum]();	hW = new wfield[Cid.DNum]();	tW = new wfield[Cid.DNum]();
	pd = new PartialD[Cid.DNum]();

	drv = new derivF[Cid.DNum]();	mpa = new mdparF[Cid.DNum](); 
	matVx2Vz = new Real*[Cid.DNum];	matVy2Vz = new Real*[Cid.DNum];

	apr = new apara[Cid.DNum]();	frc = new forceF[Cid.DNum]();	mnt = new momentF[Cid.DNum]();	Rmnt = new RmomF[Cid.DNum]();	IM = new InterpMom[Cid.DNum]();

#ifdef CFSPML
	Ax = new wfield[Cid.DNum]();	mAx = new wfield[Cid.DNum]();	hAx = new wfield[Cid.DNum]();	tAx = new wfield[Cid.DNum]();	FAx = new wfield[Cid.DNum]();
	Ay = new wfield[Cid.DNum]();	mAy = new wfield[Cid.DNum]();	hAy = new wfield[Cid.DNum]();	tAy = new wfield[Cid.DNum]();	FAy = new wfield[Cid.DNum]();
	Az = new wfield[Cid.DNum]();	mAz = new wfield[Cid.DNum]();	hAz = new wfield[Cid.DNum]();	tAz = new wfield[Cid.DNum]();	FAz = new wfield[Cid.DNum]();
#endif

	fullsize = Csize*cdx.ny*cdx.nz;//seperate size, Csize = Cxn + 2*LenFD
	
	//host side PV allocation
	if(PVF)
	{
		Hpv.Vx = new Real[Csize*cdx.ny](); Hpv.Vy = new Real[Csize*cdx.ny](); Hpv.Vz = new Real[Csize*cdx.ny]();
	}

	//host side HSpt allocation
	for(i=0;i<this->nsnap;i++)
	{
		HSpt[i].Rsn = new int[CSpn[i]]();	HSpt[i].Gsn = new int[CSpn[i]]();
		HSpt[i].locx = new int[CSpn[i]]();	HSpt[i].locy = new int[CSpn[i]]();	HSpt[i].locz = new int[CSpn[i]]();
	}

	//host side Hpt and HPW allocation
	Hpt.Rsn = new int[ppn]();	Hpt.Gsn = new int[ppn]();
	Hpt.locx = new int[ppn]();	Hpt.locy = new int[ppn]();	Hpt.locz = new int[ppn]();

	HPW.Vx = new Real[nt*ppn]();	HPW.Vy = new Real[nt*ppn]();	HPW.Vz = new Real[nt*ppn]();
	HPW.Txx = new Real[nt*ppn]();	HPW.Tyy = new Real[nt*ppn]();	HPW.Tzz = new Real[nt*ppn]();
	HPW.Txy = new Real[nt*ppn]();	HPW.Txz = new Real[nt*ppn]();	HPW.Tyz = new Real[nt*ppn]();

	//host side, node-size gather buffer
	GD.Txx = new Real [fullsize](); GD.Tyy = new Real [fullsize](); GD.Tzz = new Real [fullsize]();
	GD.Txy = new Real [fullsize](); GD.Txz = new Real [fullsize](); GD.Tyz = new Real [fullsize]();
	GD.Vx = new Real  [fullsize](); GD.Vy = new Real  [fullsize](); GD.Vz = new Real  [fullsize]();
	
	//host side, node-size, par buffer, free after deliver
	H_drv.xix  = new Real [ fullsize ](); H_drv.xiy  = new Real [ fullsize ](); H_drv.xiz  = new Real [ fullsize ](); 
	H_drv.etax = new Real [ fullsize ](); H_drv.etay = new Real [ fullsize ](); H_drv.etaz = new Real [ fullsize ](); 
	H_drv.zetax= new Real [ fullsize ](); H_drv.zetay= new Real [ fullsize ](); H_drv.zetaz= new Real [ fullsize ](); 
	H_drv.jac  = new Real [ fullsize ](); 

	H_mpa.alpha = new Real [ fullsize ](); H_mpa.beta = new Real [ fullsize ](); H_mpa.rho = new Real [ fullsize ]();

	if(nfrc)
	{
		H_frc.locx = new int [ nfrc ]();	H_frc.locy = new int [ nfrc ]();	H_frc.locz = new int [ nfrc ]();
		H_frc.fx = new Real [ nfrc ]();		H_frc.fy = new Real [ nfrc ]();		H_frc.fz = new Real [ nfrc ]();
		H_frc.stf = new Real [ nfrc*nstf ]();
#ifdef SrcSmooth
		H_frc.dnorm = new Real [ nfrc*LenNorm*LenNorm*LenNorm ]();
#endif
	}

	if(nmnt)
	{
		H_mnt.locx = new int [ nmnt ](); H_mnt.locy = new int [ nmnt ](); H_mnt.locz = new int [ nmnt ]();
		H_mnt.mxx = new Real [ nmnt ](); H_mnt.myy = new Real [ nmnt ](); H_mnt.mzz = new Real [ nmnt ]();
		H_mnt.mxy = new Real [ nmnt ](); H_mnt.mxz = new Real [ nmnt ](); H_mnt.myz = new Real [ nmnt ]();
		H_mnt.stf = new Real [ nmnt*nstf ]();
#ifdef SrcSmooth
		H_mnt.dnorm = new Real [ nmnt*LenNorm*LenNorm*LenNorm ]();
#endif
	}
	
	if(fpn)
	{
		//host side HFpt allocation
		HFpt.Rsn = new int[fpn]();	HFpt.Gsn = new int[fpn]();
		HFpt.locx = new int[fpn]();	HFpt.locy = new int[fpn]();	HFpt.locz = new int[fpn]();
		
		//host side focal data allocation
		H_Rmnt.locx = new int [ fpn ](); H_Rmnt.locy = new int [ fpn ](); H_Rmnt.locz = new int [ fpn ]();
		H_Rmnt.mxx = new Real [ fpn*FNT ](); H_Rmnt.myy = new Real [ fpn*FNT ](); H_Rmnt.mzz = new Real [ fpn*FNT ]();
		H_Rmnt.mxy = new Real [ fpn*FNT ](); H_Rmnt.mxz = new Real [ fpn*FNT ](); H_Rmnt.myz = new Real [ fpn*FNT ]();
		// fpn->focus point,  FNT->time point
#ifdef SrcSmooth
		H_Rmnt.dnorm = new Real [ fpn*LenNorm*LenNorm*LenNorm ]();
#endif
	}
	
	H_apr.nabs = new int [ SeisGeo*2 ]();
#ifdef CFSPML	
	//CFS PML's par
	H_apr.APDx = new Real [ Csize ]();	H_apr.APDy = new Real [ cdx.ny ]();	H_apr.APDz = new Real [ cdx.nz ]();
	H_apr.Bx = new Real [ Csize ]();	H_apr.By = new Real [ cdx.ny ]();	H_apr.Bz = new Real [ cdx.nz ]();
	H_apr.DBx = new Real [ Csize ]();	H_apr.DBy = new Real [ cdx.ny ]();	H_apr.DBz = new Real [ cdx.nz ]();
	H_apr.CLoc = new int [ 26*6 ]();
#else	
	//Sponge Layer's par
	H_apr.Ex = new Real [ Csize ]();	H_apr.Ey = new Real [ cdx.ny ]();	H_apr.Ez = new Real [ cdx.nz ]();
	H_apr.ELoc = new int [ 6*6 ]();
#endif

	cudaError_t err;

	//device side allocation
	for(i=0;i<Cid.DNum;i++)
	{
		err = cudaSetDevice( Cid.Rank[i] );
		if(err != 0) printf("err = %d, errS=%s, error may occur at setdev\n",err, cudaGetErrorString(err) );

		fullsize = (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*cdx.nz;//with boundary device-size
		hysize = (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*(cdx.nk2-ConIndex);
			//3<=ConIndex<=idz<cdx.nk2,should change array size I/O index
		axsize = (i_nabs[0]+i_nabs[1])*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*cdx.nz;
		aysize = (i_nabs[2]+i_nabs[3])*(Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*cdx.nz;
		azsize = (i_nabs[4]+i_nabs[5])*(Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD);

                //-------------------------------------lanuch pars ------------------------------------------------------
		(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD) <= BlockPerGrid.x ? BPG[i].x = Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD : BPG[i].x = BlockPerGrid.x;
		(Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD) <= BlockPerGrid.y ? BPG[i].y = Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD : BPG[i].y = BlockPerGrid.y;
		BPG[i].z = 1;

		//-------------------------------------wavefield variables-----------------------------------------------
		//h_FW[i].Txx = new Real [fullsize](); h_FW[i].Tyy = new Real [fullsize](); h_FW[i].Tzz = new Real [fullsize]();
		//h_FW[i].Txy = new Real [fullsize](); h_FW[i].Txz = new Real [fullsize](); h_FW[i].Tyz = new Real [fullsize]();
		//h_FW[i].Vx = new Real  [fullsize](); h_FW[i].Vy = new Real  [fullsize](); h_FW[i].Vz = new Real  [fullsize]();
		
		cudaMalloc( (Real**)&FW[i].Txx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&FW[i].Tyy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&FW[i].Tzz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&FW[i].Txy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&FW[i].Txz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&FW[i].Tyz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&FW[i].Vx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&FW[i].Vy, fullsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&FW[i].Vz, fullsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc FW\n",err, cudaGetErrorString(err) );

		cudaMemset(FW[i].Txx, 0, fullsize*sizeof(Real));
		cudaMemset(FW[i].Tyy, 0, fullsize*sizeof(Real));
		cudaMemset(FW[i].Tzz, 0, fullsize*sizeof(Real));
		cudaMemset(FW[i].Txy, 0, fullsize*sizeof(Real));
		cudaMemset(FW[i].Txz, 0, fullsize*sizeof(Real));
		cudaMemset(FW[i].Tyz, 0, fullsize*sizeof(Real));
		cudaMemset(FW[i].Vx, 0, fullsize*sizeof(Real));
		cudaMemset(FW[i].Vy, 0, fullsize*sizeof(Real));
		err = cudaMemset(FW[i].Vz, 0, fullsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at Memset FW\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&W[i].Txx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&W[i].Tyy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&W[i].Tzz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&W[i].Txy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&W[i].Txz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&W[i].Tyz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&W[i].Vx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&W[i].Vy, fullsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&W[i].Vz, fullsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc W\n",err, cudaGetErrorString(err) );

		cudaMemset(W[i].Txx, 0, fullsize*sizeof(Real));
		cudaMemset(W[i].Tyy, 0, fullsize*sizeof(Real));
		cudaMemset(W[i].Tzz, 0, fullsize*sizeof(Real));
		cudaMemset(W[i].Txy, 0, fullsize*sizeof(Real));
		cudaMemset(W[i].Txz, 0, fullsize*sizeof(Real));
		cudaMemset(W[i].Tyz, 0, fullsize*sizeof(Real));
		cudaMemset(W[i].Vx, 0, fullsize*sizeof(Real));
		cudaMemset(W[i].Vy, 0, fullsize*sizeof(Real));
		err = cudaMemset(W[i].Vz, 0, fullsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at Memset W\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&mW[i].Txx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&mW[i].Tyy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&mW[i].Tzz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&mW[i].Txy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&mW[i].Txz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&mW[i].Tyz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&mW[i].Vx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&mW[i].Vy, fullsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&mW[i].Vz, fullsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc mW\n",err, cudaGetErrorString(err) );

		cudaMemset(mW[i].Txx, 0, fullsize*sizeof(Real));
		cudaMemset(mW[i].Tyy, 0, fullsize*sizeof(Real));
		cudaMemset(mW[i].Tzz, 0, fullsize*sizeof(Real));
		cudaMemset(mW[i].Txy, 0, fullsize*sizeof(Real));
		cudaMemset(mW[i].Txz, 0, fullsize*sizeof(Real));
		cudaMemset(mW[i].Tyz, 0, fullsize*sizeof(Real));
		cudaMemset(mW[i].Vx, 0, fullsize*sizeof(Real));
		cudaMemset(mW[i].Vy, 0, fullsize*sizeof(Real));
		err = cudaMemset(mW[i].Vz, 0, fullsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at Memset mW\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&hW[i].Txx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&hW[i].Tyy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&hW[i].Tzz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&hW[i].Txy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&hW[i].Txz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&hW[i].Tyz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&hW[i].Vx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&hW[i].Vy, fullsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&hW[i].Vz, fullsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc hW\n",err, cudaGetErrorString(err) );

		cudaMemset(hW[i].Txx, 0, fullsize*sizeof(Real));
		cudaMemset(hW[i].Tyy, 0, fullsize*sizeof(Real));
		cudaMemset(hW[i].Tzz, 0, fullsize*sizeof(Real));
		cudaMemset(hW[i].Txy, 0, fullsize*sizeof(Real));
		cudaMemset(hW[i].Txz, 0, fullsize*sizeof(Real));
		cudaMemset(hW[i].Tyz, 0, fullsize*sizeof(Real));
		cudaMemset(hW[i].Vx, 0, fullsize*sizeof(Real));
		cudaMemset(hW[i].Vy, 0, fullsize*sizeof(Real));
		err = cudaMemset(hW[i].Vz, 0, fullsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at Memset hW\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&tW[i].Txx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&tW[i].Tyy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&tW[i].Tzz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&tW[i].Txy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&tW[i].Txz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&tW[i].Tyz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&tW[i].Vx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&tW[i].Vy, fullsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&tW[i].Vz, fullsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc tW\n",err, cudaGetErrorString(err) );

		cudaMemset(tW[i].Txx, 0, fullsize*sizeof(Real));
		cudaMemset(tW[i].Tyy, 0, fullsize*sizeof(Real));
		cudaMemset(tW[i].Tzz, 0, fullsize*sizeof(Real));
		cudaMemset(tW[i].Txy, 0, fullsize*sizeof(Real));
		cudaMemset(tW[i].Txz, 0, fullsize*sizeof(Real));
		cudaMemset(tW[i].Tyz, 0, fullsize*sizeof(Real));
		cudaMemset(tW[i].Vx, 0, fullsize*sizeof(Real));
		cudaMemset(tW[i].Vy, 0, fullsize*sizeof(Real));
		err = cudaMemset(tW[i].Vz, 0, fullsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at Memset tW\n",err, cudaGetErrorString(err) );
		//-------------------------------------wavefield variables-----------------------------------------------
		
		//-------------------------------------wavefield partial derivative--------------------------------------
#ifdef HYindex		
		cudaMalloc( (Real**)&pd[i].DxTyy, hysize*sizeof(Real) );//HG
		cudaMalloc( (Real**)&pd[i].DxTzz, hysize*sizeof(Real) );//HG
		cudaMalloc( (Real**)&pd[i].DxTyz, hysize*sizeof(Real) );//HG
		cudaMalloc( (Real**)&pd[i].DyTxx, hysize*sizeof(Real) );//HG
		cudaMalloc( (Real**)&pd[i].DyTzz, hysize*sizeof(Real) );//HG
		cudaMalloc( (Real**)&pd[i].DyTxz, hysize*sizeof(Real) );//HG
		cudaMalloc( (Real**)&pd[i].DzTxx, hysize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DzTyy, hysize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DzTxy, hysize*sizeof(Real) );
		cudaMemset(pd[i].DxTyy, 0, hysize*sizeof(Real));
		cudaMemset(pd[i].DxTzz, 0, hysize*sizeof(Real));
		cudaMemset(pd[i].DxTyz, 0, hysize*sizeof(Real));
		cudaMemset(pd[i].DyTxx, 0, hysize*sizeof(Real));
		cudaMemset(pd[i].DyTzz, 0, hysize*sizeof(Real));
		cudaMemset(pd[i].DyTxz, 0, hysize*sizeof(Real));
		cudaMemset(pd[i].DzTxx, 0, hysize*sizeof(Real));
		cudaMemset(pd[i].DzTyy, 0, hysize*sizeof(Real));
		cudaMemset(pd[i].DzTxy, 0, hysize*sizeof(Real));
#else		
		cudaMalloc( (Real**)&pd[i].DxTyy, fullsize*sizeof(Real) );//HG
		cudaMalloc( (Real**)&pd[i].DxTzz, fullsize*sizeof(Real) );//HG
		cudaMalloc( (Real**)&pd[i].DxTyz, fullsize*sizeof(Real) );//HG
		cudaMalloc( (Real**)&pd[i].DyTxx, fullsize*sizeof(Real) );//HG
		cudaMalloc( (Real**)&pd[i].DyTzz, fullsize*sizeof(Real) );//HG
		cudaMalloc( (Real**)&pd[i].DyTxz, fullsize*sizeof(Real) );//HG
		cudaMalloc( (Real**)&pd[i].DzTxx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DzTyy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DzTxy, fullsize*sizeof(Real) );
		cudaMemset(pd[i].DxTyy, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DxTzz, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DxTyz, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DyTxx, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DyTzz, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DyTxz, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DzTxx, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DzTyy, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DzTxy, 0, fullsize*sizeof(Real));
#endif
		
		cudaMalloc( (Real**)&pd[i].DxTxx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DxTxy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DxTxz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DxVx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DxVy, fullsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&pd[i].DxVz, fullsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc pd.dx\n",err, cudaGetErrorString(err) );
		
		cudaMemset(pd[i].DxTxx, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DxTxy, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DxTxz, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DxVx,  0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DxVy,  0, fullsize*sizeof(Real));
		err = cudaMemset(pd[i].DxVz,  0, fullsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at Memset Pd.Dx\n",err, cudaGetErrorString(err) );

		
		cudaMalloc( (Real**)&pd[i].DyTyy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DyTxy, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DyTyz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DyVx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DyVy, fullsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&pd[i].DyVz, fullsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc pd.dy\n",err, cudaGetErrorString(err) );
		
		cudaMemset(pd[i].DyTyy, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DyTxy, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DyTyz, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DyVx,  0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DyVy,  0, fullsize*sizeof(Real));
		err = cudaMemset(pd[i].DyVz,  0, fullsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at Memset Pd.Dy\n",err, cudaGetErrorString(err) );

		
		cudaMalloc( (Real**)&pd[i].DzTzz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DzTxz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DzTyz, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DzVx, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&pd[i].DzVy, fullsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&pd[i].DzVz, fullsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc pd.dz\n",err, cudaGetErrorString(err) );
		
		cudaMemset(pd[i].DzTzz, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DzTxz, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DzTyz, 0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DzVx,  0, fullsize*sizeof(Real));
		cudaMemset(pd[i].DzVy,  0, fullsize*sizeof(Real));
		err = cudaMemset(pd[i].DzVz,  0, fullsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at Memset Pd.Dz\n",err, cudaGetErrorString(err) );
		//-------------------------------------wavefield partial derivative--------------------------------------

		//-------------------------------------preprocessing pars------------------------------------------------
		//coordinate derivative
		cudaMalloc( (Real**)&drv[i].xix, fullsize*sizeof(Real) ); 
		cudaMalloc( (Real**)&drv[i].xiy, fullsize*sizeof(Real) ); 
		cudaMalloc( (Real**)&drv[i].xiz, fullsize*sizeof(Real) ); 
		cudaMalloc( (Real**)&drv[i].etax, fullsize*sizeof(Real) ); 
		cudaMalloc( (Real**)&drv[i].etay, fullsize*sizeof(Real) ); 
		cudaMalloc( (Real**)&drv[i].etaz, fullsize*sizeof(Real) ); 
		cudaMalloc( (Real**)&drv[i].zetax, fullsize*sizeof(Real) ); 
		cudaMalloc( (Real**)&drv[i].zetay, fullsize*sizeof(Real) ); 
		cudaMalloc( (Real**)&drv[i].zetaz, fullsize*sizeof(Real) ); 
		err = cudaMalloc( (Real**)&drv[i].jac, fullsize*sizeof(Real) ); 
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc drv\n",err, cudaGetErrorString(err) );

		cudaMemset( drv[i].xix, 0, fullsize*sizeof(Real) );
		cudaMemset( drv[i].xiy, 0, fullsize*sizeof(Real) );
		cudaMemset( drv[i].xiz, 0, fullsize*sizeof(Real) );
		cudaMemset( drv[i].etax, 0, fullsize*sizeof(Real) );
		cudaMemset( drv[i].etay, 0, fullsize*sizeof(Real) );
		cudaMemset( drv[i].etaz, 0, fullsize*sizeof(Real) );
		cudaMemset( drv[i].zetax, 0, fullsize*sizeof(Real) );
		cudaMemset( drv[i].zetay, 0, fullsize*sizeof(Real) );
		cudaMemset( drv[i].zetaz, 0, fullsize*sizeof(Real) );
		err = cudaMemset( drv[i].jac, 0, fullsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Memset Drv\n",err, cudaGetErrorString(err) );

		//media pars
		cudaMalloc( (Real**)&mpa[i].alpha, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&mpa[i].beta, fullsize*sizeof(Real) );
		cudaMalloc( (Real**)&mpa[i].rho, fullsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc mpa\n",err, cudaGetErrorString(err) );

		cudaMemset( mpa[i].alpha, 0, fullsize*sizeof(Real) );
		cudaMemset( mpa[i].beta, 0, fullsize*sizeof(Real) );
		err = cudaMemset( mpa[i].rho, 0, fullsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Memset mpa\n",err, cudaGetErrorString(err) );
		
		//force
		if(nfrc)
		{
			cudaMalloc( (int**)&frc[i].locx, nfrc*sizeof(int) );
			cudaMalloc( (int**)&frc[i].locy, nfrc*sizeof(int) );
			cudaMalloc( (int**)&frc[i].locz, nfrc*sizeof(int) );
			cudaMalloc( (Real**)&frc[i].fx, nfrc*sizeof(Real) );
			cudaMalloc( (Real**)&frc[i].fy, nfrc*sizeof(Real) );
			cudaMalloc( (Real**)&frc[i].fz, nfrc*sizeof(Real) );
			err = cudaMalloc( (Real**)&frc[i].stf, nfrc*nstf*sizeof(Real) );
			if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc frc\n",err, cudaGetErrorString(err) );
#ifdef SrcSmooth	
			cudaMalloc( (Real**)&frc[i].dnorm, nfrc*LenNorm*LenNorm*LenNorm*sizeof(Real) );
#endif
		}

		//moment
		if(nmnt)
		{
			cudaMalloc( (int**)&mnt[i].locx, nmnt*sizeof(int) );
			cudaMalloc( (int**)&mnt[i].locy, nmnt*sizeof(int) );
			cudaMalloc( (int**)&mnt[i].locz, nmnt*sizeof(int) );
			cudaMalloc( (Real**)&mnt[i].mxx, nmnt*sizeof(Real) );
			cudaMalloc( (Real**)&mnt[i].myy, nmnt*sizeof(Real) );
			cudaMalloc( (Real**)&mnt[i].mzz, nmnt*sizeof(Real) );
			cudaMalloc( (Real**)&mnt[i].mxy, nmnt*sizeof(Real) );
			cudaMalloc( (Real**)&mnt[i].mxz, nmnt*sizeof(Real) );
			cudaMalloc( (Real**)&mnt[i].myz, nmnt*sizeof(Real) );
			err = cudaMalloc( (Real**)&mnt[i].stf, nmnt*nstf*sizeof(Real) );
			if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc mnt\n",err, cudaGetErrorString(err) );
#ifdef SrcSmooth	
			cudaMalloc( (Real**)&mnt[i].dnorm, nmnt*LenNorm*LenNorm*LenNorm*sizeof(Real) );
#endif
		}

		//PeakVel 
		if(PVF)
		{
			cudaMalloc( (Real**)&Dpv[i].Vx, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*sizeof(Real) );
			cudaMalloc( (Real**)&Dpv[i].Vy, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*sizeof(Real) );
			err=cudaMalloc( (Real**)&Dpv[i].Vz, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*sizeof(Real) );
			if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc Dpv.vz\n",err, cudaGetErrorString(err) );
			
			cudaMemset( Dpv[i].Vx, 0, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*sizeof(Real) );
			cudaMemset( Dpv[i].Vy, 0, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*sizeof(Real) );
			err=cudaMemset( Dpv[i].Vz, 0, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*sizeof(Real) );
			if(err != 0) printf("err = %d, errS=%s, error may occur at memset Dpv.vz\n",err, cudaGetErrorString(err) );
		}

		
		//Velocity partial derivative conversion coefficient //should copperate with wave tensor accessing index.
		cudaMalloc( (Real**)&matVx2Vz[i], (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*SeisGeo*SeisGeo*sizeof(Real) );
		err = cudaMalloc( (Real**)&matVy2Vz[i], (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*SeisGeo*SeisGeo*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc Mat2Vy2Vz\n",err, cudaGetErrorString(err) );

		cudaMemset(matVx2Vz[i], 0, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*SeisGeo*SeisGeo*sizeof(Real));
		err = cudaMemset(matVy2Vz[i], 0, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*SeisGeo*SeisGeo*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset Mat2Vy2Vz\n",err, cudaGetErrorString(err) );
		//-------------------------------------preprocessing pars------------------------------------------------

		
		//------------------------------------absorb damping pars---------------------------------------------------
		err = cudaMalloc( (int**)&apr[i].nabs, SeisGeo*2*sizeof(int) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc apr\n",err, cudaGetErrorString(err) );
#ifdef CFSPML	
		//-----------------------------------------------------------
		//ADE wave field in X-dir
		cudaMalloc( (Real**)&Ax[i].Txx, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&Ax[i].Tyy, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&Ax[i].Tzz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&Ax[i].Txy, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&Ax[i].Txz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&Ax[i].Tyz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&Ax[i].Vx, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&Ax[i].Vy, axsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&Ax[i].Vz, axsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc Ax\n",err, cudaGetErrorString(err) );

		cudaMemset(Ax[i].Txx, 0, axsize*sizeof(Real));
		cudaMemset(Ax[i].Tyy, 0, axsize*sizeof(Real));
		cudaMemset(Ax[i].Tzz, 0, axsize*sizeof(Real));
		cudaMemset(Ax[i].Txy, 0, axsize*sizeof(Real));
		cudaMemset(Ax[i].Txz, 0, axsize*sizeof(Real));
		cudaMemset(Ax[i].Tyz, 0, axsize*sizeof(Real));
		cudaMemset(Ax[i].Vx, 0, axsize*sizeof(Real));
		cudaMemset(Ax[i].Vy, 0, axsize*sizeof(Real));
		err = cudaMemset(Ax[i].Vz, 0, axsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset Ax\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&mAx[i].Txx, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAx[i].Tyy, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAx[i].Tzz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAx[i].Txy, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAx[i].Txz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAx[i].Tyz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAx[i].Vx, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAx[i].Vy, axsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&mAx[i].Vz, axsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc mAx\n",err, cudaGetErrorString(err) );

		cudaMemset(mAx[i].Txx, 0, axsize*sizeof(Real));
		cudaMemset(mAx[i].Tyy, 0, axsize*sizeof(Real));
		cudaMemset(mAx[i].Tzz, 0, axsize*sizeof(Real));
		cudaMemset(mAx[i].Txy, 0, axsize*sizeof(Real));
		cudaMemset(mAx[i].Txz, 0, axsize*sizeof(Real));
		cudaMemset(mAx[i].Tyz, 0, axsize*sizeof(Real));
		cudaMemset(mAx[i].Vx, 0, axsize*sizeof(Real));
		cudaMemset(mAx[i].Vy, 0, axsize*sizeof(Real));
		err = cudaMemset(mAx[i].Vz, 0, axsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset mAx\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&hAx[i].Txx, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAx[i].Tyy, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAx[i].Tzz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAx[i].Txy, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAx[i].Txz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAx[i].Tyz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAx[i].Vx, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAx[i].Vy, axsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&hAx[i].Vz, axsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc hAx\n",err, cudaGetErrorString(err) );

		cudaMemset(hAx[i].Txx, 0, axsize*sizeof(Real));
		cudaMemset(hAx[i].Tyy, 0, axsize*sizeof(Real));
		cudaMemset(hAx[i].Tzz, 0, axsize*sizeof(Real));
		cudaMemset(hAx[i].Txy, 0, axsize*sizeof(Real));
		cudaMemset(hAx[i].Txz, 0, axsize*sizeof(Real));
		cudaMemset(hAx[i].Tyz, 0, axsize*sizeof(Real));
		cudaMemset(hAx[i].Vx, 0, axsize*sizeof(Real));
		cudaMemset(hAx[i].Vy, 0, axsize*sizeof(Real));
		err = cudaMemset(hAx[i].Vz, 0, axsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset hAx\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&tAx[i].Txx, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAx[i].Tyy, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAx[i].Tzz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAx[i].Txy, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAx[i].Txz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAx[i].Tyz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAx[i].Vx, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAx[i].Vy, axsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&tAx[i].Vz, axsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc tAx\n",err, cudaGetErrorString(err) );

		cudaMemset(tAx[i].Txx, 0, axsize*sizeof(Real));
		cudaMemset(tAx[i].Tyy, 0, axsize*sizeof(Real));
		cudaMemset(tAx[i].Tzz, 0, axsize*sizeof(Real));
		cudaMemset(tAx[i].Txy, 0, axsize*sizeof(Real));
		cudaMemset(tAx[i].Txz, 0, axsize*sizeof(Real));
		cudaMemset(tAx[i].Tyz, 0, axsize*sizeof(Real));
		cudaMemset(tAx[i].Vx, 0, axsize*sizeof(Real));
		cudaMemset(tAx[i].Vy, 0, axsize*sizeof(Real));
		err = cudaMemset(tAx[i].Vz, 0, axsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset tAx\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&FAx[i].Txx, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAx[i].Tyy, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAx[i].Tzz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAx[i].Txy, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAx[i].Txz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAx[i].Tyz, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAx[i].Vx, axsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAx[i].Vy, axsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&FAx[i].Vz, axsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc FAx\n",err, cudaGetErrorString(err) );

		cudaMemset(FAx[i].Txx, 0, axsize*sizeof(Real));
		cudaMemset(FAx[i].Tyy, 0, axsize*sizeof(Real));
		cudaMemset(FAx[i].Tzz, 0, axsize*sizeof(Real));
		cudaMemset(FAx[i].Txy, 0, axsize*sizeof(Real));
		cudaMemset(FAx[i].Txz, 0, axsize*sizeof(Real));
		cudaMemset(FAx[i].Tyz, 0, axsize*sizeof(Real));
		cudaMemset(FAx[i].Vx, 0, axsize*sizeof(Real));
		cudaMemset(FAx[i].Vy, 0, axsize*sizeof(Real));
		err = cudaMemset(FAx[i].Vz, 0, axsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset FAx\n",err, cudaGetErrorString(err) );

		//-----------------------------------------------------------
		//ADE wave field in Y-dir
		cudaMalloc( (Real**)&Ay[i].Txx, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&Ay[i].Tyy, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&Ay[i].Tzz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&Ay[i].Txy, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&Ay[i].Txz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&Ay[i].Tyz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&Ay[i].Vx, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&Ay[i].Vy, aysize*sizeof(Real) );
		err = cudaMalloc( (Real**)&Ay[i].Vz, aysize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc Ay.Vz\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&mAy[i].Txx, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&mAy[i].Tyy, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&mAy[i].Tzz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&mAy[i].Txy, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&mAy[i].Txz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&mAy[i].Tyz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&mAy[i].Vx, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&mAy[i].Vy, aysize*sizeof(Real) );
		err = cudaMalloc( (Real**)&mAy[i].Vz, aysize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at malloc mAy\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&hAy[i].Txx, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&hAy[i].Tyy, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&hAy[i].Tzz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&hAy[i].Txy, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&hAy[i].Txz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&hAy[i].Tyz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&hAy[i].Vx, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&hAy[i].Vy, aysize*sizeof(Real) );
		err = cudaMalloc( (Real**)&hAy[i].Vz, aysize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc hAy\n",err, cudaGetErrorString(err) );
		
		cudaMalloc( (Real**)&tAy[i].Txx, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&tAy[i].Tyy, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&tAy[i].Tzz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&tAy[i].Txy, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&tAy[i].Txz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&tAy[i].Tyz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&tAy[i].Vx, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&tAy[i].Vy, aysize*sizeof(Real) );
		err = cudaMalloc( (Real**)&tAy[i].Vz, aysize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at malloc tAy\n",err, cudaGetErrorString(err) );
		
		cudaMalloc( (Real**)&FAy[i].Txx, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&FAy[i].Tyy, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&FAy[i].Tzz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&FAy[i].Txy, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&FAy[i].Txz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&FAy[i].Tyz, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&FAy[i].Vx, aysize*sizeof(Real) );
		cudaMalloc( (Real**)&FAy[i].Vy, aysize*sizeof(Real) );
		err = cudaMalloc( (Real**)&FAy[i].Vz, aysize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc FAy\n",err, cudaGetErrorString(err) );
		
		cudaMemset(Ay[i].Txx, 0, aysize*sizeof(Real));
		cudaMemset(Ay[i].Tyy, 0, aysize*sizeof(Real));
		cudaMemset(Ay[i].Tzz, 0, aysize*sizeof(Real));
		cudaMemset(Ay[i].Txy, 0, aysize*sizeof(Real));
		cudaMemset(Ay[i].Txz, 0, aysize*sizeof(Real));
		cudaMemset(Ay[i].Tyz, 0, aysize*sizeof(Real));
		cudaMemset(Ay[i].Vx, 0, aysize*sizeof(Real));
		cudaMemset(Ay[i].Vy, 0, aysize*sizeof(Real));
		err = cudaMemset(Ay[i].Vz, 0, aysize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset Ay\n",err, cudaGetErrorString(err) );

		cudaMemset(mAy[i].Txx, 0, aysize*sizeof(Real));
		cudaMemset(mAy[i].Tyy, 0, aysize*sizeof(Real));
		cudaMemset(mAy[i].Tzz, 0, aysize*sizeof(Real));
		cudaMemset(mAy[i].Txy, 0, aysize*sizeof(Real));
		cudaMemset(mAy[i].Txz, 0, aysize*sizeof(Real));
		cudaMemset(mAy[i].Tyz, 0, aysize*sizeof(Real));
		cudaMemset(mAy[i].Vx, 0, aysize*sizeof(Real));
		cudaMemset(mAy[i].Vy, 0, aysize*sizeof(Real));
		err = cudaMemset(mAy[i].Vz, 0, aysize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset mAy\n",err, cudaGetErrorString(err) );

		cudaMemset(hAy[i].Txx, 0, aysize*sizeof(Real));
		cudaMemset(hAy[i].Tyy, 0, aysize*sizeof(Real));
		cudaMemset(hAy[i].Tzz, 0, aysize*sizeof(Real));
		cudaMemset(hAy[i].Txy, 0, aysize*sizeof(Real));
		cudaMemset(hAy[i].Txz, 0, aysize*sizeof(Real));
		cudaMemset(hAy[i].Tyz, 0, aysize*sizeof(Real));
		cudaMemset(hAy[i].Vx, 0, aysize*sizeof(Real));
		cudaMemset(hAy[i].Vy, 0, aysize*sizeof(Real));
		err = cudaMemset(hAy[i].Vz, 0, aysize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset mAy\n",err, cudaGetErrorString(err) );

		cudaMemset(tAy[i].Txx, 0, aysize*sizeof(Real));
		cudaMemset(tAy[i].Tyy, 0, aysize*sizeof(Real));
		cudaMemset(tAy[i].Tzz, 0, aysize*sizeof(Real));
		cudaMemset(tAy[i].Txy, 0, aysize*sizeof(Real));
		cudaMemset(tAy[i].Txz, 0, aysize*sizeof(Real));
		cudaMemset(tAy[i].Tyz, 0, aysize*sizeof(Real));
		cudaMemset(tAy[i].Vx, 0, aysize*sizeof(Real));
		cudaMemset(tAy[i].Vy, 0, aysize*sizeof(Real));
		err = cudaMemset(tAy[i].Vz, 0, aysize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset tAy\n",err, cudaGetErrorString(err) );

		cudaMemset(FAy[i].Txx, 0, aysize*sizeof(Real));
		cudaMemset(FAy[i].Tyy, 0, aysize*sizeof(Real));
		cudaMemset(FAy[i].Tzz, 0, aysize*sizeof(Real));
		cudaMemset(FAy[i].Txy, 0, aysize*sizeof(Real));
		cudaMemset(FAy[i].Txz, 0, aysize*sizeof(Real));
		cudaMemset(FAy[i].Tyz, 0, aysize*sizeof(Real));
		cudaMemset(FAy[i].Vx, 0, aysize*sizeof(Real));
		cudaMemset(FAy[i].Vy, 0, aysize*sizeof(Real));
		err = cudaMemset(FAy[i].Vz, 0, aysize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset FAy\n",err, cudaGetErrorString(err) );

		//-----------------------------------------------------------
		//ADE wave field in Z-dir
		cudaMalloc( (Real**)&Az[i].Txx, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&Az[i].Tyy, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&Az[i].Tzz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&Az[i].Txy, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&Az[i].Txz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&Az[i].Tyz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&Az[i].Vx, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&Az[i].Vy, azsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&Az[i].Vz, azsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at malloc Az\n",err, cudaGetErrorString(err) );
		
		cudaMalloc( (Real**)&mAz[i].Txx, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAz[i].Tyy, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAz[i].Tzz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAz[i].Txy, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAz[i].Txz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAz[i].Tyz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAz[i].Vx, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&mAz[i].Vy, azsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&mAz[i].Vz, azsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at malloc Az\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&hAz[i].Txx, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAz[i].Tyy, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAz[i].Tzz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAz[i].Txy, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAz[i].Txz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAz[i].Tyz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAz[i].Vx, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&hAz[i].Vy, azsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&hAz[i].Vz, azsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at malloc hAz\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&tAz[i].Txx, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAz[i].Tyy, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAz[i].Tzz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAz[i].Txy, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAz[i].Txz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAz[i].Tyz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAz[i].Vx, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&tAz[i].Vy, azsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&tAz[i].Vz, azsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at malloc tAz\n",err, cudaGetErrorString(err) );

		cudaMalloc( (Real**)&FAz[i].Txx, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAz[i].Tyy, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAz[i].Tzz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAz[i].Txy, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAz[i].Txz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAz[i].Tyz, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAz[i].Vx, azsize*sizeof(Real) );
		cudaMalloc( (Real**)&FAz[i].Vy, azsize*sizeof(Real) );
		err = cudaMalloc( (Real**)&FAz[i].Vz, azsize*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc FAz.Vz\n",err, cudaGetErrorString(err) );

		cudaMemset(Az[i].Txx, 0, azsize*sizeof(Real));
		cudaMemset(Az[i].Tyy, 0, azsize*sizeof(Real));
		cudaMemset(Az[i].Tzz, 0, azsize*sizeof(Real));
		cudaMemset(Az[i].Txy, 0, azsize*sizeof(Real));
		cudaMemset(Az[i].Txz, 0, azsize*sizeof(Real));
		cudaMemset(Az[i].Tyz, 0, azsize*sizeof(Real));
		cudaMemset(Az[i].Vx, 0, azsize*sizeof(Real));
		cudaMemset(Az[i].Vy, 0, azsize*sizeof(Real));
		err = cudaMemset(Az[i].Vz, 0, azsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset Az\n",err, cudaGetErrorString(err) );

		cudaMemset(mAz[i].Txx, 0, azsize*sizeof(Real));
		cudaMemset(mAz[i].Tyy, 0, azsize*sizeof(Real));
		cudaMemset(mAz[i].Tzz, 0, azsize*sizeof(Real));
		cudaMemset(mAz[i].Txy, 0, azsize*sizeof(Real));
		cudaMemset(mAz[i].Txz, 0, azsize*sizeof(Real));
		cudaMemset(mAz[i].Tyz, 0, azsize*sizeof(Real));
		cudaMemset(mAz[i].Vx, 0, azsize*sizeof(Real));
		cudaMemset(mAz[i].Vy, 0, azsize*sizeof(Real));
		err = cudaMemset(mAz[i].Vz, 0, azsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset mAz\n",err, cudaGetErrorString(err) );

		cudaMemset(hAz[i].Txx, 0, azsize*sizeof(Real));
		cudaMemset(hAz[i].Tyy, 0, azsize*sizeof(Real));
		cudaMemset(hAz[i].Tzz, 0, azsize*sizeof(Real));
		cudaMemset(hAz[i].Txy, 0, azsize*sizeof(Real));
		cudaMemset(hAz[i].Txz, 0, azsize*sizeof(Real));
		cudaMemset(hAz[i].Tyz, 0, azsize*sizeof(Real));
		cudaMemset(hAz[i].Vx, 0, azsize*sizeof(Real));
		cudaMemset(hAz[i].Vy, 0, azsize*sizeof(Real));
		err = cudaMemset(hAz[i].Vz, 0, azsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset hAz\n",err, cudaGetErrorString(err) );

		cudaMemset(tAz[i].Txx, 0, azsize*sizeof(Real));
		cudaMemset(tAz[i].Tyy, 0, azsize*sizeof(Real));
		cudaMemset(tAz[i].Tzz, 0, azsize*sizeof(Real));
		cudaMemset(tAz[i].Txy, 0, azsize*sizeof(Real));
		cudaMemset(tAz[i].Txz, 0, azsize*sizeof(Real));
		cudaMemset(tAz[i].Tyz, 0, azsize*sizeof(Real));
		cudaMemset(tAz[i].Vx, 0, azsize*sizeof(Real));
		cudaMemset(tAz[i].Vy, 0, azsize*sizeof(Real));
		err = cudaMemset(tAz[i].Vz, 0, azsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset tAz\n",err, cudaGetErrorString(err) );

		cudaMemset(FAz[i].Txx, 0, azsize*sizeof(Real));
		cudaMemset(FAz[i].Tyy, 0, azsize*sizeof(Real));
		cudaMemset(FAz[i].Tzz, 0, azsize*sizeof(Real));
		cudaMemset(FAz[i].Txy, 0, azsize*sizeof(Real));
		cudaMemset(FAz[i].Txz, 0, azsize*sizeof(Real));
		cudaMemset(FAz[i].Tyz, 0, azsize*sizeof(Real));
		cudaMemset(FAz[i].Vx, 0, azsize*sizeof(Real));
		cudaMemset(FAz[i].Vy, 0, azsize*sizeof(Real));
		err = cudaMemset(FAz[i].Vz, 0, azsize*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset FAz\n",err, cudaGetErrorString(err) );

		//---------------------------------------------------------------
		//CFS PML's par
		cudaMalloc( (Real**)&apr[i].APDx, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*sizeof(Real));
		cudaMalloc( (Real**)&apr[i].APDy, (Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*sizeof(Real));
		cudaMalloc( (Real**)&apr[i].APDz, cdx.nz*sizeof(Real));
		cudaMalloc( (Real**)&apr[i].Bx, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*sizeof(Real));
		cudaMalloc( (Real**)&apr[i].By, (Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*sizeof(Real));
		cudaMalloc( (Real**)&apr[i].Bz, cdx.nz*sizeof(Real));
		cudaMalloc( (Real**)&apr[i].DBx, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*sizeof(Real));
		cudaMalloc( (Real**)&apr[i].DBy, (Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*sizeof(Real));
		err = cudaMalloc( (Real**)&apr[i].DBz, cdx.nz*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at malloc apr.DBz\n",err, cudaGetErrorString(err) );
		cudaMalloc( (int**)&apr[i].CLoc, 26*6*sizeof(int) );
#else	
		//Sponge Layer's par
		cudaMalloc( (Real**)&apr[i].Ex, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*sizeof(Real) );
		cudaMalloc( (Real**)&apr[i].Ey, (Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*sizeof(Real) );
		cudaMalloc( (Real**)&apr[i].Ez, cdx.nz*sizeof(Real) );
		cudaMalloc( (int**)&apr[i].ELoc, 6*6*sizeof(int) );
#endif

		//------------------------------------absorb damping pars---------------------------------------------------

	}
	
	Kcheck( "cudaMalloc check in CP initialization of calculate program!" );
	
#ifdef MPI_DEBUG
int gdb_break=1;
while(gdb_break){};
#endif
	
	printf("---accomplished GPU device preparation work at Process[%d]\n",myid);

}

ChildProcs::~ChildProcs()
{
	fprintf(stdout,"into data free at Procs[%d],in calculate.cu\n",HostMpiRank);
	if(!Mflag)
	{
		for(int i=0;i<Cid.DNum;i++)
		{
			cudaSetDevice( Cid.Rank[i] );
			
#ifdef CFSPML		
			cudaFree(apr[i].CLoc);
			cudaFree(apr[i].DBz);	cudaFree(apr[i].DBy);	cudaFree(apr[i].DBx);
			cudaFree(apr[i].Bz);	cudaFree(apr[i].By);	cudaFree(apr[i].Bx);
			cudaFree(apr[i].APDz);	cudaFree(apr[i].APDy);	cudaFree(apr[i].APDx);

			//free ADE
			cudaFree(FAz[i].Vz);	cudaFree(FAz[i].Vy);	cudaFree(FAz[i].Vx);
			cudaFree(FAz[i].Tyz);	cudaFree(FAz[i].Txz);	cudaFree(FAz[i].Txy);
			cudaFree(FAz[i].Tzz);	cudaFree(FAz[i].Tyy);	cudaFree(FAz[i].Txx);

			cudaFree(tAz[i].Vz);	cudaFree(tAz[i].Vy);	cudaFree(tAz[i].Vx);
			cudaFree(tAz[i].Tyz);	cudaFree(tAz[i].Txz);	cudaFree(tAz[i].Txy);
			cudaFree(tAz[i].Tzz);	cudaFree(tAz[i].Tyy);	cudaFree(tAz[i].Txx);

			cudaFree(hAz[i].Vz);	cudaFree(hAz[i].Vy);	cudaFree(hAz[i].Vx);
			cudaFree(hAz[i].Tyz);	cudaFree(hAz[i].Txz);	cudaFree(hAz[i].Txy);
			cudaFree(hAz[i].Tzz);	cudaFree(hAz[i].Tyy);	cudaFree(hAz[i].Txx);

			cudaFree(mAz[i].Vz);	cudaFree(mAz[i].Vy);	cudaFree(mAz[i].Vx);
			cudaFree(mAz[i].Tyz);	cudaFree(mAz[i].Txz);	cudaFree(mAz[i].Txy);
			cudaFree(mAz[i].Tzz);	cudaFree(mAz[i].Tyy);	cudaFree(mAz[i].Txx);

			cudaFree(Az[i].Vz);	cudaFree(Az[i].Vy);	cudaFree(Az[i].Vx);
			cudaFree(Az[i].Tyz);	cudaFree(Az[i].Txz);	cudaFree(Az[i].Txy);
			cudaFree(Az[i].Tzz);	cudaFree(Az[i].Tyy);	cudaFree(Az[i].Txx);

			cudaFree(FAy[i].Vz);	cudaFree(FAy[i].Vy);	cudaFree(FAy[i].Vx);
			cudaFree(FAy[i].Tyz);	cudaFree(FAy[i].Txz);	cudaFree(FAy[i].Txy);
			cudaFree(FAy[i].Tzz);	cudaFree(FAy[i].Tyy);	cudaFree(FAy[i].Txx);

			cudaFree(tAy[i].Vz);	cudaFree(tAy[i].Vy);	cudaFree(tAy[i].Vx);
			cudaFree(tAy[i].Tyz);	cudaFree(tAy[i].Txz);	cudaFree(tAy[i].Txy);
			cudaFree(tAy[i].Tzz);	cudaFree(tAy[i].Tyy);	cudaFree(tAy[i].Txx);

			cudaFree(hAy[i].Vz);	cudaFree(hAy[i].Vy);	cudaFree(hAy[i].Vx);
			cudaFree(hAy[i].Tyz);	cudaFree(hAy[i].Txz);	cudaFree(hAy[i].Txy);
			cudaFree(hAy[i].Tzz);	cudaFree(hAy[i].Tyy);	cudaFree(hAy[i].Txx);

			cudaFree(mAy[i].Vz);	cudaFree(mAy[i].Vy);	cudaFree(mAy[i].Vx);
			cudaFree(mAy[i].Tyz);	cudaFree(mAy[i].Txz);	cudaFree(mAy[i].Txy);
			cudaFree(mAy[i].Tzz);	cudaFree(mAy[i].Tyy);	cudaFree(mAy[i].Txx);

			cudaFree(Ay[i].Vz);	cudaFree(Ay[i].Vy);	cudaFree(Ay[i].Vx);
			cudaFree(Ay[i].Tyz);	cudaFree(Ay[i].Txz);	cudaFree(Ay[i].Txy);
			cudaFree(Ay[i].Tzz);	cudaFree(Ay[i].Tyy);	cudaFree(Ay[i].Txx);

			cudaFree(FAx[i].Vz);	cudaFree(FAx[i].Vy);	cudaFree(FAx[i].Vx);
			cudaFree(FAx[i].Tyz);	cudaFree(FAx[i].Txz);	cudaFree(FAx[i].Txy);
			cudaFree(FAx[i].Tzz);	cudaFree(FAx[i].Tyy);	cudaFree(FAx[i].Txx);

			cudaFree(tAx[i].Vz);	cudaFree(tAx[i].Vy);	cudaFree(tAx[i].Vx);
			cudaFree(tAx[i].Tyz);	cudaFree(tAx[i].Txz);	cudaFree(tAx[i].Txy);
			cudaFree(tAx[i].Tzz);	cudaFree(tAx[i].Tyy);	cudaFree(tAx[i].Txx);

			cudaFree(hAx[i].Vz);	cudaFree(hAx[i].Vy);	cudaFree(hAx[i].Vx);
			cudaFree(hAx[i].Tyz);	cudaFree(hAx[i].Txz);	cudaFree(hAx[i].Txy);
			cudaFree(hAx[i].Tzz);	cudaFree(hAx[i].Tyy);	cudaFree(hAx[i].Txx);

			cudaFree(mAx[i].Vz);	cudaFree(mAx[i].Vy);	cudaFree(mAx[i].Vx);
			cudaFree(mAx[i].Tyz);	cudaFree(mAx[i].Txz);	cudaFree(mAx[i].Txy);
			cudaFree(mAx[i].Tzz);	cudaFree(mAx[i].Tyy);	cudaFree(mAx[i].Txx);

			cudaFree(Ax[i].Vz);	cudaFree(Ax[i].Vy);	cudaFree(Ax[i].Vx);
			cudaFree(Ax[i].Tyz);	cudaFree(Ax[i].Txz);	cudaFree(Ax[i].Txy);
			cudaFree(Ax[i].Tzz);	cudaFree(Ax[i].Tyy);	cudaFree(Ax[i].Txx);

#else		
			cudaFree(apr[i].ELoc);
			cudaFree(apr[i].Ez);	cudaFree(apr[i].Ey);	cudaFree(apr[i].Ex);
#endif		
			cudaFree(apr[i].nabs);

			cudaFree(matVy2Vz[i]);	cudaFree(matVx2Vz[i]);
			
			if(PVF)
			{
				cudaFree(Dpv[i].Vx); cudaFree(Dpv[i].Vy); cudaFree(Dpv[i].Vz);
			}

			if(fpn)
			{

				delete [] IM[i].mxx; delete [] IM[i].myy; delete [] IM[i].mzz;
				delete [] IM[i].mxy; delete [] IM[i].mxz; delete [] IM[i].myz;
#ifdef SrcSmooth	
				cudaFree(Rmnt[i].dnorm);
#endif
				cudaFree(Rmnt[i].myz);	cudaFree(Rmnt[i].mxz);	cudaFree(Rmnt[i].mxy);
				cudaFree(Rmnt[i].mzz);	cudaFree(Rmnt[i].myy);	cudaFree(Rmnt[i].mxx);
				cudaFree(Rmnt[i].locz);	cudaFree(Rmnt[i].locy);	cudaFree(Rmnt[i].locx);
			}

			if(nmnt)
			{
#ifdef SrcSmooth	
				cudaFree(mnt[i].dnorm);
#endif
				cudaFree(mnt[i].stf);
				cudaFree(mnt[i].myz);	cudaFree(mnt[i].mxz);	cudaFree(mnt[i].mxy);
				cudaFree(mnt[i].mzz);	cudaFree(mnt[i].myy);	cudaFree(mnt[i].mxx);
				cudaFree(mnt[i].locz);	cudaFree(mnt[i].locy);	cudaFree(mnt[i].locx);
			}

			if(nfrc)
			{
#ifdef SrcSmooth	
				cudaFree(frc[i].dnorm);
#endif
				cudaFree(frc[i].stf);
				cudaFree(frc[i].fz);	cudaFree(frc[i].fy);	cudaFree(frc[i].fx);
				cudaFree(frc[i].locz);	cudaFree(frc[i].locy);	cudaFree(frc[i].locx);
			}

			cudaFree(mpa[i].rho);	cudaFree(mpa[i].beta);	cudaFree(mpa[i].alpha);

			cudaFree(drv[i].jac);
			cudaFree(drv[i].zetaz);	cudaFree(drv[i].zetay);	cudaFree(drv[i].zetax);
			cudaFree(drv[i].etaz);	cudaFree(drv[i].etay);	cudaFree(drv[i].etax);
			cudaFree(drv[i].xiz);	cudaFree(drv[i].xiy);	cudaFree(drv[i].xix);
			
			cudaFree(pd[i].DzVz);	cudaFree(pd[i].DzVy);	cudaFree(pd[i].DzVx);
			cudaFree(pd[i].DzTyz);	cudaFree(pd[i].DzTxz);	cudaFree(pd[i].DzTxy);
			cudaFree(pd[i].DzTzz);	cudaFree(pd[i].DzTyy);	cudaFree(pd[i].DzTxx);

			cudaFree(pd[i].DyVz);	cudaFree(pd[i].DyVy);	cudaFree(pd[i].DyVx);
			cudaFree(pd[i].DyTyz);	cudaFree(pd[i].DyTxz);	cudaFree(pd[i].DyTxy);
			cudaFree(pd[i].DyTzz);	cudaFree(pd[i].DyTyy);	cudaFree(pd[i].DyTxx);

			cudaFree(pd[i].DxVz);	cudaFree(pd[i].DxVy);	cudaFree(pd[i].DxVx);
			cudaFree(pd[i].DxTyz);	cudaFree(pd[i].DxTxz);	cudaFree(pd[i].DxTxy);
			cudaFree(pd[i].DxTzz);	cudaFree(pd[i].DxTyy);	cudaFree(pd[i].DxTxx);

			cudaFree(tW[i].Vz);	cudaFree(tW[i].Vy);	cudaFree(tW[i].Vx);
			cudaFree(tW[i].Tyz);	cudaFree(tW[i].Txz);	cudaFree(tW[i].Txy);
			cudaFree(tW[i].Tzz);	cudaFree(tW[i].Tyy);	cudaFree(tW[i].Txx);

			cudaFree(hW[i].Vz);	cudaFree(hW[i].Vy);	cudaFree(hW[i].Vx);
			cudaFree(hW[i].Tyz);	cudaFree(hW[i].Txz);	cudaFree(hW[i].Txy);
			cudaFree(hW[i].Tzz);	cudaFree(hW[i].Tyy);	cudaFree(hW[i].Txx);

			cudaFree(mW[i].Vz);	cudaFree(mW[i].Vy);	cudaFree(mW[i].Vx);
			cudaFree(mW[i].Tyz);	cudaFree(mW[i].Txz);	cudaFree(mW[i].Txy);
			cudaFree(mW[i].Tzz);	cudaFree(mW[i].Tyy);	cudaFree(mW[i].Txx);

			cudaFree(W[i].Vz);	cudaFree(W[i].Vy);	cudaFree(W[i].Vx);
			cudaFree(W[i].Tyz);	cudaFree(W[i].Txz);	cudaFree(W[i].Txy);
			cudaFree(W[i].Tzz);	cudaFree(W[i].Tyy);	cudaFree(W[i].Txx);
			
			cudaFree(FW[i].Vz);	cudaFree(FW[i].Vy);	cudaFree(FW[i].Vx);
			cudaFree(FW[i].Tyz);	cudaFree(FW[i].Txz);	cudaFree(FW[i].Txy);
			cudaFree(FW[i].Tzz);	cudaFree(FW[i].Tyy);	cudaFree(FW[i].Txx);

			//delete [] h_FW[i].Vz;	delete [] h_FW[i].Vy;	delete [] h_FW[i].Vx;
			//delete [] h_FW[i].Tyz;	delete [] h_FW[i].Txz;	delete [] h_FW[i].Txy;
			//delete [] h_FW[i].Tzz;	delete [] h_FW[i].Tyy;	delete [] h_FW[i].Txx;
			
			if(fpn)
			{
				delete [] DFpt[i].locz;	delete [] DFpt[i].locy;	delete [] DFpt[i].locx;
				delete [] DFpt[i].Gsn;	delete [] DFpt[i].Rsn;
			}

			delete [] Dpt[i].locz;	delete [] Dpt[i].locy;	delete [] Dpt[i].locx;
			delete [] Dpt[i].Gsn;	delete [] Dpt[i].Rsn;

			cudaFree(D_Dpt[i].locz);	cudaFree(D_Dpt[i].locy);	cudaFree(D_Dpt[i].locx);
			cudaFree(D_Dpt[i].Gsn);	cudaFree(D_Dpt[i].Rsn);

#ifdef DevicePick			
			cudaFree(DPW[i].Tyz);	cudaFree(DPW[i].Txz);	cudaFree(DPW[i].Txy);
			cudaFree(DPW[i].Tzz);	cudaFree(DPW[i].Tyy);	cudaFree(DPW[i].Txx);
			cudaFree(DPW[i].Vz);	cudaFree(DPW[i].Vy);	cudaFree(DPW[i].Vx);
#endif

		}

		//free peak vel
		if(PVF)
		{
			delete [] Hpv.Vx; delete [] Hpv.Vy; delete [] Hpv.Vz;
		}

#ifndef PointOnly
		for(int j=0;j<nsnap;j++)
		{
			for(int i=0;i<Cid.DNum;i++)
			{
#ifdef DevicePick				
				if(HSpt[j].cmp==2 || HSpt[j].cmp==3)
				{
					cudaFree(DSW[j][i].Tyz);	cudaFree(DSW[j][i].Txz);	cudaFree(DSW[j][i].Txy);
					cudaFree(DSW[j][i].Tzz);	cudaFree(DSW[j][i].Tyy);	cudaFree(DSW[j][i].Txx);
				}
				if(HSpt[j].cmp==1 || HSpt[j].cmp==3)
				{
					cudaFree(DSW[j][i].Vz);	cudaFree(DSW[j][i].Vy);	cudaFree(DSW[j][i].Vx);
				}
#endif				

				cudaFree(D_DSpt[j][i].Gsn);	cudaFree(D_DSpt[j][i].Rsn);
				cudaFree(D_DSpt[j][i].locz);	cudaFree(D_DSpt[j][i].locy);	cudaFree(D_DSpt[j][i].locx);

				delete [] DSpt[j][i].Gsn;	delete [] DSpt[j][i].Rsn;
				delete [] DSpt[j][i].locz;	delete [] DSpt[j][i].locy;	delete [] DSpt[j][i].locx;
			}
			
			if(HSpt[j].cmp==2 || HSpt[j].cmp==3)
			{
				delete [] HSW[j].Tyz;	delete [] HSW[j].Txz;	delete [] HSW[j].Txy;
				delete [] HSW[j].Tzz;	delete [] HSW[j].Tyy;	delete [] HSW[j].Txx;
			}
			if(HSpt[j].cmp==1 || HSpt[j].cmp==3)
			{
				delete [] HSW[j].Vz;	delete [] HSW[j].Vy;	delete [] HSW[j].Vx;
			}

		}
#endif

		for(int i=0;i<Cid.DNum;i++)
			cudaDeviceReset();

		delete [] GD.Vz;	delete [] GD.Vy;	delete [] GD.Vx;
		delete [] GD.Tyz;	delete [] GD.Txz;	delete [] GD.Txy;	
		delete [] GD.Tzz;	delete [] GD.Tyy;	delete [] GD.Txx;

		delete [] HPW.Tyz;	delete [] HPW.Txz;	delete [] HPW.Txy;
		delete [] HPW.Tzz;	delete [] HPW.Tyy;	delete [] HPW.Txx;
		delete [] HPW.Vz;	delete [] HPW.Vy;	delete [] HPW.Vx;
		
		//host side focus buffer
		if(fpn)
		{
#ifdef SrcSmooth		
			delete [] H_Rmnt.dnorm;
#endif
			delete [] H_Rmnt.myz;	delete [] H_Rmnt.mxz;	delete [] H_Rmnt.mxy;
			delete [] H_Rmnt.mzz;	delete [] H_Rmnt.myy;	delete [] H_Rmnt.mxx;
			delete [] H_Rmnt.locz;	delete [] H_Rmnt.locy;	delete [] H_Rmnt.locx;
		}

		//point buffer
		delete [] Hpt.locz;	delete [] Hpt.locy;	delete [] Hpt.locx;
		delete [] Hpt.Gsn;	delete [] Hpt.Rsn;

		//snap buffer
		for(int i=0;i<nsnap;i++)
		{
			delete [] HSpt[i].Rsn;	delete [] HSpt[i].Gsn;
			delete [] HSpt[i].locx;	delete [] HSpt[i].locy;	delete [] HSpt[i].locz;
		}
		
		if(fpn)
		{
			//focus buffer
			delete [] HFpt.locz;	delete [] HFpt.locy;	delete [] HFpt.locx;
			delete [] HFpt.Gsn;	delete [] HFpt.Rsn;
		}

		//free host side struct array
#ifdef CFSPML
		delete [] FAz;	delete [] tAz;	delete [] hAz;	delete [] mAz;	delete [] Az;
		delete [] FAy;	delete [] tAy;	delete [] hAy;	delete [] mAy;	delete [] Ay;
		delete [] FAx;	delete [] tAx;	delete [] hAx;	delete [] mAx;	delete [] Ax;
#endif
		
		delete [] H_apr.nabs;

		delete [] IM;	delete [] Rmnt;	
		delete [] mnt;	delete [] frc;	delete [] apr;
		delete [] matVy2Vz;	delete [] matVx2Vz;
		delete [] mpa;	delete [] drv;
		delete [] pd;
		delete [] tW;	delete [] hW;	delete [] mW;	delete [] W;
		//delete [] h_FW;	
		delete [] FW;
		delete [] DPW;
		delete [] D_Dpt;	delete [] Dpt;
		if(fpn) delete [] DFpt;
		if(PVF) delete [] Dpv;

#ifndef PointOnly
		for(int i=0;i<nsnap;i++)
		{
			delete [] DSW[i];
			delete [] DSpt[i];	delete [] D_DSpt[i];
		}
		delete [] DSpt;	delete [] D_DSpt;
		delete [] DSW;	delete [] HSW;
#endif		
		delete [] HSpt;

		//free Cid pars
		delete [] BPG;
		delete [] Cid.yu;	delete [] Cid.yd;	delete [] Cid.xr;	delete [] Cid.xl;
		delete [] Cid.Size;	delete [] Cid.Rank;	delete [] Cid.np;	delete [] Cid.fp;
		for(int i=0;i<Cid.DNum;i++)
			delete [] Cid.Snp[i];
		delete [] Cid.Snp;
	
	}
	

	//free boundarybuffer
	delete [] IraB.Vz;	delete [] IraB.Vy;	delete [] IraB.Vx;
	delete [] IraB.Tyz;	delete [] IraB.Txz;	delete [] IraB.Txy;
	delete [] IraB.Tzz;	delete [] IraB.Tyy;	delete [] IraB.Txx;

	delete [] CSpn;

	fprintf(stdout,"data free at Procs[%d],in calculate.cu\n",HostMpiRank);
}

void ChildProcs::VelCoeff()
{
	//This subroutine used to compute trasfrom coeffcients
	int i;
	for(i=0;i<Cid.DNum;i++)
	{
		cudaSetDevice( Cid.Rank[i] );
		VelPDcoeff<<<BPG[i].y,BPG[i].x>>>(drv[i], mpa[i], apr[i], matVx2Vz[i], matVy2Vz[i]);
		
		CC( cudaDeviceSynchronize() );
	}
	fprintf(stdout,"accomplished VelPDcoeff computation at PCS[%d]\n",HostMpiRank);
}

void ChildProcs::RKite(int RKindex, int currT, int Xvec, int Yvec, int Zvec)
{
	//Note-1
	//forward and backward differential direction informations
	//the odd-order is opposite to even order for 4-Step Runge Kuuta 
	//(Thesis Equation 2.26 and Equation 2.37)
	// forward was represented by actual parameter value 1, 
	// and backward was represented by actual parameter value -1;
	
	//Note-2
	// in the RK begin part, use time increase 0
	// in the RK inner part, use time increase 1
	// in the RK final part, use time increase 2
	// time increase 0, 1 and 2 corresponding to the RK alpha time 0, 0.5, 0.5 and 1
	// at here it was used to extract force and moment source time function value

	//Note-3
	//before RK begin, in synchronization work, transfer FW to W,
	//at RK begin,
	//	first, calculate the space-domain partial derivative of W to get "pd"---P(T6V3)/P(xi,eta,zeta)
	//	then, mutiply the "pd" with covariant variables to get the time domain partial derivative hW ---P(T6V3)/Pt
	//	then, update tempral differential wave field---W by sum hW and FW
	//	and update final wave field---tW by sum hW and FW
	//at RK inner,
	//	this part execute two times
	//	first, differentiate W to get pd
	//	then, mutiply pd with covariant variables to get hW
	//	then, update W by sum hW and FW
	//	and update tW by sum tW and hW
	//at RK final,
	//	first, differentiate W to get pd
	//	then, mutiply pd with covariant variables to get hW
	//	then, update tW(will be represented by FW) by sum tW and hW
	//after RK final, transfer FW to outside and do storing works.
	//So,
	//	FW, keep unchanged, represent last big time step's wave field;
	//	W,  changes every iteration and use for next small step's differention work;
	//	hW, changed every iteration, represent this small step's wave filed;
	//	tW, changed every iteration, represent this big time step's final wave field;
	//	input FW, output FW


	int i;
	int tinc;
	int Tindex;
	Real time;
	Real alpha,beta;
	cudaError_t err;
	
	if(RKindex == 0)
	{
		tinc = 0;
		alpha = RK4A[0];
		beta = RK4B[0];
	}
	else if( RKindex == 1 || RKindex == 2 )
	{
		tinc = 1;
		alpha = RK4A[RKindex];
		beta = RK4B[RKindex];
	}
	else
	{
		tinc = 2;//RKindex=3
		beta = RK4B[3];
	}
	Tindex = 2*currT + tinc;//use for source and moment
	time = (currT + tinc*0.5)*stept;//use for focus
	
	//calculate current focal data{ Cid.fp[i] length }
	if(fpn && time!=InterpTime)
	{
		InterpFocus(time);
		InterpTime = time;
	}
	
	//for(i=0;i<0;i++)
	for(i=0;i<Cid.DNum;i++)
	{
		
		err = cudaSetDevice( Cid.Rank[i] );
		if(err!=0) printf("errS=%s, error may occur before RKite SetDev\n",cudaGetErrorString(err) );
		
#ifdef DisBug
if(i==0 && HostMpiRank==1&& currT>=0)
{		
printf("\n at Procs[%d].Dev[%d]----->CurrtTime= %d RKindex=%d, tinc=%d, time=%f, Tindex= %d, RKite starts(%d,%d,%d),"
	"load mnt=%d frc=%d, focus=%d input flags=%d, %d, %d refer to ",
	HostMpiRank,Cid.Rank[i], currT, RKindex, tinc, time, Tindex,	
	zbx,zby,zbz,this->nmnt,this->nfrc, Cid.fp[i], Xvec,Yvec,Zvec);//AAA
if(Xvec==1) printf("F"); else printf("B");
if(Yvec==1) printf("F"); else printf("B");
if(Zvec==1) printf("F\n"); else printf("B\n");
}
#endif

		//-----------------------------------verification---------mpa/drv/wave field  transport---------------
		//generatewave<<<BPG[i],ThreadPerBlock>>>(FW[i], currT, Cstart);
		//generatewave<<<BlockPerGrid,ThreadPerBlock>>>(FW[i], currT, Cstart);
		//err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error at PCS [%d] Lanuch Perform/Generate\n",err, cudaGetErrorString(err), HostMpiRank );
		
		if(this->HyGrid)
		{
		CalDiff<<<BPG[i],ThreadPerBlock>>>(Xvec, Yvec, Zvec, ConIndex, steph, matVx2Vz[i], matVy2Vz[i], mW[i], pd[i]);//mW used as W
		err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch HG-DiffCL\n",err, cudaGetErrorString(err) );
		cudaDeviceSynchronize();

		CalWave<<<BPG[i],ThreadPerBlock>>>(ConIndex, drv[i], mpa[i], pd[i], apr[i], matVx2Vz[i], matVy2Vz[i], hW[i],
							    mAx[i], hAx[i], mAy[i], hAy[i], mAz[i], hAz[i]);
		err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch HG-WaveCL\n",err, cudaGetErrorString(err) );
		cudaDeviceSynchronize();
		}
		else
		{
		CalDiffCL<<<BPG[i],ThreadPerBlock>>>(Xvec, Yvec, Zvec, ConIndex, steph, matVx2Vz[i], matVy2Vz[i], mW[i], pd[i]);//mW used as W
		err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch DiffCL\n",err, cudaGetErrorString(err) );
		cudaDeviceSynchronize();

		CalWaveCL<<<BPG[i],ThreadPerBlock>>>(ConIndex, drv[i], mpa[i], pd[i], apr[i], matVx2Vz[i], matVy2Vz[i], hW[i],
							    mAx[i], hAx[i], mAy[i], hAy[i], mAz[i], hAz[i]);
		err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch WaveCL\n",err, cudaGetErrorString(err) );
		cudaDeviceSynchronize();
		}
			
		
#ifdef CondFreeTIMG
		CalTIMG<<<BPG[i].y,BPG[i].x>>>(Xvec,Yvec,Zvec, steph, mpa[i].rho, drv[i], mW[i], hW[i], 
							   mAx[i], hAx[i], mAy[i], hAy[i], apr[i]);//mW used as W
		err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch TIMG\n",err, cudaGetErrorString(err) );
		cudaDeviceSynchronize();
#endif

#ifdef CondFreeVUCD
		//wrong
		CalVUCD<<<BPG[i].y,BPG[i].x>>>(Xvec,Yvec,Zvec, steph, matVx2Vz[i], matVy2Vz[i], mpa[i], drv[i], mW[i], hW[i],
							mAx[i], hAx[i], mAy[i], hAy[i], apr[i]);//mW used as W
		err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch VUCD\n",err, cudaGetErrorString(err) );
		cudaDeviceSynchronize();
#endif

		err = cudaGetLastError(); 
		if(err!=0)printf("err=%d, errS=%s, error check after source at PCS[%d]Dev[%d](%d,%d,%d)\n",
				err, cudaGetErrorString(err),HostMpiRank,Cid.Rank[i],this->nfrc,this->nmnt,Cid.fp[i]);
		
		if(this->nfrc)
		{
			LoadForce<<<BPG[i],ThreadPerBlock>>>(Tindex, cdx, steph, nfrc, nstf, frc[i], drv[i].jac, mpa[i].rho, hW[i]);
			err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch FORCE\n",err, cudaGetErrorString(err) );
			cudaDeviceSynchronize();
		}

		if(this->nmnt)
		{
			LoadMoment<<<BPG[i],ThreadPerBlock>>>(Tindex, cdx, steph, nmnt, nstf, mnt[i], drv[i].jac, hW[i]);
			err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch MOM\n",err, cudaGetErrorString(err) );
			cudaDeviceSynchronize();
		}
		
		if(Cid.fp[i])
		{
			LoadRmom<<<BPG[i],ThreadPerBlock>>>(cdx, steph, Cid.fp[i], Rmnt[i], drv[i].jac, hW[i]);
			err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch MOM\n",err, cudaGetErrorString(err) );
			cudaDeviceSynchronize();
		}

		err = cudaGetLastError(); 
		if(err!=0)printf("err=%d, errS=%s, error check after source at PCS[%d]Dev[%d](%d,%d,%d)\n",
				err, cudaGetErrorString(err),HostMpiRank,Cid.Rank[i],this->nfrc,this->nmnt,Cid.fp[i]);
		
		if(RKindex ==0)
		{
			IterationBegin<<<BPG[i],ThreadPerBlock>>>(stept, alpha, beta, FW[i], hW[i], tW[i], W[i],
				apr[i].nabs,FAx[i], hAx[i], tAx[i], Ax[i], 	FAy[i], hAy[i], tAy[i], Ay[i], 	FAz[i], hAz[i], tAz[i], Az[i]);
			err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch IteBegin\n",err, cudaGetErrorString(err) );
		}
		else if(RKindex==1 || RKindex==2)
		{
			IterationInner<<<BPG[i],ThreadPerBlock>>>(stept, alpha, beta, FW[i], hW[i], tW[i], W[i],
				apr[i].nabs,FAx[i], hAx[i], tAx[i], Ax[i], 	FAy[i], hAy[i], tAy[i],Ay[i], 	FAz[i], hAz[i], tAz[i], Az[i]);
			err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch IteInn\n",err, cudaGetErrorString(err) );
		}
		else
		{
			if(PVF)
			{//apply extract peak velocity from W
			IterationFinalPV<<<BPG[i],ThreadPerBlock>>>(stept, beta, Dpv[i], hW[i], tW[i], W[i],
				apr[i].nabs,hAx[i], tAx[i], Ax[i], 	hAy[i], tAy[i],Ay[i], 	hAz[i],tAz[i], Az[i]);
			err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch IteFin\n",err, cudaGetErrorString(err) );
			}
			else
			{
			IterationFinal<<<BPG[i],ThreadPerBlock>>>(stept, beta, hW[i], tW[i], W[i],
				apr[i].nabs,hAx[i], tAx[i], Ax[i], 	hAy[i], tAy[i],Ay[i], 	hAz[i],tAz[i], Az[i]);
			err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after Lanuch IteFin\n",err, cudaGetErrorString(err) );
			}
#ifdef withABS
#ifndef CFSPML			
			AbsExp<<<BPG[i],ThreadPerBlock>>>(apr[i].Ex, apr[i].Ey, apr[i].Ez, apr[i].nabs, W[i]);
			err = cudaGetLastError(); if(err!=0)printf("err=%d, errS=%s, error may occur after AbsExp\n",err, cudaGetErrorString(err) );
#endif		
#endif		
		}



		CC( cudaDeviceSynchronize() );//OK no problem
			
	}
	
	//(*currT)++;//simulate one step forward

	

}

void ChildProcs::GatherData(wfield HOST, wfield *DEVICE, int kind)
{
	//from seperate device to full node
	//fullsize = Csize*cdx.ny*cdx.nz;//seperate node-size, Csize = Cxn + 2*LenFD
	//fullsize = (Cid.xr[i]-Cid.xl[i]+1+2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1+2*LenFD)*cdx.nz;//valid device-size with boundary

	//kind = 1, cudaMemcpyHostToDevice;	H2D	to scatter computing parameters such as drv,mpa etc.
	//kind = 2, cudaMemcpyDeviceToHost;	D2H	to gather wavefield.
	//kind = 3, cudaMemcpyDeviceToDevice;	D2D	to be continue.

	int i;
	int idx,idy;
	int Rindex,Gindex;
	
	for(i=0;i<Cid.DNum;i++)
	{
		cudaSetDevice( Cid.Rank[i] );

		for(idx=Cid.xl[i];idx<=Cid.xr[i];idx++)
			for(idy=Cid.yd[i];idy<=Cid.yu[i];idy++)
			{
				Gindex = idx*cdx.ny*cdx.nz + idy*cdx.nz;
				Rindex = (idx-Cid.xl[i]+LenFD)*(Cid.yu[i]-Cid.yd[i]+1+2*LenFD)*cdx.nz + (idy-Cid.yd[i]+LenFD)*cdx.nz;

				//cudaMemcpy(GD.Vx+Gindex, FW[i].Vx+Rindex, cdx.nz*sizeof(Real), cudaMemcpyDeviceToHost);
				
				if(kind==1)
				{
					cudaMemcpy(DEVICE[i].Vx+Rindex, HOST.Vx+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
					cudaMemcpy(DEVICE[i].Vy+Rindex, HOST.Vy+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
					cudaMemcpy(DEVICE[i].Vz+Rindex, HOST.Vz+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
					cudaMemcpy(DEVICE[i].Txx+Rindex, HOST.Txx+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
					cudaMemcpy(DEVICE[i].Tyy+Rindex, HOST.Tyy+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
					cudaMemcpy(DEVICE[i].Tzz+Rindex, HOST.Tzz+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
					cudaMemcpy(DEVICE[i].Txy+Rindex, HOST.Txy+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
					cudaMemcpy(DEVICE[i].Txz+Rindex, HOST.Txz+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
					cudaMemcpy(DEVICE[i].Tyz+Rindex, HOST.Tyz+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
				}
				else if(kind==2)
				{
					cudaMemcpy(HOST.Vx+Gindex, DEVICE[i].Vx+Rindex, cdx.nz*sizeof(Real), cudaMemcpyDeviceToHost);
					cudaMemcpy(HOST.Vy+Gindex, DEVICE[i].Vy+Rindex, cdx.nz*sizeof(Real), cudaMemcpyDeviceToHost);
					cudaMemcpy(HOST.Vz+Gindex, DEVICE[i].Vz+Rindex, cdx.nz*sizeof(Real), cudaMemcpyDeviceToHost);
					cudaMemcpy(HOST.Txx+Gindex, DEVICE[i].Txx+Rindex, cdx.nz*sizeof(Real), cudaMemcpyDeviceToHost);
					cudaMemcpy(HOST.Tyy+Gindex, DEVICE[i].Tyy+Rindex, cdx.nz*sizeof(Real), cudaMemcpyDeviceToHost);
					cudaMemcpy(HOST.Tzz+Gindex, DEVICE[i].Tzz+Rindex, cdx.nz*sizeof(Real), cudaMemcpyDeviceToHost);
					cudaMemcpy(HOST.Txy+Gindex, DEVICE[i].Txy+Rindex, cdx.nz*sizeof(Real), cudaMemcpyDeviceToHost);
					cudaMemcpy(HOST.Txz+Gindex, DEVICE[i].Txz+Rindex, cdx.nz*sizeof(Real), cudaMemcpyDeviceToHost);
					cudaMemcpy(HOST.Tyz+Gindex, DEVICE[i].Tyz+Rindex, cdx.nz*sizeof(Real), cudaMemcpyDeviceToHost);
				}
				else
					fprintf(stdout,"WARNING: doing D2D node-size level Gather/Scatter works(kind=%d)\n",kind);

			}

	
	}
	//printf("Procs[%d], pass GatherData()\n",HostMpiRank);
}

void ChildProcs::SynTopo()
{
	//cudaMalloc( (Real**)&matVx2Vz[i], (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*SeisGeo*SeisGeo*sizeof(Real) );
	//cudaMallocManaged( (Real**)&matVx2Vz, Csize*cdx.ny*SeisGeo*SeisGeo*sizeof(Real) );//should copperate with wave tensor accessing index.

	int i,j,k;
	int idx,idy;
	int Rindex,Gindex;

	Real *MXZ,*MYZ;
	MXZ = new Real [Csize*cdx.ny*SeisGeo*SeisGeo]();
	MYZ = new Real [Csize*cdx.ny*SeisGeo*SeisGeo]();
	
	for(i=0;i<Cid.DNum;i++)
	{
		cudaSetDevice( Cid.Rank[i] );

		printf("myid=%d,Cstart=%d, Csize=%d,cdx.ny=%d,range=(%d, %d, %d, %d)\n",HostMpiRank,Cstart,Csize,cdx.ny,Cid.xl[i],Cid.xr[i],Cid.yd[i],Cid.yu[i]);
		
		for(idx=Cid.xl[i];idx<=Cid.xr[i];idx++)
			for(idy=Cid.yd[i];idy<=Cid.yu[i];idy++)
			{
				Gindex = idx*cdx.ny*SeisGeo*SeisGeo + idy*SeisGeo*SeisGeo;
				Rindex = (idx-Cid.xl[i]+LenFD)*(Cid.yu[i]-Cid.yd[i]+1+2*LenFD)*SeisGeo*SeisGeo + (idy-Cid.yd[i]+LenFD)*SeisGeo*SeisGeo;
				
				cudaMemcpy(MXZ+Gindex, matVx2Vz[i]+Rindex, SeisGeo*SeisGeo*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy(MYZ+Gindex, matVy2Vz[i]+Rindex, SeisGeo*SeisGeo*sizeof(Real), cudaMemcpyDeviceToHost);
			}
	}
	
	for(i=0;i<Csize;i++)
		for(j=0;j<cdx.ny;j++)
		{
			printf("mXz[%d][%d]=",i+Cstart,j);
			for(k=0;k<SeisGeo*SeisGeo;k++)
			{
				Gindex = i*cdx.ny*SeisGeo*SeisGeo+j*SeisGeo*SeisGeo+k;
				printf("%g\t",MXZ[Gindex]);
			}
			cout<<endl;
		}

	
	delete [] MYZ;
	delete [] MXZ;

}

void ChildProcs::SynPV()
{
	if(PVF==0)//does not apply peak vel extraction
		return;

	//Hpv.Vx = new Real[Csize*cdx.ny](); Hpv.Vy = new Real[Csize*cdx.ny](); Hpv.Vz = new Real[Csize*cdx.ny]();
	//cudaMalloc( (Real**)&Dpv[i].Vx, (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*sizeof(Real) );

	int i;
	int idx;
	int Src,Dst,size;

	
	for(i=0;i<Cid.DNum;i++)
	{
		size = Cid.yu[i]-Cid.yd[i]+1;

		cudaSetDevice( Cid.Rank[i] );

		//printf("myid=%d,Cstart=%d, Csize=%d,cdx.ny=%d,range=(%d, %d, %d, %d)\n",HostMpiRank,Cstart,Csize,cdx.ny,Cid.xl[i],Cid.xr[i],Cid.yd[i],Cid.yu[i]);
		
		for(idx=Cid.xl[i];idx<=Cid.xr[i];idx++)
		{
			Src = (idx-Cid.xl[i]+LenFD)*(Cid.yu[i]-Cid.yd[i]+1+2*LenFD) + LenFD;
			Dst = idx*cdx.ny + Cid.yd[i];
			//Dst = idx*cdx.ny + LenFD;

			cudaMemcpy(Hpv.Vx+Dst, Dpv[i].Vx+Src, size*sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(Hpv.Vy+Dst, Dpv[i].Vy+Src, size*sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(Hpv.Vz+Dst, Dpv[i].Vz+Src, size*sizeof(Real), cudaMemcpyDeviceToHost);
		}
		
	}
	
}

void ChildProcs::SynData()
{//only for display seperate device use
	//from seperate device to host, device size
	
	int i;
	int size;
	
	for(i=0;i<Cid.DNum;i++)
	{
		cudaSetDevice( Cid.Rank[i] );
		
		size = (Cid.xr[i]-Cid.xl[i]+1+2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1+2*LenFD)*cdx.nz;//valid device-size with boundary
	
		cudaMemcpy(h_FW[i].Vx, FW[i].Vx, size*sizeof(Real), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_FW[i].Vy, FW[i].Vy, size*sizeof(Real), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_FW[i].Vz, FW[i].Vz, size*sizeof(Real), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_FW[i].Txx, FW[i].Txx, size*sizeof(Real), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_FW[i].Tyy, FW[i].Tyy, size*sizeof(Real), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_FW[i].Tzz, FW[i].Tzz, size*sizeof(Real), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_FW[i].Txy, FW[i].Txy, size*sizeof(Real), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_FW[i].Txz, FW[i].Txz, size*sizeof(Real), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_FW[i].Tyz, FW[i].Tyz, size*sizeof(Real), cudaMemcpyDeviceToHost);
	
	}

	//printf("Procs[%d], pass SynData()\n",HostMpiRank);
}

void ChildProcs::ShareData()
{
	int i,j,k;
	int srcDev,dstDev;
	int srcLoc,dstLoc;
	int size;
	cudaError_t err;
	
	//printf("totally device number is %d, Ycolumn=%d, Xcolumn=%d\n",Cid.DNum,Cid.ydim,Cid.xdim);
	//fullsize = (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*cdx.nz;//with boundary device-size
	//Processing in X-direction, delivering Y
	//forward-dir

	for(i=0;i<Cid.xdim-1;i++)
	{
		for(j=0;j<Cid.ydim;j++)
		{
			dstDev = i*Cid.ydim+j;
			srcDev = (i+1)*Cid.ydim+j;
			//printf("PCS[%d]set device[%d][%d](%d)<-----------get from device[%d][%d](%d):  size=%d, from %d, to %d"
			//	"\tinDST[%d]: xl=%d xr=%d yd=%d yu=%d"
			//	"\tinSRC[%d]: xl=%d xr=%d yd=%d yu=%d\n",
			//	HostMpiRank,i,j,dstDev,i+1,j,srcDev,
			//	Cid.yu[dstDev]-Cid.yd[dstDev]+1+2*LenFD, LenFD, LenFD + Cid.xr[dstDev]-Cid.xl[dstDev]+1,
			//	dstDev,Cid.xl[dstDev],Cid.xr[dstDev],Cid.yd[dstDev],Cid.yu[dstDev],
			//	srcDev,Cid.xl[srcDev],Cid.xr[srcDev],Cid.yd[srcDev],Cid.yu[srcDev]);
			
			size = (Cid.yu[dstDev]-Cid.yd[dstDev]+1+2*LenFD)*cdx.nz;//should be same in X dir, single silce of NY*NZ
			srcLoc = LenFD*size;//inner of left
			dstLoc = (LenFD + Cid.xr[dstDev]-Cid.xl[dstDev]+1)*size;//outter of right

			cudaSetDevice( Cid.Rank[dstDev] );
			cudaMemcpyPeer( W[dstDev].Vx+dstLoc,  Cid.Rank[dstDev], W[srcDev].Vx+srcLoc,  Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Vy+dstLoc,  Cid.Rank[dstDev], W[srcDev].Vy+srcLoc,  Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Vz+dstLoc,  Cid.Rank[dstDev], W[srcDev].Vz+srcLoc,  Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Txx+dstLoc, Cid.Rank[dstDev], W[srcDev].Txx+srcLoc, Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Tyy+dstLoc, Cid.Rank[dstDev], W[srcDev].Tyy+srcLoc, Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Tzz+dstLoc, Cid.Rank[dstDev], W[srcDev].Tzz+srcLoc, Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Txy+dstLoc, Cid.Rank[dstDev], W[srcDev].Txy+srcLoc, Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Txz+dstLoc, Cid.Rank[dstDev], W[srcDev].Txz+srcLoc, Cid.Rank[srcDev], LenFD*size*sizeof(Real));
		  err = cudaMemcpyPeer( W[dstDev].Tyz+dstLoc, Cid.Rank[dstDev], W[srcDev].Tyz+srcLoc, Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			if(err != 0) printf("err = %d, errS=%s, error may occur in all of X-dir forward delivering part\n",err, cudaGetErrorString(err) );
			
			cudaDeviceSynchronize();
			
		}
	}
	
	//backward-dir
	for(i=Cid.xdim-1;i>0;i--)
	{
		for(j=0;j<Cid.ydim;j++)
		{
			dstDev = i*Cid.ydim+j;
			srcDev = (i-1)*Cid.ydim+j;
			
			//printf("PCS[%d]set device[%d][%d](%d)<-----------get from device[%d][%d](%d):  size=%d, from %d, to %d"
			//	"\tinDST[%d]: xl=%d xr=%d yd=%d yu=%d"
			//	"\tinSRC[%d]: xl=%d xr=%d yd=%d yu=%d\n",
			//	HostMpiRank,i,j,dstDev,i-1,j,srcDev,
			//	Cid.yu[dstDev]-Cid.yd[dstDev]+1+2*LenFD, Cid.xr[dstDev]-Cid.xl[dstDev]+1, 0,
			//	dstDev,Cid.xl[dstDev],Cid.xr[dstDev],Cid.yd[dstDev],Cid.yu[dstDev],
			//	srcDev,Cid.xl[srcDev],Cid.xr[srcDev],Cid.yd[srcDev],Cid.yu[srcDev]);
			
			size = (Cid.yu[dstDev]-Cid.yd[dstDev]+1+2*LenFD)*cdx.nz;//should be same in X dir, single silce of NY*NZ
			srcLoc = (Cid.xr[srcDev]-Cid.xl[srcDev]+1)*size;//inner of right
			dstLoc = 0;//outter of left

			err = cudaSetDevice( Cid.Rank[dstDev] );
			if(err != 0) printf("err = %d, errS=%s error may occur in all of X-dir backward DevSet\n",err, cudaGetErrorString(err) );
			cudaMemcpyPeer( W[dstDev].Vx+dstLoc,  Cid.Rank[dstDev], W[srcDev].Vx+srcLoc,  Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Vy+dstLoc,  Cid.Rank[dstDev], W[srcDev].Vy+srcLoc,  Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Vz+dstLoc,  Cid.Rank[dstDev], W[srcDev].Vz+srcLoc,  Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Txx+dstLoc, Cid.Rank[dstDev], W[srcDev].Txx+srcLoc, Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Tyy+dstLoc, Cid.Rank[dstDev], W[srcDev].Tyy+srcLoc, Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Tzz+dstLoc, Cid.Rank[dstDev], W[srcDev].Tzz+srcLoc, Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Txy+dstLoc, Cid.Rank[dstDev], W[srcDev].Txy+srcLoc, Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			cudaMemcpyPeer( W[dstDev].Txz+dstLoc, Cid.Rank[dstDev], W[srcDev].Txz+srcLoc, Cid.Rank[srcDev], LenFD*size*sizeof(Real));
		  err = cudaMemcpyPeer( W[dstDev].Tyz+dstLoc, Cid.Rank[dstDev], W[srcDev].Tyz+srcLoc, Cid.Rank[srcDev], LenFD*size*sizeof(Real));
			if(err != 0) printf("err = %d, errS=%s error may occur in all of X-dir backward delivering part\n",err, cudaGetErrorString(err) );
			cudaDeviceSynchronize();
			
		}
	}
	
	//Processing in Y-direction, delivering X
	//forward-dir
	for(i=0;i<Cid.xdim;i++)
	{
		for(j=0;j<Cid.ydim-1;j++)
		{
			dstDev = i*Cid.ydim+j;
			srcDev = i*Cid.ydim+j+1;
			
			//seperate delivering in X
			for(k=0;k<Cid.xr[dstDev]-Cid.xl[dstDev]+1+2*LenFD;k++)
			{
				size = LenFD*cdx.nz;//every X term have LenFDs Y terms.
				srcLoc = k*(Cid.yu[srcDev]-Cid.yd[srcDev]+1+2*LenFD)*cdx.nz + size;//inner of bottom
				dstLoc = k*(Cid.yu[dstDev]-Cid.yd[dstDev]+1+2*LenFD)*cdx.nz + (LenFD + Cid.yu[dstDev]-Cid.yd[dstDev]+1)*cdx.nz;//outter of top

				err = cudaSetDevice( Cid.Rank[dstDev] );
				if(err != 0) printf("err = %d, errS=%s error may occur in all of Y-dir forward SetDev\n",err, cudaGetErrorString(err) );
				cudaMemcpyPeer( W[dstDev].Vx+dstLoc,  Cid.Rank[dstDev], W[srcDev].Vx+srcLoc,  Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Vy+dstLoc,  Cid.Rank[dstDev], W[srcDev].Vy+srcLoc,  Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Vz+dstLoc,  Cid.Rank[dstDev], W[srcDev].Vz+srcLoc,  Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Txx+dstLoc, Cid.Rank[dstDev], W[srcDev].Txx+srcLoc, Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Tyy+dstLoc, Cid.Rank[dstDev], W[srcDev].Tyy+srcLoc, Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Tzz+dstLoc, Cid.Rank[dstDev], W[srcDev].Tzz+srcLoc, Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Txy+dstLoc, Cid.Rank[dstDev], W[srcDev].Txy+srcLoc, Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Txz+dstLoc, Cid.Rank[dstDev], W[srcDev].Txz+srcLoc, Cid.Rank[srcDev], size*sizeof(Real));
			  err = cudaMemcpyPeer( W[dstDev].Tyz+dstLoc, Cid.Rank[dstDev], W[srcDev].Tyz+srcLoc, Cid.Rank[srcDev], size*sizeof(Real));
				if(err != 0) printf("err = %d, errS=%s error may occur in all of Y-dir forward delivering part\n",err, cudaGetErrorString(err) );
				
				cudaDeviceSynchronize();

			}
		}
	}
	
	//backward-dir
	for(i=0;i<Cid.xdim;i++)
	{
		for(j=Cid.ydim-1;j>0;j--)
		{
			dstDev = i*Cid.ydim+j;
			srcDev = i*Cid.ydim+j-1;
			//printf("set device[%d][%d](%d)<-----------get from device[%d][%d](%d)\n",i,j,dstDev,i,j-1,srcDev);
			
			//seperate delivering in X
			for(k=0;k<Cid.xr[dstDev]-Cid.xl[dstDev]+1+2*LenFD;k++)
			{
				size = LenFD*cdx.nz;//every X term have LenFDs Y terms.
				srcLoc = k*(Cid.yu[srcDev]-Cid.yd[srcDev]+1+2*LenFD)*cdx.nz + (Cid.yu[srcDev]-Cid.yd[srcDev]+1)*cdx.nz;//inner of top
				dstLoc = k*(Cid.yu[dstDev]-Cid.yd[dstDev]+1+2*LenFD)*cdx.nz + 0;//outter of bottom

				err = cudaSetDevice( Cid.Rank[dstDev] );
				if(err != 0) printf("err = %d, errS=%s error may occur in all of Y-dir backward SetDev\n",err, cudaGetErrorString(err) );
				cudaMemcpyPeer( W[dstDev].Vx+dstLoc,  Cid.Rank[dstDev], W[srcDev].Vx+srcLoc,  Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Vy+dstLoc,  Cid.Rank[dstDev], W[srcDev].Vy+srcLoc,  Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Vz+dstLoc,  Cid.Rank[dstDev], W[srcDev].Vz+srcLoc,  Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Txx+dstLoc, Cid.Rank[dstDev], W[srcDev].Txx+srcLoc, Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Tyy+dstLoc, Cid.Rank[dstDev], W[srcDev].Tyy+srcLoc, Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Tzz+dstLoc, Cid.Rank[dstDev], W[srcDev].Tzz+srcLoc, Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Txy+dstLoc, Cid.Rank[dstDev], W[srcDev].Txy+srcLoc, Cid.Rank[srcDev], size*sizeof(Real));
				cudaMemcpyPeer( W[dstDev].Txz+dstLoc, Cid.Rank[dstDev], W[srcDev].Txz+srcLoc, Cid.Rank[srcDev], size*sizeof(Real));
			  err = cudaMemcpyPeer( W[dstDev].Tyz+dstLoc, Cid.Rank[dstDev], W[srcDev].Tyz+srcLoc, Cid.Rank[srcDev], size*sizeof(Real));
				if(err != 0) printf("err = %d, errS=%s error may occur in all of Y-dir backward delivering part\n",err, cudaGetErrorString(err) );
				
				cudaDeviceSynchronize();

			}
		}
	}
	
	
}

void ChildProcs::IntraBoundGS(int GSflag)
{
	int i,j,k;
	int Dev;
	int size;
	int srcLoc,dstLoc;
	
	//before ShareData copy valid Y
	//after ShareData copy full Y, interweave
	
	if(GSflag)
	{
		//gather
		//from inner of Dev-left to IraB-left
		i = 0;
		for(j=0;j<Cid.ydim;j++)
		{
			Dev = i*Cid.ydim+j;
			cudaSetDevice( Cid.Rank[Dev] );
			size = (Cid.yu[Dev]-Cid.yd[Dev]+1)*cdx.nz;
			for(k=0;k<LenFD;k++)
			{
				srcLoc = (LenFD+k)*(Cid.yu[Dev]-Cid.yd[Dev]+1+2*LenFD)*cdx.nz + LenFD*cdx.nz;
				dstLoc = k*cdx.ny*cdx.nz + Cid.yd[Dev]*cdx.nz;

				cudaMemcpy( IraB.Vx+dstLoc,  W[Dev].Vx+srcLoc,  size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Vy+dstLoc,  W[Dev].Vy+srcLoc,  size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Vz+dstLoc,  W[Dev].Vz+srcLoc,  size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Txx+dstLoc, W[Dev].Txx+srcLoc, size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Tyy+dstLoc, W[Dev].Tyy+srcLoc, size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Tzz+dstLoc, W[Dev].Tzz+srcLoc, size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Txy+dstLoc, W[Dev].Txy+srcLoc, size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Txz+dstLoc, W[Dev].Txz+srcLoc, size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Tyz+dstLoc, W[Dev].Tyz+srcLoc, size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
			}
		}

		//from inner of Dev-right to IraB-right
		i = Cid.xdim-1;
		for(j=0;j<Cid.ydim;j++)
		{
			Dev = i*Cid.ydim+j;
			cudaSetDevice( Cid.Rank[Dev] );
			size = (Cid.yu[Dev]-Cid.yd[Dev]+1)*cdx.nz;
			for(k=0;k<LenFD;k++)
			{
				srcLoc = (Cid.xr[Dev]-Cid.xl[Dev]+1+k)*(Cid.yu[Dev]-Cid.yd[Dev]+1+2*LenFD)*cdx.nz + LenFD*cdx.nz;
				dstLoc = (LenFD+k)*cdx.ny*cdx.nz + Cid.yd[Dev]*cdx.nz;

				cudaMemcpy( IraB.Vx+dstLoc,  W[Dev].Vx+srcLoc,  size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Vy+dstLoc,  W[Dev].Vy+srcLoc,  size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Vz+dstLoc,  W[Dev].Vz+srcLoc,  size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Txx+dstLoc, W[Dev].Txx+srcLoc, size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Tyy+dstLoc, W[Dev].Tyy+srcLoc, size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Tzz+dstLoc, W[Dev].Tzz+srcLoc, size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Txy+dstLoc, W[Dev].Txy+srcLoc, size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Txz+dstLoc, W[Dev].Txz+srcLoc, size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaMemcpy( IraB.Tyz+dstLoc, W[Dev].Tyz+srcLoc, size*sizeof(Real), cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
			}
		}
	}
	else
	{
		//scatter
		//from IraB-left to outter of Dev_left
		i = 0;
		for(j=0;j<Cid.ydim;j++)
		{
			Dev = i*Cid.ydim+j;
			cudaSetDevice( Cid.Rank[Dev] );
			size = (Cid.yu[Dev]-Cid.yd[Dev]+1+2*LenFD)*cdx.nz;
			for(k=0;k<LenFD;k++)
			{
				srcLoc = k*cdx.ny*cdx.nz + (Cid.yd[Dev]-LenFD)*cdx.nz;
				dstLoc = k*(Cid.yu[Dev]-Cid.yd[Dev]+1+2*LenFD)*cdx.nz;

				cudaMemcpy( W[Dev].Vx+dstLoc,  IraB.Vx+srcLoc,  size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Vy+dstLoc,  IraB.Vy+srcLoc,  size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Vz+dstLoc,  IraB.Vz+srcLoc,  size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Txx+dstLoc, IraB.Txx+srcLoc, size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Tyy+dstLoc, IraB.Tyy+srcLoc, size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Tzz+dstLoc, IraB.Tzz+srcLoc, size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Txy+dstLoc, IraB.Txy+srcLoc, size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Txz+dstLoc, IraB.Txz+srcLoc, size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Tyz+dstLoc, IraB.Tyz+srcLoc, size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
			}
		}

		//from IraB-right to outter of Dev-right
		i = Cid.xdim-1;
		for(j=0;j<Cid.ydim;j++)
		{
			Dev = i*Cid.ydim+j;
			cudaSetDevice( Cid.Rank[Dev] );
			size = (Cid.yu[Dev]-Cid.yd[Dev]+1+2*LenFD)*cdx.nz;
			for(k=0;k<LenFD;k++)
			{
				srcLoc = (LenFD+k)*cdx.ny*cdx.nz + (Cid.yd[Dev]-LenFD)*cdx.nz;
				dstLoc = (Cid.xr[Dev]-Cid.xl[Dev]+1+LenFD+k)*(Cid.yu[Dev]-Cid.yd[Dev]+1+2*LenFD)*cdx.nz;

				cudaMemcpy( W[Dev].Vx+dstLoc,  IraB.Vx+srcLoc,  size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Vy+dstLoc,  IraB.Vy+srcLoc,  size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Vz+dstLoc,  IraB.Vz+srcLoc,  size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Txx+dstLoc, IraB.Txx+srcLoc, size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Tyy+dstLoc, IraB.Tyy+srcLoc, size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Tzz+dstLoc, IraB.Tzz+srcLoc, size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Txy+dstLoc, IraB.Txy+srcLoc, size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Txz+dstLoc, IraB.Txz+srcLoc, size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaMemcpy( W[Dev].Tyz+dstLoc, IraB.Tyz+srcLoc, size*sizeof(Real), cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
			}
		}
	}

}

void ChildProcs::ParH2D()
{
	//from node to device, distribute parameters, free host buffer
	//full size drv/mpa and special size apr,frc,mnt

	//fullsize = Csize*cdx.ny*cdx.nz;//seperate node-size, Csize = Cxn + 2*LenFD
	//fullsize = (Cid.xr[i]-Cid.xl[i]+1+2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1+2*LenFD)*cdx.nz;//valid device-size with boundary

	//kind = 1, cudaMemcpyHostToDevice;	H2D	to scatter computing parameters such as drv,mpa etc.
	//kind = 2, cudaMemcpyDeviceToHost;	D2H	to gather wavefield.
	//kind = 3, cudaMemcpyDeviceToDevice;	D2D	to be continue.
	
	cudaError_t err;
	int i;
	int idx,idy;
	int Rindex,Gindex;

	fprintf(stdout,"Procs[%d] into parH2D\n",HostMpiRank);
	
	for(i=0;i<Cid.DNum;i++)
	{
		cudaSetDevice( Cid.Rank[i] );

		for(idx=Cid.xl[i]-LenFD;idx<=Cid.xr[i]+LenFD;idx++)
			for(idy=Cid.yd[i]-LenFD;idy<=Cid.yu[i]+LenFD;idy++)
			{//copy full device size
				Gindex = idx*cdx.ny*cdx.nz + idy*cdx.nz;
				Rindex = (idx-Cid.xl[i]+LenFD)*(Cid.yu[i]-Cid.yd[i]+1+2*LenFD)*cdx.nz + (idy-Cid.yd[i]+LenFD)*cdx.nz;

				//drv;
				err = cudaMemcpy(drv[i].xix+Rindex,   H_drv.xix+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d mmecpy drv_xix\n",err, cudaGetErrorString(err) );
				err = cudaMemcpy(drv[i].xiy+Rindex,   H_drv.xiy+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d mmecpy drv_xiy\n",err, cudaGetErrorString(err) );
				err = cudaMemcpy(drv[i].xiz+Rindex,   H_drv.xiz+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d mmecpy drv_xiz\n",err, cudaGetErrorString(err) );
				err = cudaMemcpy(drv[i].etax+Rindex,  H_drv.etax+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d mmecpy drv_etax\n",err, cudaGetErrorString(err) );
				err = cudaMemcpy(drv[i].etay+Rindex,  H_drv.etay+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d mmecpy drv_etay\n",err, cudaGetErrorString(err) );
				err = cudaMemcpy(drv[i].etaz+Rindex,  H_drv.etaz+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d mmecpy drv_etaz\n",err, cudaGetErrorString(err) );
				err = cudaMemcpy(drv[i].zetax+Rindex, H_drv.zetax+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d mmecpy drv_zetax\n",err, cudaGetErrorString(err) );
				err = cudaMemcpy(drv[i].zetay+Rindex, H_drv.zetay+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d mmecpy drv_zetay\n",err, cudaGetErrorString(err) );
				err = cudaMemcpy(drv[i].zetaz+Rindex, H_drv.zetaz+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d mmecpy drv_zetaz\n",err, cudaGetErrorString(err) );
				err = cudaMemcpy(drv[i].jac+Rindex,   H_drv.jac+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy jac\n",err, cudaGetErrorString(err) );

				//mpa
				err = cudaMemcpy(mpa[i].alpha+Rindex, H_mpa.alpha+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy alpha\n",err, cudaGetErrorString(err) );
				err = cudaMemcpy(mpa[i].beta+Rindex,  H_mpa.beta+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy beta\n",err, cudaGetErrorString(err) );
				err = cudaMemcpy(mpa[i].rho+Rindex,   H_mpa.rho+Gindex, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
			if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy rho\n",err, cudaGetErrorString(err) );

				cudaDeviceSynchronize();
			}
		
		//frc
		if(nfrc)
		{
			cudaMemcpy(frc[i].locx, H_frc.locx, nfrc*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(frc[i].locy, H_frc.locy, nfrc*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(frc[i].locz, H_frc.locz, nfrc*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(frc[i].fx,   H_frc.fx,   nfrc*sizeof(Real), cudaMemcpyHostToDevice);
			cudaMemcpy(frc[i].fy,   H_frc.fy,   nfrc*sizeof(Real), cudaMemcpyHostToDevice);
			cudaMemcpy(frc[i].fz,   H_frc.fz,   nfrc*sizeof(Real), cudaMemcpyHostToDevice);
			cudaMemcpy(frc[i].stf,  H_frc.stf,  nfrc*nstf*sizeof(Real), cudaMemcpyHostToDevice);
#ifdef SrcSmooth
			cudaMemcpy(frc[i].dnorm,  H_frc.dnorm,  nfrc*LenNorm*LenNorm*LenNorm*sizeof(Real), cudaMemcpyHostToDevice);
#endif
			cudaDeviceSynchronize();
		}

		//mnt
		if(nmnt)
		{
			cudaMemcpy(mnt[i].locx, H_mnt.locx, nmnt*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(mnt[i].locy, H_mnt.locy, nmnt*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(mnt[i].locz, H_mnt.locz, nmnt*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(mnt[i].mxx,  H_mnt.mxx,  nmnt*sizeof(Real), cudaMemcpyHostToDevice);
			cudaMemcpy(mnt[i].myy,  H_mnt.myy,  nmnt*sizeof(Real), cudaMemcpyHostToDevice);
			cudaMemcpy(mnt[i].mzz,  H_mnt.mzz,  nmnt*sizeof(Real), cudaMemcpyHostToDevice);
			cudaMemcpy(mnt[i].mxy,  H_mnt.mxy,  nmnt*sizeof(Real), cudaMemcpyHostToDevice);
			cudaMemcpy(mnt[i].mxz,  H_mnt.mxz,  nmnt*sizeof(Real), cudaMemcpyHostToDevice);
			cudaMemcpy(mnt[i].myz,  H_mnt.myz,  nmnt*sizeof(Real), cudaMemcpyHostToDevice);
			cudaMemcpy(mnt[i].stf,  H_mnt.stf,  nmnt*nstf*sizeof(Real), cudaMemcpyHostToDevice);
#ifdef SrcSmooth
			cudaMemcpy(mnt[i].dnorm,  H_mnt.dnorm,  nmnt*LenNorm*LenNorm*LenNorm*sizeof(Real), cudaMemcpyHostToDevice);
#endif

			cudaDeviceSynchronize();
		}

		//apr
		err = cudaMemcpy(apr[i].nabs, H_apr.nabs, SeisGeo*2*sizeof(int), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy apr nabs\n",err, cudaGetErrorString(err) );
#ifdef CFSPML
		err = cudaMemcpy(apr[i].APDx, H_apr.APDx+Cid.xl[i]-LenFD, (Cid.xr[i]-Cid.xl[i]+1+2*LenFD)*sizeof(Real), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy apr cfl apdx par\n",err, cudaGetErrorString(err) );
		err = cudaMemcpy(apr[i].APDy, H_apr.APDy+Cid.yd[i]-LenFD, (Cid.yu[i]-Cid.yd[i]+1+2*LenFD)*sizeof(Real), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy apr cfl apdy par\n",err, cudaGetErrorString(err) );
		err = cudaMemcpy(apr[i].APDz, H_apr.APDz, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy apr cfl apdz par\n",err, cudaGetErrorString(err) );
		err = cudaMemcpy(apr[i].Bx,   H_apr.Bx+Cid.xl[i]-LenFD, (Cid.xr[i]-Cid.xl[i]+1+2*LenFD)*sizeof(Real), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy apr cfl bx par\n",err, cudaGetErrorString(err) );
		err = cudaMemcpy(apr[i].By,   H_apr.By+Cid.yd[i]-LenFD, (Cid.yu[i]-Cid.yd[i]+1+2*LenFD)*sizeof(Real), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy apr cfl by par\n",err, cudaGetErrorString(err) );
		err = cudaMemcpy(apr[i].Bz,   H_apr.Bz, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy apr cfl Bz par\n",err, cudaGetErrorString(err) );
		err = cudaMemcpy(apr[i].DBx,  H_apr.DBx+Cid.xl[i]-LenFD, (Cid.xr[i]-Cid.xl[i]+1+2*LenFD)*sizeof(Real), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy apr cfl dx par\n",err, cudaGetErrorString(err) );
		err = cudaMemcpy(apr[i].DBy,  H_apr.DBy+Cid.yd[i]-LenFD, (Cid.yu[i]-Cid.yd[i]+1+2*LenFD)*sizeof(Real), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy apr cfl dy par\n",err, cudaGetErrorString(err) );
		err = cudaMemcpy(apr[i].DBz,  H_apr.DBz, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy apr cfl DB par\n",err, cudaGetErrorString(err) );
		err = cudaMemcpy(apr[i].CLoc, H_apr.CLoc, 26*6*sizeof(int), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at parh2d memcpy apr cfl cloc par\n",err, cudaGetErrorString(err) );
#else
		cudaMemcpy(apr[i].Ex,   H_apr.Ex+Cid.xl[i]-LenFD, (Cid.xr[i]-Cid.xl[i]+1+2*LenFD)*sizeof(Real), cudaMemcpyHostToDevice);
		cudaMemcpy(apr[i].Ey,   H_apr.Ey+Cid.yd[i]-LenFD, (Cid.yu[i]-Cid.yd[i]+1+2*LenFD)*sizeof(Real), cudaMemcpyHostToDevice);
		cudaMemcpy(apr[i].Ez,   H_apr.Ez, cdx.nz*sizeof(Real), cudaMemcpyHostToDevice);
		cudaMemcpy(apr[i].ELoc, H_apr.ELoc, 6*6*sizeof(int), cudaMemcpyHostToDevice);
#endif
		cudaDeviceSynchronize();
	
	}

	//free host par buffer
#ifdef CFSPML
	delete [] H_apr.CLoc;
	delete [] H_apr.DBz;	delete [] H_apr.DBy;	delete [] H_apr.DBx;
	delete [] H_apr.Bz;	delete [] H_apr.By;	delete [] H_apr.Bx;
	delete [] H_apr.APDz;	delete [] H_apr.APDy;	delete [] H_apr.APDx;
#else
	delete [] H_apr.ELoc;
	delete [] H_apr.Ez;	delete [] H_apr.Ey;	delete [] H_apr.Ex;
#endif
	//delete [] H_apr.nabs;
	
	//can not free focus here, becaues it will used in every step

	if(nmnt)
	{
#ifdef SrcSmooth		
		delete [] H_mnt.dnorm;
#endif
		delete [] H_mnt.stf;
		delete [] H_mnt.myz;	delete [] H_mnt.mxz;	delete [] H_mnt.mxy;
		delete [] H_mnt.mzz;	delete [] H_mnt.myy;	delete [] H_mnt.mxx;
		delete [] H_mnt.locz;	delete [] H_mnt.locy;	delete [] H_mnt.locx;
	}
	
	if(nfrc)
	{
#ifdef SrcSmooth		
		delete [] H_frc.dnorm;
#endif
		delete [] H_frc.stf;
		delete [] H_frc.fz;	delete [] H_frc.fy;	delete [] H_frc.fx;
		delete [] H_frc.locz;	delete [] H_frc.locy;	delete [] H_frc.locx;
	}

	delete [] H_mpa.rho;	delete [] H_mpa.beta;	delete [] H_mpa.alpha;

	delete [] H_drv.jac;
	delete [] H_drv.zetaz;	delete [] H_drv.zetay;	delete [] H_drv.zetax;
	delete [] H_drv.etaz;	delete [] H_drv.etay;	delete [] H_drv.etax;
	delete [] H_drv.xiz;	delete [] H_drv.xiy;	delete [] H_drv.xix;

	fprintf(stdout,"Procs[%d], pass ParH2D()\n",HostMpiRank);
}

void ChildProcs::C2DSnapPick()
{
#ifndef PointOnly
	
	int i,j,k;
	int numD;
	int nTime;
	cudaError_t err;
/*	
	for(i=0;i<nsnap;i++)
		for(j=0;j<CSpn[i];j++)
			printf("inCal-Snapshot[%d],PCS[%d]->Rsn[%4d],Gsn[%4d]->(%3d,%3d,%3d),tinv=%d,cmp=%d\n",i+1,HostMpiRank,
				HSpt[i].Rsn[j],HSpt[i].Gsn[j],HSpt[i].locx[j],HSpt[i].locy[j],HSpt[i].locz[j],HSpt[i].tinv,HSpt[i].cmp);
*/	

	for(i=0;i<nsnap;i++)
	{
		for(k=0;k<Cid.DNum;k++)
		{
			numD=0;
			for(j=0;j<CSpn[i];j++)
			{
				if(HSpt[i].locx[j]>=Cid.xl[k]+Cstart && HSpt[i].locx[j]<=Cid.xr[k]+Cstart &&
				   HSpt[i].locy[j]>=Cid.yd[k] && HSpt[i].locy[j]<=Cid.yu[k])
				{
					numD++;
				}
			}
			Cid.Snp[k][i] = numD;
			//printf("inCal-Snap[%d],PCS[%d],Dev[%d],have point %d(CID->%d), in range(%d,%d)and(%d,%d)\n",
			//	i+1,HostMpiRank,k,numD,Cid.Snp[k][i],Cid.xl[k],Cid.xr[k],Cid.yd[k],Cid.yu[k]);
			
			DSpt[i][k].Rsn = new int[numD]();	DSpt[i][k].Gsn = new int[numD]();
			DSpt[i][k].locx = new int[numD]();	DSpt[i][k].locy = new int[numD]();	DSpt[i][k].locz = new int[numD]();

			err = cudaSetDevice(Cid.Rank[k]);
			if(err != 0) printf("err = %d, errS=%s, error may occur at setdev\n",err, cudaGetErrorString(err) );
			cudaMalloc( (int**)&D_DSpt[i][k].Rsn, numD*sizeof(int) );
			cudaMalloc( (int**)&D_DSpt[i][k].Gsn, numD*sizeof(int) );
			cudaMalloc( (int**)&D_DSpt[i][k].locx, numD*sizeof(int) );
			cudaMalloc( (int**)&D_DSpt[i][k].locy, numD*sizeof(int) );
			err = cudaMalloc( (int**)&D_DSpt[i][k].locz, numD*sizeof(int) );
			if(err != 0) printf("err = %d, errS=%s, error may occur at malloc\n",err, cudaGetErrorString(err) );
		}

	}

	for(i=0;i<nsnap;i++)
	{
		for(k=0;k<Cid.DNum;k++)
		{
			numD=0;
			for(j=0;j<CSpn[i];j++)
			{
				if(HSpt[i].locx[j]>=Cid.xl[k]+Cstart && HSpt[i].locx[j]<=Cid.xr[k]+Cstart &&
				   HSpt[i].locy[j]>=Cid.yd[k] && HSpt[i].locy[j]<=Cid.yu[k])
				{
					DSpt[i][k].Rsn[numD] = numD;
					DSpt[i][k].Gsn[numD] = HSpt[i].Rsn[j];
					DSpt[i][k].locx[numD] = HSpt[i].locx[j];
					DSpt[i][k].locy[numD] = HSpt[i].locy[j];
					DSpt[i][k].locz[numD] = HSpt[i].locz[j];
					numD++;
				}
			}
			DSpt[i][k].tinv = HSpt[i].tinv;
			DSpt[i][k].cmp = HSpt[i].cmp;
			
			if(Cid.Snp[k][i])
			{
				//printf("MemcpyStep:snap%d,pcs%d,dev%d,numD=%d\n",i+1,HostMpiRank,k,numD);
				err = cudaSetDevice(Cid.Rank[k]);
				if(err != 0) printf("err = %d, errS=%s, error may occur at setdev\n",err, cudaGetErrorString(err) );
				cudaMemcpy(D_DSpt[i][k].Rsn, DSpt[i][k].Rsn, numD*sizeof(int), cudaMemcpyHostToDevice);	
				cudaMemcpy(D_DSpt[i][k].Gsn, DSpt[i][k].Gsn, numD*sizeof(int), cudaMemcpyHostToDevice);	
				cudaMemcpy(D_DSpt[i][k].locx, DSpt[i][k].locx, numD*sizeof(int), cudaMemcpyHostToDevice);	
				cudaMemcpy(D_DSpt[i][k].locy, DSpt[i][k].locy, numD*sizeof(int), cudaMemcpyHostToDevice);	
				err = cudaMemcpy(D_DSpt[i][k].locz, DSpt[i][k].locz, numD*sizeof(int), cudaMemcpyHostToDevice);	
				if(err != 0) printf("err = %d, errS=%s, error may occur at memcpy\n",err, cudaGetErrorString(err) );
				D_DSpt[i][k].tinv = DSpt[i][k].tinv;
				D_DSpt[i][k].cmp = DSpt[i][k].cmp;//transfer to device side par struct, do Value-Trans when use
			}
		}

	}
	
	
	//HSW and DSW allocation
	for(i=0;i<nsnap;i++)
	{
		nTime = ceil(1.0*this->nt/HSpt[i].tinv);
		//printf("for snap[%d],PCS[%d]->HSW holds time points as %d and spatial point as %d \n",i+1,HostMpiRank,nTime,CSpn[i]);
		
		if(HSpt[i].cmp==1 || HSpt[i].cmp==3)
		{
			HSW[i].Vx = new Real[ nTime*CSpn[i] ]();	HSW[i].Vy = new Real[ nTime*CSpn[i] ]();	HSW[i].Vz = new Real[ nTime*CSpn[i] ]();
		}
		if(HSpt[i].cmp==2 || HSpt[i].cmp==3)
		{
			HSW[i].Txx = new Real[ nTime*CSpn[i] ](); 	HSW[i].Tyy = new Real[ nTime*CSpn[i] ]();	HSW[i].Tzz = new Real[ nTime*CSpn[i] ]();
			HSW[i].Txy = new Real[ nTime*CSpn[i] ]();	HSW[i].Txz = new Real[ nTime*CSpn[i] ]();	HSW[i].Tyz = new Real[ nTime*CSpn[i] ]();
		}

#ifdef DevicePick
		for(k=0;k<Cid.DNum;k++)
		{
			//printf("inDev-Snap[%d],PCS[%d],Dev[%d]->HSW holds time points as %d and spatial point as %d \n",i+1,HostMpiRank,k,nTime,Cid.Snp[k][i]);
			if(Cid.Snp[k][i])
			{
				err = cudaSetDevice(Cid.Rank[k]);
				if(err != 0) printf("err = %d, errS=%s, error may occur at setdev\n",err, cudaGetErrorString(err) );
				if(HSpt[i].cmp==1 || HSpt[i].cmp==3)
				{
					cudaMalloc( (Real**)&DSW[i][k].Vx, sizeof(Real)*nTime*Cid.Snp[k][i]);	
					cudaMalloc( (Real**)&DSW[i][k].Vy, sizeof(Real)*nTime*Cid.Snp[k][i]);	
					err = cudaMalloc( (Real**)&DSW[i][k].Vz, sizeof(Real)*nTime*Cid.Snp[k][i]);
					if(err != 0) printf("err = %d, errS=%s, error may occur at malloc 1\n",err, cudaGetErrorString(err) );
				}
				if(HSpt[i].cmp==2 || HSpt[i].cmp==3)
				{
					cudaMalloc( (Real**)&DSW[i][k].Txx, sizeof(Real)*nTime*Cid.Snp[k][i]); 	
					cudaMalloc( (Real**)&DSW[i][k].Tyy, sizeof(Real)*nTime*Cid.Snp[k][i]);	
					cudaMalloc( (Real**)&DSW[i][k].Tzz, sizeof(Real)*nTime*Cid.Snp[k][i]);
					cudaMalloc( (Real**)&DSW[i][k].Txy, sizeof(Real)*nTime*Cid.Snp[k][i]);	
					cudaMalloc( (Real**)&DSW[i][k].Txz, sizeof(Real)*nTime*Cid.Snp[k][i]);	
					err = cudaMalloc( (Real**)&DSW[i][k].Tyz, sizeof(Real)*nTime*Cid.Snp[k][i]);
					if(err != 0) printf("err = %d, errS=%s, error may occur at malloc for snap[%d]pcs[%d]dev[%d]\n",
							     err, cudaGetErrorString(err), i+1,HostMpiRank,k );
				}
			}
		}
#endif
	
	}
#endif
	fprintf(stdout,"Procs[%d], pass snappick()\n",HostMpiRank);

}
void ChildProcs::SWpick(wfield *speW, int currT)
{
#ifndef PointOnly	
	int i,j,k;
	int time,nTime;
	int src,dst;
	int idx,idy,idz;

	
	for(j=0;j<this->nsnap;j++)
	{
		if( currT%HSpt[j].tinv != 0 )
			continue;

		time = currT/HSpt[j].tinv;
		nTime = ceil(1.0*this->nt/HSpt[j].tinv);
		//printf("for snap[%d],INOtime=%d,----->pickTime=%d\n",j+1,currT,time);
		
		for(i=0;i<Cid.DNum;i++)
		{
			if(Cid.Snp[i][j])
			{
				cudaSetDevice( Cid.Rank[i] );
				//wave pick by kernel, Abandoned
				//SnapWavefieldPick<<<BlockPerGrid,ThreadPerBlock>>>(speW[i], this->DSW[j][i], D_DSpt[j][i], time, Cid.Snp[i][j], nTime);
				//CC(cudaDeviceSynchronize());
				for(k=0;k<Cid.Snp[i][j];k++)
				{
					//(Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*cdx.nz;
					idx = DSpt[j][i].locx[k]-Cstart-(Cid.xl[i]-LenFD);
					idy = DSpt[j][i].locy[k]-(Cid.yd[i]-LenFD);
					idz = DSpt[j][i].locz[k];
					
#ifdef DevicePick
					//pick to Device
					dst = DSpt[j][i].Rsn[k]*nTime + time;
					src = idx*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*cdx.nz + idy*cdx.nz + idz;
					if(DSpt[j][i].cmp==1 || DSpt[j][i].cmp==3)
					{
						cudaMemcpy(DSW[j][i].Vx+dst, speW[i].Vx+src, sizeof(Real), cudaMemcpyDeviceToDevice);
						cudaMemcpy(DSW[j][i].Vy+dst, speW[i].Vy+src, sizeof(Real), cudaMemcpyDeviceToDevice);
						cudaMemcpy(DSW[j][i].Vz+dst, speW[i].Vz+src, sizeof(Real), cudaMemcpyDeviceToDevice);
					}
					if(DSpt[j][i].cmp==2 || DSpt[j][i].cmp==3)
					{
						cudaMemcpy(DSW[j][i].Txx+dst, speW[i].Txx+src, sizeof(Real), cudaMemcpyDeviceToDevice);
						cudaMemcpy(DSW[j][i].Tyy+dst, speW[i].Tyy+src, sizeof(Real), cudaMemcpyDeviceToDevice);
						cudaMemcpy(DSW[j][i].Tzz+dst, speW[i].Tzz+src, sizeof(Real), cudaMemcpyDeviceToDevice);
						cudaMemcpy(DSW[j][i].Txy+dst, speW[i].Txy+src, sizeof(Real), cudaMemcpyDeviceToDevice);
						cudaMemcpy(DSW[j][i].Txz+dst, speW[i].Txz+src, sizeof(Real), cudaMemcpyDeviceToDevice);
						cudaMemcpy(DSW[j][i].Tyz+dst, speW[i].Tyz+src, sizeof(Real), cudaMemcpyDeviceToDevice);
					}
					
#else
					
					//pick to Host
					dst = DSpt[j][i].Gsn[k]*nTime + time;
					src = idx*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*cdx.nz + idy*cdx.nz + idz;
					if(DSpt[j][i].cmp==1 || DSpt[j][i].cmp==3)
					{
						cudaMemcpy(HSW[j].Vx+dst, speW[i].Vx+src, sizeof(Real),   cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[j].Vy+dst, speW[i].Vy+src, sizeof(Real),   cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[j].Vz+dst, speW[i].Vz+src, sizeof(Real),   cudaMemcpyDeviceToHost);
					}
					if(DSpt[j][i].cmp==2 || DSpt[j][i].cmp==3)
					{
						cudaMemcpy(HSW[j].Txx+dst, speW[i].Txx+src, sizeof(Real), cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[j].Tyy+dst, speW[i].Tyy+src, sizeof(Real), cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[j].Tzz+dst, speW[i].Tzz+src, sizeof(Real), cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[j].Txy+dst, speW[i].Txy+src, sizeof(Real), cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[j].Txz+dst, speW[i].Txz+src, sizeof(Real), cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[j].Tyz+dst, speW[i].Tyz+src, sizeof(Real), cudaMemcpyDeviceToHost);
					}
					
#endif
				}

			}

		}

	}
#endif	

}
void ChildProcs::SWgather(int currT)
{
	//from Device to Host, DSW[i][j] to HSW[i], D2H
	//point is low dimension, time is fast dimension
#ifndef PointOnly	
#ifdef DevicePick	
	int i,j,k;
	int src,dst;
	int Tlen,nTime;//here is total time length, in iteration is time step
	for(k=0;k<this->nsnap;k++)
	{
		Tlen = ceil(1.0*currT/HSpt[k].tinv);
		nTime = ceil(1.0*this->nt/HSpt[k].tinv);
		//printf("for snap[%d],OTtime=%d,----->tinv=%d,Tlen=%d,nTime=%d\n",k+1,currT,HSpt[k].tinv,Tlen,nTime);
		
		for(i=0;i<Cid.DNum;i++)
		{
			if(Cid.Snp[i][k])
			{
				cudaSetDevice( Cid.Rank[i] );
				for(j=0;j<Cid.Snp[i][k];j++)
				{
					src = DSpt[k][i].Rsn[j]*nTime;
					dst = DSpt[k][i].Gsn[j]*nTime;
					if(HSpt[k].cmp==1 || HSpt[k].cmp==3)
					{
						cudaMemcpy(HSW[k].Vx+dst,  DSW[k][i].Vx+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[k].Vy+dst,  DSW[k][i].Vy+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[k].Vz+dst,  DSW[k][i].Vz+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
					}
					if(HSpt[k].cmp==2 || HSpt[k].cmp==3)
					{
						cudaMemcpy(HSW[k].Txx+dst, DSW[k][i].Txx+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[k].Tyy+dst, DSW[k][i].Tyy+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[k].Tzz+dst, DSW[k][i].Tzz+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[k].Txy+dst, DSW[k][i].Txy+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[k].Txz+dst, DSW[k][i].Txz+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
						cudaMemcpy(HSW[k].Tyz+dst, DSW[k][i].Tyz+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
					}
				}
			}
		}
		
	}
#endif	
#endif

}

void ChildProcs::C2DFocalPick()
{
	int i,j,numD;
	cudaError_t err;
	
	//if(HostMpiRank==3)
	for(i=0;i<Cid.DNum;i++)
	{
		numD = 0;
		for(j=0;j<this->fpn;j++)
			if(HFpt.locx[j]>=Cid.xl[i]+Cstart && HFpt.locx[j]<=Cid.xr[i]+Cstart && HFpt.locy[j]>=Cid.yd[i] && HFpt.locy[j]<=Cid.yu[i])
			{
				numD++;
			}

		//malloc host side DFpt; 
		Cid.fp[i] = numD;
		DFpt[i].Rsn = new int[numD]();	DFpt[i].Gsn = new int[numD](); 
		DFpt[i].locx = new int[numD]();	DFpt[i].locy = new int[numD]();	DFpt[i].locz = new int[numD](); 
		
		//malloc device side focal data
		cudaSetDevice(Cid.Rank[i]);
		cudaMalloc( (int**)&Rmnt[i].locx, numD*sizeof(int) );
		cudaMalloc( (int**)&Rmnt[i].locy, numD*sizeof(int) );
		cudaMalloc( (int**)&Rmnt[i].locz, numD*sizeof(int) );
		cudaMalloc( (Real**)&Rmnt[i].mxx, numD*sizeof(Real) );
		cudaMalloc( (Real**)&Rmnt[i].myy, numD*sizeof(Real) );
		cudaMalloc( (Real**)&Rmnt[i].mzz, numD*sizeof(Real) );
		cudaMalloc( (Real**)&Rmnt[i].mxy, numD*sizeof(Real) );
		cudaMalloc( (Real**)&Rmnt[i].mxz, numD*sizeof(Real) );
		err = cudaMalloc( (Real**)&Rmnt[i].myz, numD*sizeof(Real) );
		if(err != 0) printf("err = %d, errS=%s, error may occur at Malloc Rmnt\n",err, cudaGetErrorString(err) );
#ifdef SrcSmooth	
		cudaMalloc( (Real**)&Rmnt[i].dnorm, numD*LenNorm*LenNorm*LenNorm*sizeof(Real) );
#endif

		cudaMemset( Rmnt[i].locx, 0, numD*sizeof(int));
		cudaMemset( Rmnt[i].locy, 0, numD*sizeof(int));
		cudaMemset( Rmnt[i].locz, 0, numD*sizeof(int));
		cudaMemset( Rmnt[i].mxx, 0, numD*sizeof(Real));
		cudaMemset( Rmnt[i].myy, 0, numD*sizeof(Real));
		cudaMemset( Rmnt[i].mzz, 0, numD*sizeof(Real));
		cudaMemset( Rmnt[i].mxy, 0, numD*sizeof(Real));
		cudaMemset( Rmnt[i].mxz, 0, numD*sizeof(Real));
		cudaMemset( Rmnt[i].myz, 0, numD*sizeof(Real));
#ifdef SrcSmooth	
		err = cudaMemset( Rmnt[i].dnorm, 0, numD*LenNorm*LenNorm*LenNorm*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset Rmnt\n",
				err, cudaGetErrorString(err) );
#endif

		//malloc host side interp focal data
		IM[i].mxx = new Real[numD](); IM[i].myy = new Real[numD](); IM[i].mzz = new Real[numD]();
		IM[i].mxy = new Real[numD](); IM[i].mxz = new Real[numD](); IM[i].myz = new Real[numD]();

	}
	
#ifdef SrcSmooth
	int Src;
	Src=0;
#endif
	
	for(i=0;i<Cid.DNum;i++)
	{
		//assign DFpt in host and device side
		numD = 0;
		for(j=0;j<this->fpn;j++)
			if(HFpt.locx[j]>=Cid.xl[i]+Cstart && HFpt.locx[j]<=Cid.xr[i]+Cstart && HFpt.locy[j]>=Cid.yd[i] && HFpt.locy[j]<=Cid.yu[i])
			{
				DFpt[i].Rsn[numD] = numD;
				DFpt[i].Gsn[numD] = HFpt.Rsn[j];
				DFpt[i].locx[numD] = HFpt.locx[j];
				DFpt[i].locy[numD] = HFpt.locy[j];
				DFpt[i].locz[numD] = HFpt.locz[j];
				numD++;
			}
		
		cudaSetDevice(Cid.Rank[i]);
		cudaMemcpy(Rmnt[i].locx, DFpt[i].locx, Cid.fp[i]*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(Rmnt[i].locy, DFpt[i].locy, Cid.fp[i]*sizeof(int), cudaMemcpyHostToDevice);
		err = cudaMemcpy(Rmnt[i].locz, DFpt[i].locz, Cid.fp[i]*sizeof(int), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at memcpy Rmnt loc\n", err, cudaGetErrorString(err) );
#ifdef SrcSmooth
		//start from 0, copy Cid.fp[i], then shift Cid.fp[i];
		//printf("at PCS[%d]Dev[%d], memcpy dorm, from %d\n",HostMpiRank, i, Src);
		err = cudaMemcpy(Rmnt[i].dnorm,  H_Rmnt.dnorm+Src,  Cid.fp[i]*LenNorm*LenNorm*LenNorm*sizeof(Real), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at memcpy Rmnt loc\n", err, cudaGetErrorString(err) );
		Src = (Src + Cid.fp[i])*LenNorm*LenNorm*LenNorm;
#endif

	}
	
	fprintf(stdout,"PCS[%d], pass focalpick\n",HostMpiRank);
}

Real ChildProcs::ExtractValue(Real *stf, Real Tstart, Real Tend, Real DT, Real time)
{
	int PL, PR;
	Real value;

	value=0;
	
	if( time > Tend || time< Tstart )
		value = 0;
	else
	{
		if( fabs( time-DT*floor(time/DT) ) < 1e-5 )
		{
			PL = (int) (time/DT);
			PR = PL;
		}
		else
		{
			PL = (int)(time/DT);
			PR = PL+1;
		}

		value = stf[PL] + (stf[PR]-stf[PL])*( (time-DT*PL)/DT );
	}

	return value;

}

void ChildProcs::InterpFocus(Real time)
{
	int i,j;
	int Src;
	int numD;
	cudaError_t err;
	
	Real Tstart,Tend;//orginal focal data time sereis length
	//current time---->time = (currt + tinc*0.5)*stept;

	Tstart = 0;	Tend = FDT*(FNT-1);

	//H_Rmnt.mxx[ fpn*FNT ]
	//IM[ Cid.DNum ].mxx[ Cid.fp[i] ]
	//Rmnt[ Cid.DNum ].mxx[ Cid.fp[i] ]
		
	//interp
	for(i=0;i<Cid.DNum;i++)
	{
		for(j=0;j<Cid.fp[i];j++)
		{
			Src = DFpt[i].Gsn[j] * FNT;
			//Dst = DFpt[i].Rsn[j];//j
			// from H_Rmnt.mxx[Src] 
			// to IM[i].mxx[Dst]

			IM[i].mxx[j] =  ExtractValue( H_Rmnt.mxx+Src, Tstart, Tend, FDT, time);
			IM[i].myy[j] =  ExtractValue( H_Rmnt.myy+Src, Tstart, Tend, FDT, time);
			IM[i].mzz[j] =  ExtractValue( H_Rmnt.mzz+Src, Tstart, Tend, FDT, time);
			IM[i].mxy[j] =  ExtractValue( H_Rmnt.mxy+Src, Tstart, Tend, FDT, time);
			IM[i].mxz[j] =  ExtractValue( H_Rmnt.mxz+Src, Tstart, Tend, FDT, time);
			IM[i].myz[j] =  ExtractValue( H_Rmnt.myz+Src, Tstart, Tend, FDT, time);

		}
	}

	//memcpy
	for(i=0;i<Cid.DNum;i++)
	{
		numD = Cid.fp[i];
		
		cudaSetDevice(Cid.Rank[i]);
		//flush first
		err = cudaMemset( Rmnt[i].mxx, 0, numD*sizeof(Real));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memset Rmnt mxx at interpstep\n", err, cudaGetErrorString(err) );
		cudaMemset( Rmnt[i].myy, 0, numD*sizeof(Real));
		cudaMemset( Rmnt[i].mzz, 0, numD*sizeof(Real));
		cudaMemset( Rmnt[i].mxy, 0, numD*sizeof(Real));
		cudaMemset( Rmnt[i].mxz, 0, numD*sizeof(Real));
		cudaMemset( Rmnt[i].myz, 0, numD*sizeof(Real));
		
		err = cudaMemcpy( Rmnt[i].mxx,  IM[i].mxx,  numD*sizeof(Real), cudaMemcpyHostToDevice);
		if(err != 0) printf("err = %d, errS=%s, error may occur at memcpy Rmnt mxx at intepstep\n", err, cudaGetErrorString(err) );
		cudaMemcpy( Rmnt[i].myy,  IM[i].myy,  numD*sizeof(Real), cudaMemcpyHostToDevice);
		cudaMemcpy( Rmnt[i].mzz,  IM[i].mzz,  numD*sizeof(Real), cudaMemcpyHostToDevice);
		cudaMemcpy( Rmnt[i].mxy,  IM[i].mxy,  numD*sizeof(Real), cudaMemcpyHostToDevice);
		cudaMemcpy( Rmnt[i].mxz,  IM[i].mxz,  numD*sizeof(Real), cudaMemcpyHostToDevice);
		cudaMemcpy( Rmnt[i].myz,  IM[i].myz,  numD*sizeof(Real), cudaMemcpyHostToDevice);

	}

}

void ChildProcs::C2DPointPick()
{
	int i,j,numD;
	cudaError_t err;
	
	//if(HostMpiRank==3)
	for(i=0;i<Cid.DNum;i++)
	{
		numD = 0;
		for(j=0;j<this->ppn;j++)
			if(Hpt.locx[j]>=Cid.xl[i]+Cstart && Hpt.locx[j]<=Cid.xr[i]+Cstart && Hpt.locy[j]>=Cid.yd[i] && Hpt.locy[j]<=Cid.yu[i])
			{
				numD++;
			}

		//malloc host side Dpt and device side Dpt;
		Cid.np[i] = numD;
		Dpt[i].Rsn = new int[numD]();	Dpt[i].Gsn = new int[numD](); 
		Dpt[i].locx = new int[numD]();	Dpt[i].locy = new int[numD]();	Dpt[i].locz = new int[numD](); 
		
		cudaSetDevice(Cid.Rank[i]);
		cudaMalloc( (int**)&D_Dpt[i].Rsn, numD*sizeof(int));
		cudaMalloc( (int**)&D_Dpt[i].Gsn, numD*sizeof(int));
		cudaMalloc( (int**)&D_Dpt[i].locx, numD*sizeof(int));
		cudaMalloc( (int**)&D_Dpt[i].locy, numD*sizeof(int));
		err=cudaMalloc( (int**)&D_Dpt[i].locz, numD*sizeof(int));
		if(err != 0) printf("err = %d, errS=%s, error may occur at memcpy point loc\n", err, cudaGetErrorString(err) );
		
#ifdef DevicePick		
		//malloc device wave point buffer
		cudaMalloc( (Real**)&DPW[i].Vx, nt*Cid.np[i]*sizeof(Real) );
		cudaMalloc( (Real**)&DPW[i].Vy, nt*Cid.np[i]*sizeof(Real) ); 
		cudaMalloc( (Real**)&DPW[i].Vz, nt*Cid.np[i]*sizeof(Real) ); 
		cudaMalloc( (Real**)&DPW[i].Txx, nt*Cid.np[i]*sizeof(Real) ); 
		cudaMalloc( (Real**)&DPW[i].Tyy, nt*Cid.np[i]*sizeof(Real) ); 
		cudaMalloc( (Real**)&DPW[i].Tzz, nt*Cid.np[i]*sizeof(Real) ); 
		cudaMalloc( (Real**)&DPW[i].Txy, nt*Cid.np[i]*sizeof(Real) ); 
		cudaMalloc( (Real**)&DPW[i].Txz, nt*Cid.np[i]*sizeof(Real) ); 
		err=cudaMalloc( (Real**)&DPW[i].Tyz, nt*Cid.np[i]*sizeof(Real) ); 
		if(err != 0) printf("err = %d, errS=%s, error may occur at memcpy DPW \n", err, cudaGetErrorString(err) );
#endif

	}
	
	for(i=0;i<Cid.DNum;i++)
	{
		//assign Dpt in host and device side
		numD = 0;
		for(j=0;j<this->ppn;j++)
			if(Hpt.locx[j]>=Cid.xl[i]+Cstart && Hpt.locx[j]<=Cid.xr[i]+Cstart && Hpt.locy[j]>=Cid.yd[i] && Hpt.locy[j]<=Cid.yu[i])
			{
				Dpt[i].Rsn[numD] = numD;
				Dpt[i].Gsn[numD] = Hpt.Rsn[j];
				Dpt[i].locx[numD] = Hpt.locx[j];
				Dpt[i].locy[numD] = Hpt.locy[j];
				Dpt[i].locz[numD] = Hpt.locz[j];
				numD++;
			}
	
		cudaSetDevice(Cid.Rank[i]);
		cudaMemcpy(D_Dpt[i].Rsn, Dpt[i].Rsn, Cid.np[i]*sizeof(int), cudaMemcpyHostToDevice);	
		cudaMemcpy(D_Dpt[i].Gsn, Dpt[i].Gsn, Cid.np[i]*sizeof(int), cudaMemcpyHostToDevice);	
		cudaMemcpy(D_Dpt[i].locx, Dpt[i].locx, Cid.np[i]*sizeof(int), cudaMemcpyHostToDevice);	
		cudaMemcpy(D_Dpt[i].locy, Dpt[i].locy, Cid.np[i]*sizeof(int), cudaMemcpyHostToDevice);	
		err=cudaMemcpy(D_Dpt[i].locz, Dpt[i].locz, Cid.np[i]*sizeof(int), cudaMemcpyHostToDevice);	
		if(err != 0) printf("err = %d, errS=%s, error may occur at memcpy point loc\n", err, cudaGetErrorString(err) );
	}
	fprintf(stdout,"Procs[%d], pass pointpick()\n",HostMpiRank);

}
void ChildProcs::PWgather(int currT)
{

#ifdef DevicePick	
	//from Device to Host, DPW[i] to HPW, D2H
	//point is low dimension, time is fast dimension
	int i,j;
	int src,dst;
	int Tlen;
	Tlen = currT;//gen ju shi jian wei zhi tiao zheng
	//Tlen = currT+1;//gen ju shi jian wei zhi tiao zheng
	for(i=0;i<Cid.DNum;i++)
	{
		cudaSetDevice( Cid.Rank[i] );
		for(j=0;j<Cid.np[i];j++)
		{
			src = Dpt[i].Rsn[j]*this->nt;
			dst = Dpt[i].Gsn[j]*this->nt;
			cudaMemcpy(HPW.Vx+dst, DPW[i].Vx+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Vy+dst, DPW[i].Vy+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Vz+dst, DPW[i].Vz+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Txx+dst, DPW[i].Txx+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Tyy+dst, DPW[i].Tyy+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Tzz+dst, DPW[i].Tzz+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Txy+dst, DPW[i].Txy+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Txz+dst, DPW[i].Txz+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Tyz+dst, DPW[i].Tyz+src, Tlen*sizeof(Real), cudaMemcpyDeviceToHost);
		}
	}
#endif	
	
}
void ChildProcs::PWpick(wfield *speW, int currT)
{
	int i,j;
	int src,dst;
	int idx,idy,idz;

	for(i=0;i<Cid.DNum;i++)
	{
		cudaSetDevice( Cid.Rank[i] );
		//WavefieldPick<<<BlockPerGrid,ThreadPerBlock>>>(speW[i], this->DPW[i], D_Dpt[i], currT, Cid.np[i], nt);
		//CC(cudaDeviceSynchronize());
		for(j=0;j<Cid.np[i];j++)
		{
			idx = Dpt[i].locx[j]-Cstart-(Cid.xl[i]-LenFD);
			idy = Dpt[i].locy[j]-(Cid.yd[i]-LenFD);
			idz = Dpt[i].locz[j];
			
#ifdef DevicePick			
			//pick to Device
			dst = Dpt[i].Rsn[j]*nt+currT;
			src = idx*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*cdx.nz + idy*cdx.nz + idz;
			cudaMemcpy(DPW[i].Vx+dst, speW[i].Vx+src, sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(DPW[i].Vy+dst, speW[i].Vy+src, sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(DPW[i].Vz+dst, speW[i].Vz+src, sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(DPW[i].Txx+dst, speW[i].Txx+src, sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(DPW[i].Tyy+dst, speW[i].Tyy+src, sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(DPW[i].Tzz+dst, speW[i].Tzz+src, sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(DPW[i].Txy+dst, speW[i].Txy+src, sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(DPW[i].Txz+dst, speW[i].Txz+src, sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(DPW[i].Tyz+dst, speW[i].Tyz+src, sizeof(Real), cudaMemcpyDeviceToDevice);
#else			
			//pick to Host side
			dst = Dpt[i].Gsn[j]*nt+currT;
			src = idx*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*cdx.nz + idy*cdx.nz + idz;
			cudaMemcpy(HPW.Vx+dst, speW[i].Vx+src, sizeof(Real),   cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Vy+dst, speW[i].Vy+src, sizeof(Real),   cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Vz+dst, speW[i].Vz+src, sizeof(Real),   cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Txx+dst, speW[i].Txx+src, sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Tyy+dst, speW[i].Tyy+src, sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Tzz+dst, speW[i].Tzz+src, sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Txy+dst, speW[i].Txy+src, sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Txz+dst, speW[i].Txz+src, sizeof(Real), cudaMemcpyDeviceToHost);
			cudaMemcpy(HPW.Tyz+dst, speW[i].Tyz+src, sizeof(Real), cudaMemcpyDeviceToHost);
			
#endif	
		}

	}
}


//------------------------------private-------------------------
void ChildProcs::GpuAbility(const char *filename)
{
	char parpath[SeisStrLen];
	char name[SeisStrLen2];
	char errstr[SeisStrLen];
	char devfile[SeisStrLen];
	int i,j;
	int B1,B2,B3,T1,T2,T3;
	int deviceNum;
	int pid=0;
	char hostname[256];
	char Str[256];

	FILE *fp;
	fp = fopen(filename,"r");
	if(!fp)
	{
		sprintf(errstr,"Fail to open main par file %s in GpuAbility", filename);
		errprt(Fail2Open,errstr);
	}
	com.get_conf(fp, "seispath", 3, parpath);
	com.get_conf(fp, "device_filename", 3, name);
	com.get_conf(fp, "BlockPerGrid", 3, &B1);
	com.get_conf(fp, "BlockPerGrid", 4, &B2);
	com.get_conf(fp, "BlockPerGrid", 5, &B3);
	com.get_conf(fp, "ThreadPerBlock", 3, &T1);
	com.get_conf(fp, "ThreadPerBlock", 4, &T2);
	com.get_conf(fp, "ThreadPerBlock", 5, &T3);
	fclose(fp);

	sprintf(devfile,"%s/%s",parpath,name);

	//comfirm lanuch parameters should with boundary
	cdx.nx <= B2 ? BlockPerGrid.y = cdx.nx : BlockPerGrid.y = B2;
	cdx.ny <= B1 ? BlockPerGrid.x = cdx.ny : BlockPerGrid.x = B1;
	cdx.nz <= T1 ? ThreadPerBlock.x = cdx.nz : ThreadPerBlock.x = T1;
	BlockPerGrid.z = 1;
	ThreadPerBlock.y = 1; 
	ThreadPerBlock.z = 1;
	//BPG(B1,B2,1) TPB(T1,1,1)
	//<<<BPG,TPB>>>

	//check device number
	gethostname(hostname,256);
	pid=getpid();
	cudaGetDeviceCount(&deviceNum);

	printf("\nOn current node %s, Rank is %d and PID is %d, has %d GPU device\n",hostname,HostMpiRank,pid,deviceNum);

	//DEC confirmation
	fp = fopen(devfile,"r");

	memset(Str,'\0',256*sizeof(char));
	sprintf(Str,"used_device_number_%s",hostname);

	com.get_conf(fp, Str, 3, &Cid.DNum);
	if(Cid.DNum > deviceNum)
		Cid.DNum = deviceNum;

	Cid.Snp = new int*[Cid.DNum];
	for(int iii=0;iii<Cid.DNum;iii++)
		Cid.Snp[iii] = new int [this->nsnap]();
	Cid.fp = new int [Cid.DNum]();//focal number
	Cid.np = new int [Cid.DNum]();//point number
	Cid.Rank = new int [Cid.DNum]();
	Cid.Size = new int [Cid.DNum]();
	Cid.xl = new int [Cid.DNum]();
	Cid.xr = new int [Cid.DNum]();
	Cid.yd = new int [Cid.DNum]();
	Cid.yu = new int [Cid.DNum]();
	BPG = new dim3 [Cid.DNum]();

	memset(Str,'\0',256*sizeof(char));
	sprintf(Str,"device_ydims_%s",hostname);
	com.get_conf(fp, Str, 3, &Cid.ydim);

	memset(Str,'\0',256*sizeof(char));
	sprintf(Str,"used_device_id_%s",hostname);
	
	for(i=0;i<Cid.DNum;i++)
		com.get_conf(fp, Str, 3+i, &Cid.Rank[i]);
	printf("On this node %s, totally used %d devices, the Device ID is :",hostname,Cid.DNum);
	for(i=0;i<Cid.DNum;i++)
		printf("%d ",Cid.Rank[i]);
	cout<<endl;
	
	fclose(fp);
	
	
	//enable P2P
	//enable unified memory
	int tempflag=0;
	//cudaError_t errmessage;
	cudaDeviceProp deviceProp;
	cout<<"Check needed GPU device features (P2P, ManagedMemory)\n";
	for(i=0;i<Cid.DNum;i++)
	{
		cudaSetDevice( Cid.Rank[i] );
		cudaGetDeviceProperties(&deviceProp,Cid.Rank[i]);
		
		//P2P
		for(j=0;j<Cid.DNum;j++)
		{
			if(Cid.Rank[i]==Cid.Rank[j])
				continue;
			checkCudaErrors(cudaDeviceCanAccessPeer(&tempflag,Cid.Rank[i],Cid.Rank[j]));//on I to active J
			if(tempflag)
			{
				if(cudaSuccess != cudaDeviceEnablePeerAccess(Cid.Rank[j],0))
					printf("On device %d to active device %d, errmessage is %s\n",
					Cid.Rank[i],Cid.Rank[j],cudaGetErrorString(cudaGetLastError()));
			}
		}
		
		//Unified Memory
		if(!deviceProp.managedMemory)
			printf("On device %d doesn't support managed memory\n",Cid.Rank[i]);
		//concurrent managed access
		if(!deviceProp.concurrentManagedAccess)
			printf("On device %d doesn't support concurrently managed memory access by CPU\n",Cid.Rank[i]);
	}
	
}

void ChildProcs::wavesyn(wfield *Output, wfield *Input)
{
	int inputflag=1;
	int size;
	
	for(int i=0;i<Cid.DNum;i++)
	{
		cudaSetDevice( Cid.Rank[i] );
		size = (Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*cdx.nz;
		if(inputflag)
		{//should use this, in GPU side;
			cudaMemcpy(Output[i].Txx, Input[i].Txx, size*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(Output[i].Tyy, Input[i].Tyy, size*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(Output[i].Tzz, Input[i].Tzz, size*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(Output[i].Txy, Input[i].Txy, size*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(Output[i].Txz, Input[i].Txz, size*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(Output[i].Tyz, Input[i].Tyz, size*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(Output[i].Vx,  Input[i].Vx,  size*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(Output[i].Vy,  Input[i].Vy,  size*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(Output[i].Vz,  Input[i].Vz,  size*sizeof(Real), cudaMemcpyDeviceToDevice);
		}
		else
		{
			memcpy(Output[i].Txx, Input[i].Txx, size*sizeof(Real) );
			memcpy(Output[i].Tyy, Input[i].Tyy, size*sizeof(Real) );
			memcpy(Output[i].Tzz, Input[i].Tzz, size*sizeof(Real) );
			memcpy(Output[i].Txy, Input[i].Txy, size*sizeof(Real) );
			memcpy(Output[i].Txz, Input[i].Txz, size*sizeof(Real) );
			memcpy(Output[i].Tyz, Input[i].Tyz, size*sizeof(Real) );
			memcpy(Output[i].Vx,  Input[i].Vx,  size*sizeof(Real) );
			memcpy(Output[i].Vy,  Input[i].Vy,  size*sizeof(Real) );
			memcpy(Output[i].Vz,  Input[i].Vz,  size*sizeof(Real) );
		}
	}

}

void ChildProcs::abssyn(int TransDir)
{
	int Xsize,Ysize,Zsize;
	//before a new RKite, input FW as fixed field,
	//everytimes reflush mW and update W,
	//after RKite, store FW from W.
	//So, TD=1 means FW to mW,	TD=2 means W to mW,	TD=3 means W to FW;

#ifdef DisBug
	//printf("input ABS-syn direction is %d\n",TransDir);
#endif
	
	if(TransDir!=1 && TransDir!=2 && TransDir!=3)
		printf("Absorption wavefield synchronizing direction error\n");
	for(int i=0; i<Cid.DNum; i++)
	{
		cudaSetDevice( Cid.Rank[i] );
		Xsize = (H_apr.nabs[0]+H_apr.nabs[1])*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD)*cdx.nz;
		Ysize = (H_apr.nabs[2]+H_apr.nabs[3])*(Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*cdx.nz;
		Zsize = (H_apr.nabs[4]+H_apr.nabs[5])*(Cid.xr[i]-Cid.xl[i]+1 + 2*LenFD)*(Cid.yu[i]-Cid.yd[i]+1 + 2*LenFD);
		
		if(TransDir==1)
		{
			cudaMemcpy(mAx[i].Txx, FAx[i].Txx, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Tyy, FAx[i].Tyy, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Tzz, FAx[i].Tzz, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Txy, FAx[i].Txy, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Txz, FAx[i].Txz, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Tyz, FAx[i].Tyz, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Vx,  FAx[i].Vx,  Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Vy,  FAx[i].Vy,  Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Vz,  FAx[i].Vz,  Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);

			cudaMemcpy(mAy[i].Txx, FAy[i].Txx, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Tyy, FAy[i].Tyy, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Tzz, FAy[i].Tzz, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Txy, FAy[i].Txy, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Txz, FAy[i].Txz, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Tyz, FAy[i].Tyz, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Vx,  FAy[i].Vx,  Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Vy,  FAy[i].Vy,  Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Vz,  FAy[i].Vz,  Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);

			cudaMemcpy(mAz[i].Txx, FAz[i].Txx, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Tyy, FAz[i].Tyy, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Tzz, FAz[i].Tzz, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Txy, FAz[i].Txy, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Txz, FAz[i].Txz, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Tyz, FAz[i].Tyz, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Vx,  FAz[i].Vx,  Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Vy,  FAz[i].Vy,  Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Vz,  FAz[i].Vz,  Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
		}

		if(TransDir==2)
		{
			cudaMemcpy(mAx[i].Txx, Ax[i].Txx, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Tyy, Ax[i].Tyy, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Tzz, Ax[i].Tzz, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Txy, Ax[i].Txy, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Txz, Ax[i].Txz, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Tyz, Ax[i].Tyz, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Vx,  Ax[i].Vx,  Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Vy,  Ax[i].Vy,  Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAx[i].Vz,  Ax[i].Vz,  Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);

			cudaMemcpy(mAy[i].Txx, Ay[i].Txx, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Tyy, Ay[i].Tyy, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Tzz, Ay[i].Tzz, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Txy, Ay[i].Txy, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Txz, Ay[i].Txz, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Tyz, Ay[i].Tyz, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Vx,  Ay[i].Vx,  Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Vy,  Ay[i].Vy,  Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAy[i].Vz,  Ay[i].Vz,  Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);

			cudaMemcpy(mAz[i].Txx, Az[i].Txx, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Tyy, Az[i].Tyy, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Tzz, Az[i].Tzz, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Txy, Az[i].Txy, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Txz, Az[i].Txz, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Tyz, Az[i].Tyz, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Vx,  Az[i].Vx,  Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Vy,  Az[i].Vy,  Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(mAz[i].Vz,  Az[i].Vz,  Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
		}

		if(TransDir==3)
		{
			cudaMemcpy(FAx[i].Txx, Ax[i].Txx, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAx[i].Tyy, Ax[i].Tyy, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAx[i].Tzz, Ax[i].Tzz, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAx[i].Txy, Ax[i].Txy, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAx[i].Txz, Ax[i].Txz, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAx[i].Tyz, Ax[i].Tyz, Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAx[i].Vx,  Ax[i].Vx,  Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAx[i].Vy,  Ax[i].Vy,  Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAx[i].Vz,  Ax[i].Vz,  Xsize*sizeof(Real), cudaMemcpyDeviceToDevice);

			cudaMemcpy(FAy[i].Txx, Ay[i].Txx, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAy[i].Tyy, Ay[i].Tyy, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAy[i].Tzz, Ay[i].Tzz, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAy[i].Txy, Ay[i].Txy, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAy[i].Txz, Ay[i].Txz, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAy[i].Tyz, Ay[i].Tyz, Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAy[i].Vx,  Ay[i].Vx,  Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAy[i].Vy,  Ay[i].Vy,  Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAy[i].Vz,  Ay[i].Vz,  Ysize*sizeof(Real), cudaMemcpyDeviceToDevice);

			cudaMemcpy(FAz[i].Txx, Az[i].Txx, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAz[i].Tyy, Az[i].Tyy, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAz[i].Tzz, Az[i].Tzz, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAz[i].Txy, Az[i].Txy, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAz[i].Txz, Az[i].Txz, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAz[i].Tyz, Az[i].Tyz, Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAz[i].Vx,  Az[i].Vx,  Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAz[i].Vy,  Az[i].Vy,  Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(FAz[i].Vz,  Az[i].Vz,  Zsize*sizeof(Real), cudaMemcpyDeviceToDevice);
		}
		

	}

}



//-----------------------------------------kernel-----------------------------------------

__global__ void VelPDcoeff(derivF drv, mdparF mpa, apara apr, Real *matVx2Vz, Real *matVy2Vz)
{
	//this part only used under free surface condition and that means must apply CondFree macro
	//when apply free surface, the top layer doesn't need absorbtion, so only need to calculate
	//damped velocity partial derivative on four side and one bottom (without the inner of the top layer)

	//int i,j,k;
	//  <<<BPG.y, BPG.x>>>
	//gridDim.x<=cdx.nx  blockDim.x<=cdx.ny
	int countX,countY;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int xiaoI;
	int i;

	Real e11,e12,e13,e21,e22,e23,e31,e32,e33;
	Real lambda,miu,lam2mu;
	Real A[9],B[9],C[9],temp[9];
	Real Bzx,Bzy;//use for PML
	
	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.x)//loop in current compute range with step of Launch Par
	{
		idx = countX + blockIdx.x + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=blockDim.x)
			{
				idy = countY + threadIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{  
					idz = ipam[8]+LenFD-1;

					Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
#ifdef CFSPML
					Bzx = apr.Bz[idz]/apr.Bx[idx];	Bzy = apr.Bz[idz]/apr.By[idy];
#else	
					Bzx = 1.0;	Bzy = 1.0;
#endif
					miu = mpa.rho[Gindex]*mpa.beta[Gindex]*mpa.beta[Gindex];
					lam2mu = mpa.rho[Gindex]*mpa.alpha[Gindex]*mpa.alpha[Gindex];
					lambda = lam2mu - 2.0*miu;

					e11 = drv.xix[Gindex];
					e12 = drv.xiy[Gindex];
					e13 = drv.xiz[Gindex];
					e21 = drv.etax[Gindex];
					e22 = drv.etay[Gindex];
					e23 = drv.etaz[Gindex];
					e31 = drv.zetax[Gindex];
					e32 = drv.zetay[Gindex];
					e33 = drv.zetaz[Gindex];
					
					A[0] = lam2mu*e31*e31 + miu*(e32*e32+e33*e33);
					A[1] = lambda*e31*e32 + miu*e32*e31;
					A[2] = lambda*e31*e33 + miu*e33*e31;
					A[3] = lambda*e32*e31 + miu*e31*e32;
					A[4] = lam2mu*e32*e32 + miu*(e31*e31+e33*e33);
					A[5] = lambda*e32*e33 + miu*e33*e32;
					A[6] = lambda*e33*e31 + miu*e31*e33;
					A[7] = lambda*e33*e32 + miu*e32*e33;
					A[8] = lam2mu*e33*e33 + miu*(e31*e31+e32*e32);

					matinv(A);

					B[0] = lam2mu*e31*e11 + miu*(e32*e12+e33*e13);
					B[1] = lambda*e31*e12 + miu*e32*e11;
					B[2] = lambda*e31*e13 + miu*e33*e11;
					B[3] = lambda*e32*e11 + miu*e31*e12;
					B[4] = lam2mu*e32*e12 + miu*(e31*e11+e33*e13);
					B[5] = lambda*e32*e13 + miu*e33*e12;
					B[6] = lambda*e33*e11 + miu*e31*e13;
					B[7] = lambda*e33*e12 + miu*e32*e13;
					B[8] = lam2mu*e33*e13 + miu*(e32*e12+e31*e11);

					C[0] = lam2mu*e31*e21 + miu*(e32*e22+e33*e23);
					C[1] = lambda*e31*e22 + miu*e32*e21;
					C[2] = lambda*e31*e23 + miu*e33*e21;
					C[3] = lambda*e32*e21 + miu*e31*e22;
					C[4] = lam2mu*e32*e22 + miu*(e31*e21+e33*e23);
					C[5] = lambda*e32*e23 + miu*e33*e22;
					C[6] = lambda*e33*e21 + miu*e31*e23;
					C[7] = lambda*e33*e22 + miu*e32*e23;
					C[8] = lam2mu*e33*e23 + miu*(e31*e21+e32*e22);

					xiaoI=idx*(ipam[5]-ipam[4]+1+2*LenFD)*SeisGeo*SeisGeo + idy*SeisGeo*SeisGeo;//valid Y

					matmul(A,B,temp);
					for(i=0;i<SeisGeo*SeisGeo;i++)
						matVx2Vz[xiaoI+i] = -1*temp[i]*Bzx;

					matmul(A,C,temp);
					for(i=0;i<SeisGeo*SeisGeo;i++)
						matVy2Vz[xiaoI+i] = -1*temp[i]*Bzy;

				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

__device__ void matmul(Real *A, Real *B, Real *C)
{
	int n;
	int i,j,k;
	n=3;

	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
		{
			C[i*n+j] = 0.0;
			for(k=0;k<n;k++)
				C[i*n+j]=C[i*n+j]+A[i*n+k]*B[k*n+j];
		}
}
__device__ void matinv(Real *A)
{
	int i,j,k,n;
	
	n=3;
	Real con;
	for(i=0;i<n;i++)
	{
		con = A[i*n+i];
		A[i*n+i] = 1;
		for(j=0;j<n;j++)
			A[i*n+j]=A[i*n+j]/con;

		for(j=0;j<n;j++)
			if(j!=i)
			{
				con = A[j*n+i];
				A[j*n+i] = 0;
				for(k=0;k<n;k++)
					A[j*n+k]=A[j*n+k]-A[i*n+k]*con;
			}
	}
}


__global__ void perform()
{
	printf("display data: ");
	for(int i=0;i<11;i++)
		printf("%d ",ipam[i]);
	printf("\n");
}

__global__ void generatewave(wfield wfake, int time, int Cstart)
{
	//int i,j,k;
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	Real value;

	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<ipam[8]+LenFD)//vaild point with one virtual bounds
						{
							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;

							value = ( time*1E11 + (idx+Cstart+1+ipam[2]-LenFD)*1E6 + (idy+1+ipam[4]-LenFD)*1E3 + idz+1 )/1E6; 

							wfake.Vx[Gindex] = 1*1000 + value;
							wfake.Vy[Gindex] = 2*1000 + value;
							wfake.Vz[Gindex] = 3*1000 + value;
							wfake.Txx[Gindex] = 4*1000 + value;
							wfake.Tyy[Gindex] = 5*1000 + value;
							wfake.Tzz[Gindex] = 6*1000 + value;
							wfake.Txy[Gindex] = 7*1000 + value;
							wfake.Txz[Gindex] = 8*1000 + value;
							wfake.Tyz[Gindex] = 9*1000 + value;
							
							//check pass
							//if(idx==53 && idy==53 && idz==53)
							//	printf("index=%d, value=%lf,Txx=%lf\n",Gindex,value,wfake.Txx[Gindex]);

						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

__global__ void CalDiff(int Xvec, int Yvec, int Zvec, int ConIndex, Real steph, Real *CoVx, Real* CoVy, wfield W, PartialD pd)
{
	//int i,j,k;
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int xiaoI;
#ifdef HYindex	
	int Hyindex;
#endif

	Real xstep, ystep, zstep;
	int xinc, yinc, zinc;
	
	xstep = steph*Xvec;
	ystep = steph*Yvec;
	zstep = steph*Zvec;
	xinc = Xvec*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD);//skip cdx.ny*cdx.nz
	yinc = Yvec*(ipam[8]+2*LenFD);//skip cdx.nz
	zinc = Zvec*1;//skip 1


	//generally use DRP/opt MacCormack scheme to get derivative, as Equation 2.23 and coefficients is Equation 2.24 in Thesis.
	//for the top layer transfrom the derivative of xi and eta to get zeta direction derivative, as Equation 3.4 in Thesis.
	
	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current device compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						//if(idz<ipam[8]+LenFD && idz>=ConIndex)//vaild point with one virtual bounds
						if(idz<ipam[8]+LenFD)//vaild point with one virtual bounds
						{

							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
							xiaoI = idx*(ipam[5]-ipam[4]+1+2*LenFD)*SeisGeo*SeisGeo + idy*SeisGeo*SeisGeo;//valid Y
							
							if(idz>=ConIndex)
							{
#ifdef HYindex
								Hyindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+LenFD-ConIndex) 
									+ idy*(ipam[8]+LenFD-ConIndex) + idz-ConIndex;
								pd.DxTyy[Hyindex] = DRPFD( W.Tyy, Gindex, xstep, xinc);
								pd.DxTzz[Hyindex] = DRPFD( W.Tzz, Gindex, xstep, xinc);
								pd.DxTyz[Hyindex] = DRPFD( W.Tyz, Gindex, xstep, xinc);
								pd.DyTxx[Hyindex] = DRPFD( W.Txx, Gindex, ystep, yinc);
								pd.DyTzz[Hyindex] = DRPFD( W.Tzz, Gindex, ystep, yinc);
								pd.DyTxz[Hyindex] = DRPFD( W.Txz, Gindex, ystep, yinc);
								pd.DzTxx[Hyindex] = DRPFD( W.Txx, Gindex, zstep, zinc);
								pd.DzTyy[Hyindex] = DRPFD( W.Tyy, Gindex, zstep, zinc);
								pd.DzTxy[Hyindex] = DRPFD( W.Txy, Gindex, zstep, zinc);
#else						
								pd.DxTyy[Gindex] = DRPFD( W.Tyy, Gindex, xstep, xinc);
								pd.DxTzz[Gindex] = DRPFD( W.Tzz, Gindex, xstep, xinc);
								pd.DxTyz[Gindex] = DRPFD( W.Tyz, Gindex, xstep, xinc);
								pd.DyTxx[Gindex] = DRPFD( W.Txx, Gindex, ystep, yinc);
								pd.DyTzz[Gindex] = DRPFD( W.Tzz, Gindex, ystep, yinc);
								pd.DyTxz[Gindex] = DRPFD( W.Txz, Gindex, ystep, yinc);
								pd.DzTxx[Gindex] = DRPFD( W.Txx, Gindex, zstep, zinc);
								pd.DzTyy[Gindex] = DRPFD( W.Tyy, Gindex, zstep, zinc);
								pd.DzTxy[Gindex] = DRPFD( W.Txy, Gindex, zstep, zinc);
#endif
							}

							pd.DxTxx[Gindex] = DRPFD( W.Txx, Gindex, xstep, xinc);
							pd.DxTxy[Gindex] = DRPFD( W.Txy, Gindex, xstep, xinc);
							pd.DxTxz[Gindex] = DRPFD( W.Txz, Gindex, xstep, xinc);
							pd.DxVx[Gindex] = DRPFD( W.Vx, Gindex, xstep, xinc);
							pd.DxVy[Gindex] = DRPFD( W.Vy, Gindex, xstep, xinc);
							pd.DxVz[Gindex] = DRPFD( W.Vz, Gindex, xstep, xinc);

							pd.DyTyy[Gindex] = DRPFD( W.Tyy, Gindex, ystep, yinc);
							pd.DyTxy[Gindex] = DRPFD( W.Txy, Gindex, ystep, yinc);
							pd.DyTyz[Gindex] = DRPFD( W.Tyz, Gindex, ystep, yinc);
							pd.DyVx[Gindex] = DRPFD( W.Vx, Gindex, ystep, yinc);
							pd.DyVy[Gindex] = DRPFD( W.Vy, Gindex, ystep, yinc);
							pd.DyVz[Gindex] = DRPFD( W.Vz, Gindex, ystep, yinc);

							pd.DzTzz[Gindex] = DRPFD( W.Tzz, Gindex, zstep, zinc);
							pd.DzTxz[Gindex] = DRPFD( W.Txz, Gindex, zstep, zinc);
							pd.DzTyz[Gindex] = DRPFD( W.Tyz, Gindex, zstep, zinc);
#ifndef CondFree //no free surface == full space == should apply ABS							
							pd.DzVx[Gindex] = DRPFD( W.Vx, Gindex, zstep, zinc);
							pd.DzVy[Gindex] = DRPFD( W.Vy, Gindex, zstep, zinc);
							pd.DzVz[Gindex] = DRPFD( W.Vz, Gindex, zstep, zinc);
#endif

							//  P(V3)/P(zeta), for VLOW, should deal with 3 top layer and other layer, totally 4 cases.
							//		   for VUCD, should deal with 3 top layer and other layer, totally 4 cases.
							//		   for Default, only deal with 1 top layer and other layer, totally 2 cases.
							//		   VLOW and VUCD, pick one!


#ifndef CondFreeVUCD//Velocity free surface condition---Unilateral compact difference
#ifdef CondFreeVLOW
							if(idz == ipam[8]+LenFD-1)//surface layer
							{
								//202
						pd.DzVx[Gindex] = CoVx[xiaoI+0]*pd.DxVx[Gindex] + CoVx[xiaoI+1]*pd.DxVy[Gindex] + CoVx[xiaoI+2]*pd.DxVz[Gindex]
								+ CoVy[xiaoI+0]*pd.DyVx[Gindex] + CoVy[xiaoI+1]*pd.DyVy[Gindex] + CoVy[xiaoI+2]*pd.DyVz[Gindex];
						pd.DzVy[Gindex] = CoVx[xiaoI+3]*pd.DxVx[Gindex] + CoVx[xiaoI+4]*pd.DxVy[Gindex] + CoVx[xiaoI+5]*pd.DxVz[Gindex]
								+ CoVy[xiaoI+3]*pd.DyVx[Gindex] + CoVy[xiaoI+4]*pd.DyVy[Gindex] + CoVy[xiaoI+5]*pd.DyVz[Gindex];
						pd.DzVz[Gindex] = CoVx[xiaoI+6]*pd.DxVx[Gindex] + CoVx[xiaoI+7]*pd.DxVy[Gindex] + CoVx[xiaoI+8]*pd.DxVz[Gindex]
								+ CoVy[xiaoI+6]*pd.DyVx[Gindex] + CoVy[xiaoI+7]*pd.DyVy[Gindex] + CoVy[xiaoI+8]*pd.DyVz[Gindex];
							}
							else if(idz == ipam[8]+LenFD-2)//one layer inner surface
							{
								//201
								pd.DzVx[Gindex] = M22FD( W.Vx, Gindex, zstep, zinc);
								pd.DzVy[Gindex] = M22FD( W.Vy, Gindex, zstep, zinc);
								pd.DzVz[Gindex] = M22FD( W.Vz, Gindex, zstep, zinc);
							}
							else if(idz == ipam[8]+LenFD-3)//two layer inner surface
							{
								//200
								pd.DzVx[Gindex] = M24FD( W.Vx, Gindex, zstep, zinc);
								pd.DzVy[Gindex] = M24FD( W.Vy, Gindex, zstep, zinc);
								pd.DzVz[Gindex] = M24FD( W.Vz, Gindex, zstep, zinc);
							}
							else
							{
								pd.DzVx[Gindex] = DRPFD( W.Vx, Gindex, zstep, zinc);
								pd.DzVy[Gindex] = DRPFD( W.Vy, Gindex, zstep, zinc);
								pd.DzVz[Gindex] = DRPFD( W.Vz, Gindex, zstep, zinc);
							}
#endif// end of with VLOW
#endif// end of without VUCD

#if !defined(CondFreeVLOW) && !defined(CondFreeVUCD)
#ifdef CondFree
							//if there is a free surface condition
							//the Dz in top surface must be accquired by Dx and Dy
							if(idz == ipam[8]+LenFD-1)//surface layer
							{
						pd.DzVx[Gindex] = CoVx[xiaoI+0]*pd.DxVx[Gindex] + CoVx[xiaoI+1]*pd.DxVy[Gindex] + CoVx[xiaoI+2]*pd.DxVz[Gindex]
								+ CoVy[xiaoI+0]*pd.DyVx[Gindex] + CoVy[xiaoI+1]*pd.DyVy[Gindex] + CoVy[xiaoI+2]*pd.DyVz[Gindex];
						pd.DzVy[Gindex] = CoVx[xiaoI+3]*pd.DxVx[Gindex] + CoVx[xiaoI+4]*pd.DxVy[Gindex] + CoVx[xiaoI+5]*pd.DxVz[Gindex]
								+ CoVy[xiaoI+3]*pd.DyVx[Gindex] + CoVy[xiaoI+4]*pd.DyVy[Gindex] + CoVy[xiaoI+5]*pd.DyVz[Gindex];
						pd.DzVz[Gindex] = CoVx[xiaoI+6]*pd.DxVx[Gindex] + CoVx[xiaoI+7]*pd.DxVy[Gindex] + CoVx[xiaoI+8]*pd.DxVz[Gindex]
								+ CoVy[xiaoI+6]*pd.DyVx[Gindex] + CoVy[xiaoI+7]*pd.DyVy[Gindex] + CoVy[xiaoI+8]*pd.DyVz[Gindex];
							}
							else
							{
								pd.DzVx[Gindex] = DRPFD( W.Vx, Gindex, zstep, zinc);
								pd.DzVy[Gindex] = DRPFD( W.Vy, Gindex, zstep, zinc);
								pd.DzVz[Gindex] = DRPFD( W.Vz, Gindex, zstep, zinc);
							}
#endif//end of define CondFree(except vlow and vucd)
#endif//end of doesnot define VLOW and VUCD


						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

__global__ void CalWave(int ConIndex, derivF drv, mdparF mpa, PartialD pd, apara apr, Real *CoVx, Real *CoVy, wfield hW,
			  wfield Ax, wfield hAx, wfield Ay, wfield hAy, wfield Az, wfield hAz)
{
	//int i,j,k;
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
#ifdef HYindex	
	int Hyindex;
#endif	

	Real lambda,miu,rho,lam2mu;
	Real xix,xiy,xiz, etx,ety,etz, ztx,zty,ztz;//covariants
	
	Real DxiVx,DetVx,DztVx, DxiVy,DetVy,DztVy, DxiVz,DetVz,DztVz;
	Real DxiTxx,DetTxx,DztTxx, DxiTyy,DetTyy,DztTyy, DxiTzz,DetTzz,DztTzz;
	Real DxiTxy,DetTxy,DztTxy, DxiTxz,DetTxz,DztTxz, DxiTyz,DetTyz,DztTyz;
	Real Bx,By,Bz;//absorb boundary pars

#ifdef CFSPML
	Real APDx,APDy,APDz, DBx,DBy,DBz;
	int Pidx,tempIdx;
#ifdef CondFree	
	Real DzVx1,DzVx2, DzVy1,DzVy2, DzVz1,DzVz2;
	int xiaoI;
#endif	
#endif



	//the time-domain derivative is get by two equations, the momentum equation and the genaralized hooke's equation, 
	//which is Equation 2.20 and 2.21 respectively. And those two equation will also apply to the TIMG and VUCD free 
	//surface conditions.

	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current device compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						//if(idz<ipam[8]+LenFD && idz>=ConIndex)//contain convers interface
						if(idz<ipam[8]+LenFD)//contain convers interface
						{

							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
							
							rho = mpa.rho[Gindex];
							miu = rho*mpa.beta[Gindex]*mpa.beta[Gindex];
							lam2mu = rho*mpa.alpha[Gindex]*mpa.alpha[Gindex];
							lambda = lam2mu - 2.0*miu;
							rho = 1.0/rho;

							xix = drv.xix[Gindex];
							ety = drv.etay[Gindex];
							ztz = drv.zetaz[Gindex];
							
							if(idz>=ConIndex)
							{
								xiy = drv.xiy[Gindex];
								xiz = drv.xiz[Gindex];
								etx = drv.etax[Gindex];
								etz = drv.etaz[Gindex];
								ztx = drv.zetax[Gindex];
								zty = drv.zetay[Gindex];
#ifdef HYindex
								Hyindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+LenFD-ConIndex) 
									+ idy*(ipam[8]+LenFD-ConIndex) + idz-ConIndex;
								//hVx
								DxiVx = (xix*pd.DxTxx[Gindex] + xiy*pd.DxTxy[Gindex] + xiz*pd.DxTxz[Gindex])*rho;
								DetVx = (etx*pd.DyTxx[Hyindex] + ety*pd.DyTxy[Gindex] + etz*pd.DyTxz[Hyindex])*rho;//HYGRID
								DztVx = (ztx*pd.DzTxx[Hyindex] + zty*pd.DzTxy[Hyindex] + ztz*pd.DzTxz[Gindex])*rho;
								
								//hVy
								DxiVy = (xix*pd.DxTxy[Gindex] + xiy*pd.DxTyy[Hyindex] + xiz*pd.DxTyz[Hyindex])*rho;//HYGRID
								DetVy = (etx*pd.DyTxy[Gindex] + ety*pd.DyTyy[Gindex] + etz*pd.DyTyz[Gindex])*rho;
								DztVy = (ztx*pd.DzTxy[Hyindex] + zty*pd.DzTyy[Hyindex] + ztz*pd.DzTyz[Gindex])*rho;
								
								//hVz
								DxiVz = (xix*pd.DxTxz[Gindex] + xiy*pd.DxTyz[Hyindex] + xiz*pd.DxTzz[Hyindex])*rho;//HYGRID
								DetVz = (etx*pd.DyTxz[Hyindex] + ety*pd.DyTyz[Gindex] + etz*pd.DyTzz[Hyindex])*rho;//HYGRID
								DztVz = (ztx*pd.DzTxz[Gindex] + zty*pd.DzTyz[Gindex] + ztz*pd.DzTzz[Gindex])*rho;
#else								
								//hVx
								DxiVx = (xix*pd.DxTxx[Gindex] + xiy*pd.DxTxy[Gindex] + xiz*pd.DxTxz[Gindex])*rho;
								DetVx = (etx*pd.DyTxx[Gindex] + ety*pd.DyTxy[Gindex] + etz*pd.DyTxz[Gindex])*rho;
								DztVx = (ztx*pd.DzTxx[Gindex] + zty*pd.DzTxy[Gindex] + ztz*pd.DzTxz[Gindex])*rho;
								
								//hVy
								DxiVy = (xix*pd.DxTxy[Gindex] + xiy*pd.DxTyy[Gindex] + xiz*pd.DxTyz[Gindex])*rho;
								DetVy = (etx*pd.DyTxy[Gindex] + ety*pd.DyTyy[Gindex] + etz*pd.DyTyz[Gindex])*rho;
								DztVy = (ztx*pd.DzTxy[Gindex] + zty*pd.DzTyy[Gindex] + ztz*pd.DzTyz[Gindex])*rho;
								
								//hVz
								DxiVz = (xix*pd.DxTxz[Gindex] + xiy*pd.DxTyz[Gindex] + xiz*pd.DxTzz[Gindex])*rho;
								DetVz = (etx*pd.DyTxz[Gindex] + ety*pd.DyTyz[Gindex] + etz*pd.DyTzz[Gindex])*rho;
								DztVz = (ztx*pd.DzTxz[Gindex] + zty*pd.DzTyz[Gindex] + ztz*pd.DzTzz[Gindex])*rho;
#endif


								//hTxx
								DxiTxx = lam2mu*xix*pd.DxVx[Gindex] + lambda*xiy*pd.DxVy[Gindex] + lambda*xiz*pd.DxVz[Gindex];
								DetTxx = lam2mu*etx*pd.DyVx[Gindex] + lambda*ety*pd.DyVy[Gindex] + lambda*etz*pd.DyVz[Gindex];
								DztTxx = lam2mu*ztx*pd.DzVx[Gindex] + lambda*zty*pd.DzVy[Gindex] + lambda*ztz*pd.DzVz[Gindex];

								//hTyy
								DxiTyy = lambda*xix*pd.DxVx[Gindex] + lam2mu*xiy*pd.DxVy[Gindex] + lambda*xiz*pd.DxVz[Gindex];
								DetTyy = lambda*etx*pd.DyVx[Gindex] + lam2mu*ety*pd.DyVy[Gindex] + lambda*etz*pd.DyVz[Gindex];
								DztTyy = lambda*ztx*pd.DzVx[Gindex] + lam2mu*zty*pd.DzVy[Gindex] + lambda*ztz*pd.DzVz[Gindex];

								//hTzz
								DxiTzz = lambda*xix*pd.DxVx[Gindex] + lambda*xiy*pd.DxVy[Gindex] + lam2mu*xiz*pd.DxVz[Gindex];
								DetTzz = lambda*etx*pd.DyVx[Gindex] + lambda*ety*pd.DyVy[Gindex] + lam2mu*etz*pd.DyVz[Gindex];
								DztTzz = lambda*ztx*pd.DzVx[Gindex] + lambda*zty*pd.DzVy[Gindex] + lam2mu*ztz*pd.DzVz[Gindex];

								//hTxy
								DxiTxy = (xiy*pd.DxVx[Gindex] + xix*pd.DxVy[Gindex])*miu;
								DetTxy = (ety*pd.DyVx[Gindex] + etx*pd.DyVy[Gindex])*miu;
								DztTxy = (zty*pd.DzVx[Gindex] + ztx*pd.DzVy[Gindex])*miu;

								//hTxz
								DxiTxz = (xiz*pd.DxVx[Gindex] + xix*pd.DxVz[Gindex])*miu;
								DetTxz = (etz*pd.DyVx[Gindex] + etx*pd.DyVz[Gindex])*miu;
								DztTxz = (ztz*pd.DzVx[Gindex] + ztx*pd.DzVz[Gindex])*miu;

								//hTyz
								DxiTyz = (xiz*pd.DxVy[Gindex] + xiy*pd.DxVz[Gindex])*miu;
								DetTyz = (etz*pd.DyVy[Gindex] + ety*pd.DyVz[Gindex])*miu;
								DztTyz = (ztz*pd.DzVy[Gindex] + zty*pd.DzVz[Gindex])*miu;
							}
							else
							{
								//hVx
								DxiVx = rho*xix*pd.DxTxx[Gindex];
								DetVx = rho*ety*pd.DyTxy[Gindex];
								DztVx = ztz*pd.DzTxz[Gindex]*rho;

								//hVy
								DxiVy = rho*xix*pd.DxTxy[Gindex];
								DetVy = rho*ety*pd.DyTyy[Gindex];
								DztVy = ztz*pd.DzTyz[Gindex]*rho;

								//hVz
								DxiVz = rho*xix*pd.DxTxz[Gindex];
								DetVz = rho*ety*pd.DyTyz[Gindex];
								DztVz = ztz*pd.DzTzz[Gindex]*rho;

								//hTxx
								DxiTxx = lam2mu*xix*pd.DxVx[Gindex];
								DetTxx = lambda*ety*pd.DyVy[Gindex];
								DztTxx = lambda*ztz*pd.DzVz[Gindex];

								//hTyy
								DxiTyy = lambda*xix*pd.DxVx[Gindex];
								DetTyy = lam2mu*ety*pd.DyVy[Gindex];
								DztTyy = lambda*ztz*pd.DzVz[Gindex];

								//hTzz
								DxiTzz = lambda*xix*pd.DxVx[Gindex];
								DetTzz = lambda*ety*pd.DyVy[Gindex];
								DztTzz = lam2mu*ztz*pd.DzVz[Gindex];

								//hTxy
								DxiTxy = miu*xix*pd.DxVy[Gindex];
								DetTxy = miu*ety*pd.DyVx[Gindex];
								DztTxy = 0.0;

								//hTxz
								DxiTxz = miu*xix*pd.DxVz[Gindex];
								DetTxz = 0.0;
								DztTxz = ztz*pd.DzVx[Gindex]*miu;

								//hTyz
								DxiTyz = 0.0;
								DetTyz = miu*ety*pd.DyVz[Gindex];
								DztTyz = ztz*pd.DzVy[Gindex]*miu;
							}


#ifdef CFSPML
	APDx = apr.APDx[idx];	APDy = apr.APDy[idy];	APDz = apr.APDz[idz];
	Bx = apr.Bx[idx];	By = apr.By[idy];	Bz = apr.Bz[idz];
	DBx = apr.DBx[idx];	DBy = apr.DBy[idy];	DBz = apr.DBz[idz];
#else	
	Bx = 1.0;	By = 1.0;	Bz = 1.0;
#endif

							//time domain partial derivative--->wave field
							hW.Txx[Gindex] = DxiTxx/Bx + DetTxx/By + DztTxx/Bz;
							hW.Tyy[Gindex] = DxiTyy/Bx + DetTyy/By + DztTyy/Bz;
							hW.Tzz[Gindex] = DxiTzz/Bx + DetTzz/By + DztTzz/Bz;
							hW.Txy[Gindex] = DxiTxy/Bx + DetTxy/By + DztTxy/Bz;
							hW.Txz[Gindex] = DxiTxz/Bx + DetTxz/By + DztTxz/Bz;
							hW.Tyz[Gindex] = DxiTyz/Bx + DetTyz/By + DztTyz/Bz;
							hW.Vx[Gindex] = DxiVx/Bx + DetVx/By + DztVx/Bz;
							hW.Vy[Gindex] = DxiVy/Bx + DetVy/By + DztVy/Bz;
							hW.Vz[Gindex] = DxiVz/Bx + DetVz/By + DztVz/Bz;

#ifdef DisBug
//if(zbx == idx+(ipam[2]-LenFD)+ipam[9] && zby == idy+(ipam[4]-LenFD) && zbz == idz)
//	printf("at PCS[%d]DEV[%d](%d,%d,%d),CalWave->hW.Txx=%e, DxiTxx=%e, DetTxx=%e, DztTxx=%e, Bx=%e,By=%e,Bz=%e\n",
//		ipam[2],ipam[1], zbx,zby,zbz, hW.Txx[Gindex], DxiTxx,DetTxx,DztTxx, Bx,By,Bz);
#endif

#ifdef CFSPML
#ifdef CondFree
							//top surface partial derivative conversion
							xiaoI=idx*(ipam[5]-ipam[4]+1+2*LenFD)*SeisGeo*SeisGeo + idy*SeisGeo*SeisGeo;//valid Y
							
							if(idz == ipam[8]+LenFD-1)//surface layer
							{
						DzVx1 = CoVx[xiaoI+0]*pd.DxVx[Gindex] + CoVx[xiaoI+1]*pd.DxVy[Gindex] + CoVx[xiaoI+2]*pd.DxVz[Gindex];
						DzVx2 = CoVy[xiaoI+0]*pd.DyVx[Gindex] + CoVy[xiaoI+1]*pd.DyVy[Gindex] + CoVy[xiaoI+2]*pd.DyVz[Gindex];
						DzVy1 = CoVx[xiaoI+3]*pd.DxVx[Gindex] + CoVx[xiaoI+4]*pd.DxVy[Gindex] + CoVx[xiaoI+5]*pd.DxVz[Gindex];
						DzVy2 = CoVy[xiaoI+3]*pd.DyVx[Gindex] + CoVy[xiaoI+4]*pd.DyVy[Gindex] + CoVy[xiaoI+5]*pd.DyVz[Gindex];
						DzVz1 = CoVx[xiaoI+6]*pd.DxVx[Gindex] + CoVx[xiaoI+7]*pd.DxVy[Gindex] + CoVx[xiaoI+8]*pd.DxVz[Gindex];
						DzVz2 = CoVy[xiaoI+6]*pd.DyVx[Gindex] + CoVy[xiaoI+7]*pd.DyVy[Gindex] + CoVy[xiaoI+8]*pd.DyVz[Gindex];
							}
#endif							
							tempIdx = idx+(ipam[2]-LenFD)+ipam[9];//idx+ipam[9]
							if(tempIdx<=apr.nabs[0]+LenFD-1 || tempIdx>=ipam[10]+LenFD-apr.nabs[1])//X-dir
							{
						tempIdx<apr.nabs[0]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[10]+LenFD-apr.nabs[1])+apr.nabs[0];
						Pidx = Pidx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
						
						//wavefield attenuation (Equation 14 of Zhang 2010)
						hW.Vx[Gindex]  -= Ax.Vx[Pidx]/Bx;  
						hW.Vy[Gindex]  -= Ax.Vy[Pidx]/Bx;  
						hW.Vz[Gindex]  -= Ax.Vz[Pidx]/Bx;
						hW.Txx[Gindex] -= Ax.Txx[Pidx]/Bx;
						hW.Tyy[Gindex] -= Ax.Tyy[Pidx]/Bx;
						hW.Tzz[Gindex] -= Ax.Tzz[Pidx]/Bx;
						hW.Txy[Gindex] -= Ax.Txy[Pidx]/Bx;
						hW.Txz[Gindex] -= Ax.Txz[Pidx]/Bx;
						hW.Tyz[Gindex] -= Ax.Tyz[Pidx]/Bx;
						
						//ADE update (Equation A10 of Zhang 2010)
						hAx.Vx[Pidx]  = DxiVx*DBx - APDx*Ax.Vx[Pidx];
						hAx.Vy[Pidx]  = DxiVy*DBx - APDx*Ax.Vy[Pidx];
						hAx.Vz[Pidx]  = DxiVz*DBx - APDx*Ax.Vz[Pidx];
						hAx.Txx[Pidx] = DxiTxx*DBx - APDx*Ax.Txx[Pidx];
						hAx.Tyy[Pidx] = DxiTyy*DBx - APDx*Ax.Tyy[Pidx];
						hAx.Tzz[Pidx] = DxiTzz*DBx - APDx*Ax.Tzz[Pidx];
						hAx.Txy[Pidx] = DxiTxy*DBx - APDx*Ax.Txy[Pidx];
						hAx.Txz[Pidx] = DxiTxz*DBx - APDx*Ax.Txz[Pidx];
						hAx.Tyz[Pidx] = DxiTyz*DBx - APDx*Ax.Tyz[Pidx];

#ifdef CondFree
						//top surface 
								if(idz == ipam[8]+LenFD-1)
								{
							hAx.Txx[Pidx] += DBx*Bx*( lam2mu*ztx*DzVx1 + lambda*zty*DzVy1 + lambda*ztz*DzVz1);
							hAx.Tyy[Pidx] += DBx*Bx*( lambda*ztx*DzVx1 + lam2mu*zty*DzVy1 + lambda*ztz*DzVz1);
							hAx.Tzz[Pidx] += DBx*Bx*( lambda*ztx*DzVx1 + lambda*zty*DzVy1 + lam2mu*ztz*DzVz1);
							hAx.Txy[Pidx] += DBx*Bx*( zty*DzVx1 + ztx*DzVy1 )*miu; 
							hAx.Txz[Pidx] += DBx*Bx*( ztz*DzVx1 + ztx*DzVz1 )*miu; 
							hAx.Tyz[Pidx] += DBx*Bx*( ztz*DzVy1 + zty*DzVz1 )*miu; 
								}
#endif						

							}
							
							tempIdx = idy + (ipam[4]-LenFD);//idy
							if(tempIdx<=apr.nabs[2]+LenFD-1 || tempIdx>=ipam[7]+LenFD-apr.nabs[3])//Y-dir
							{
						tempIdx<apr.nabs[2]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[7]+LenFD-apr.nabs[3])+apr.nabs[2];		
						Pidx = Pidx*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[8]+2*LenFD) + idx*(ipam[8]+2*LenFD) + idz;		
						
						//wavefield attenuation (Equation 14 of Zhang 2010)
						hW.Vx[Gindex]  -= Ay.Vx[Pidx]/By;  
						hW.Vy[Gindex]  -= Ay.Vy[Pidx]/By;  
						hW.Vz[Gindex]  -= Ay.Vz[Pidx]/By;
						hW.Txx[Gindex] -= Ay.Txx[Pidx]/By;
						hW.Tyy[Gindex] -= Ay.Tyy[Pidx]/By;
						hW.Tzz[Gindex] -= Ay.Tzz[Pidx]/By;
						hW.Txy[Gindex] -= Ay.Txy[Pidx]/By;
						hW.Txz[Gindex] -= Ay.Txz[Pidx]/By;
						hW.Tyz[Gindex] -= Ay.Tyz[Pidx]/By;
						
						//ADE update (Equation A10 of Zhang 2010)
						hAy.Vx[Pidx]  = DetVx*DBy - APDy*Ay.Vx[Pidx];
						hAy.Vy[Pidx]  = DetVy*DBy - APDy*Ay.Vy[Pidx];
						hAy.Vz[Pidx]  = DetVz*DBy - APDy*Ay.Vz[Pidx];
						hAy.Txx[Pidx] = DetTxx*DBy - APDy*Ay.Txx[Pidx];
						hAy.Tyy[Pidx] = DetTyy*DBy - APDy*Ay.Tyy[Pidx];
						hAy.Tzz[Pidx] = DetTzz*DBy - APDy*Ay.Tzz[Pidx];
						hAy.Txy[Pidx] = DetTxy*DBy - APDy*Ay.Txy[Pidx];
						hAy.Txz[Pidx] = DetTxz*DBy - APDy*Ay.Txz[Pidx];
						hAy.Tyz[Pidx] = DetTyz*DBy - APDy*Ay.Tyz[Pidx];

#ifdef CondFree
						//top surface 
								if(idz == ipam[8]+LenFD-1)
								{
							hAy.Txx[Pidx] += DBy*By*( lam2mu*ztx*DzVx2 + lambda*zty*DzVy2 + lambda*ztz*DzVz2);
							hAy.Tyy[Pidx] += DBy*By*( lambda*ztx*DzVx2 + lam2mu*zty*DzVy2 + lambda*ztz*DzVz2);
							hAy.Tzz[Pidx] += DBy*By*( lambda*ztx*DzVx2 + lambda*zty*DzVy2 + lam2mu*ztz*DzVz2);
							hAy.Txy[Pidx] += DBy*By*( zty*DzVx2 + ztx*DzVy2 )*miu; 
							hAy.Txz[Pidx] += DBy*By*( ztz*DzVx2 + ztx*DzVz2 )*miu; 
							hAy.Tyz[Pidx] += DBy*By*( ztz*DzVy2 + zty*DzVz2 )*miu; 
								}
#endif						
							
							}


							if(idz<=apr.nabs[4]+LenFD-1 || idz>=ipam[8]+LenFD-apr.nabs[5])//Z-dir
							{
						idz<apr.nabs[4]+LenFD ? Pidx=idz-LenFD : Pidx=idz-(ipam[8]+LenFD-apr.nabs[5])+apr.nabs[4];		
						Pidx = Pidx*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[5]-ipam[4]+1+2*LenFD) + idx*(ipam[5]-ipam[4]+1+2*LenFD) + idy;		
							
						//wavefield attenuation (Equation 14 of Zhang 2010)
						hW.Vx[Gindex]  -= Az.Vx[Pidx]/Bz;  
						hW.Vy[Gindex]  -= Az.Vy[Pidx]/Bz;  
						hW.Vz[Gindex]  -= Az.Vz[Pidx]/Bz;
						hW.Txx[Gindex] -= Az.Txx[Pidx]/Bz;
						hW.Tyy[Gindex] -= Az.Tyy[Pidx]/Bz;
						hW.Tzz[Gindex] -= Az.Tzz[Pidx]/Bz;
						hW.Txy[Gindex] -= Az.Txy[Pidx]/Bz;
						hW.Txz[Gindex] -= Az.Txz[Pidx]/Bz;
						hW.Tyz[Gindex] -= Az.Tyz[Pidx]/Bz;
						
						//ADE update (Equation A10 of Zhang 2010)
						hAz.Vx[Pidx]  = DztVx*DBz - APDz*Az.Vx[Pidx];
						hAz.Vy[Pidx]  = DztVy*DBz - APDz*Az.Vy[Pidx];
						hAz.Vz[Pidx]  = DztVz*DBz - APDz*Az.Vz[Pidx];
						hAz.Txx[Pidx] = DztTxx*DBz - APDz*Az.Txx[Pidx];
						hAz.Tyy[Pidx] = DztTyy*DBz - APDz*Az.Tyy[Pidx];
						hAz.Tzz[Pidx] = DztTzz*DBz - APDz*Az.Tzz[Pidx];
						hAz.Txy[Pidx] = DztTxy*DBz - APDz*Az.Txy[Pidx];
						hAz.Txz[Pidx] = DztTxz*DBz - APDz*Az.Txz[Pidx];
						hAz.Tyz[Pidx] = DztTyz*DBz - APDz*Az.Tyz[Pidx];
							
							}

#endif

						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

__global__ void CalDiffCL(int Xvec, int Yvec, int Zvec, int ConIndex, Real steph, Real *CoVx, Real* CoVy, wfield W, PartialD pd)
{
	//int i,j,k;
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int xiaoI;
#ifdef HYindex	
	int Hyindex;
#endif

	Real xstep, ystep, zstep;
	int xinc, yinc, zinc;
	
	xstep = steph*Xvec;
	ystep = steph*Yvec;
	zstep = steph*Zvec;
	xinc = Xvec*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD);//skip cdx.ny*cdx.nz
	yinc = Yvec*(ipam[8]+2*LenFD);//skip cdx.nz
	zinc = Zvec*1;//skip 1


	//generally use DRP/opt MacCormack scheme to get derivative, as Equation 2.23 and coefficients is Equation 2.24 in Thesis.
	//for the top layer transfrom the derivative of xi and eta to get zeta direction derivative, as Equation 3.4 in Thesis.
	
	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current device compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<ipam[8]+LenFD && idz>=ConIndex)//vaild point with one virtual bounds
						{

							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
							xiaoI = idx*(ipam[5]-ipam[4]+1+2*LenFD)*SeisGeo*SeisGeo + idy*SeisGeo*SeisGeo;//valid Y

#ifdef DisBug
/*
if( idx+(ipam[2]-LenFD)+ipam[9] >=96  && idx+(ipam[2]-LenFD)+ipam[9] <=100&& zby == idy+(ipam[4]-LenFD) && zbz == idz)
{
	printf(" -->W.Vy(%d,%d,%d)=%e\n",idx+(ipam[2]-LenFD)+ipam[9],zby,zbz,W.Vy[Gindex]);
}
*/
if( idx+(ipam[2]-LenFD)+ipam[9] ==zbx && zby == idy+(ipam[4]-LenFD) && idz>=228 && idz<=233)
{
	printf(" -->(%d,%d,%d) Txx=%e, Tyy=%e, Tzz=%e\n\tTxy=%e, Txz=%e,Tzz=%e\n\tVx=%e Vy=%e Vz=%e\n",
		zbx,zby,idz,W.Txx[Gindex],W.Tyy[Gindex],W.Tzz[Gindex],W.Txy[Gindex],W.Txz[Gindex],W.Tyz[Gindex],W.Vx[Gindex],W.Vy[Gindex],W.Vz[Gindex]);
}

#endif

#ifdef HYindex
							//with Hyindex
							Hyindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+LenFD-ConIndex) 
								+ idy*(ipam[8]+LenFD-ConIndex) + idz-ConIndex;
							
							pd.DxTyy[Hyindex] = DRPFD( W.Tyy, Gindex, xstep, xinc);
							pd.DxTzz[Hyindex] = DRPFD( W.Tzz, Gindex, xstep, xinc);
						        pd.DxTyz[Hyindex] = DRPFD( W.Tyz, Gindex, xstep, xinc);
							pd.DyTxx[Hyindex] = DRPFD( W.Txx, Gindex, ystep, yinc);
							pd.DyTzz[Hyindex] = DRPFD( W.Tzz, Gindex, ystep, yinc);
							pd.DyTxz[Hyindex] = DRPFD( W.Txz, Gindex, ystep, yinc);
							pd.DzTxx[Hyindex] = DRPFD( W.Txx, Gindex, zstep, zinc);
							pd.DzTyy[Hyindex] = DRPFD( W.Tyy, Gindex, zstep, zinc);
							pd.DzTxy[Hyindex] = DRPFD( W.Txy, Gindex, zstep, zinc);
#else
							//with Gindex
							pd.DxTyy[Gindex] = DRPFD( W.Tyy, Gindex, xstep, xinc);
							pd.DxTzz[Gindex] = DRPFD( W.Tzz, Gindex, xstep, xinc);
							pd.DxTyz[Gindex] = DRPFD( W.Tyz, Gindex, xstep, xinc);
							pd.DyTxx[Gindex] = DRPFD( W.Txx, Gindex, ystep, yinc);
							pd.DyTzz[Gindex] = DRPFD( W.Tzz, Gindex, ystep, yinc);
							pd.DyTxz[Gindex] = DRPFD( W.Txz, Gindex, ystep, yinc);
							pd.DzTxx[Gindex] = DRPFD( W.Txx, Gindex, zstep, zinc);
							pd.DzTyy[Gindex] = DRPFD( W.Tyy, Gindex, zstep, zinc);
							pd.DzTxy[Gindex] = DRPFD( W.Txy, Gindex, zstep, zinc);
#endif

							pd.DxTxx[Gindex] = DRPFD( W.Txx, Gindex, xstep, xinc);
							pd.DxTxy[Gindex] = DRPFD( W.Txy, Gindex, xstep, xinc);
							pd.DxTxz[Gindex] = DRPFD( W.Txz, Gindex, xstep, xinc);
							pd.DxVx[Gindex] = DRPFD( W.Vx, Gindex, xstep, xinc);
							pd.DxVy[Gindex] = DRPFD( W.Vy, Gindex, xstep, xinc);
							pd.DxVz[Gindex] = DRPFD( W.Vz, Gindex, xstep, xinc);

							pd.DyTyy[Gindex] = DRPFD( W.Tyy, Gindex, ystep, yinc);
							pd.DyTxy[Gindex] = DRPFD( W.Txy, Gindex, ystep, yinc);
							pd.DyTyz[Gindex] = DRPFD( W.Tyz, Gindex, ystep, yinc);
							pd.DyVx[Gindex] = DRPFD( W.Vx, Gindex, ystep, yinc);
							pd.DyVy[Gindex] = DRPFD( W.Vy, Gindex, ystep, yinc);
							pd.DyVz[Gindex] = DRPFD( W.Vz, Gindex, ystep, yinc);

							pd.DzTzz[Gindex] = DRPFD( W.Tzz, Gindex, zstep, zinc);
							pd.DzTxz[Gindex] = DRPFD( W.Txz, Gindex, zstep, zinc);
							pd.DzTyz[Gindex] = DRPFD( W.Tyz, Gindex, zstep, zinc);
#ifndef CondFree //no free surface == full space == should apply ABS							
							pd.DzVx[Gindex] = DRPFD( W.Vx, Gindex, zstep, zinc);
							pd.DzVy[Gindex] = DRPFD( W.Vy, Gindex, zstep, zinc);
							pd.DzVz[Gindex] = DRPFD( W.Vz, Gindex, zstep, zinc);
#endif

#ifdef DisBug
/*
if(zbx == idx+(ipam[2]-LenFD)+ipam[9] && zby == idy+(ipam[4]-LenFD) && zbz == idz)
{
	printf("\tat PCS[%d]DEV[%d](%d,%d,%d),CalDiff--->DzTxx=%e, DzTyy=%e, DzTzz=%e, DzTxy=%e, DzTxz=%e, DzTyz=%e\n",
		ipam[2],ipam[1], zbx,zby,zbz, pd.DzTxx[Hyindex], pd.DzTyy[Hyindex], pd.DzTzz[Gindex],pd.DzTxy[Hyindex],pd.DzTxz[Gindex],pd.DzTyz[Gindex]);
	printf("\tat PCS[%d]DEV[%d](%d,%d,%d),CalDiff--->DxVx=%e, DxVy=%e, DxVz=%e,  Vx=%e, Vy=%e, Vz=%e\n",
		ipam[2],ipam[1], zbx,zby,zbz, pd.DxVx[Gindex], pd.DxVy[Gindex], pd.DxVz[Gindex],W.Vx[Gindex],W.Vy[Gindex],W.Vz[Gindex]);
}
*/
#endif

							//  P(V3)/P(zeta), for VLOW, should deal with 3 top layer and other layer, totally 4 cases.
							//		   for VUCD, should deal with 3 top layer and other layer, totally 4 cases.
							//		   for Default, only deal with 1 top layer and other layer, totally 2 cases.
							//		   VLOW and VUCD, pick one!


#ifndef CondFreeVUCD//Velocity free surface condition---Unilateral compact difference
#ifdef CondFreeVLOW
							if(idz == ipam[8]+LenFD-1)//surface layer
							{
								//202
						pd.DzVx[Gindex] = CoVx[xiaoI+0]*pd.DxVx[Gindex] + CoVx[xiaoI+1]*pd.DxVy[Gindex] + CoVx[xiaoI+2]*pd.DxVz[Gindex]
								+ CoVy[xiaoI+0]*pd.DyVx[Gindex] + CoVy[xiaoI+1]*pd.DyVy[Gindex] + CoVy[xiaoI+2]*pd.DyVz[Gindex];
						pd.DzVy[Gindex] = CoVx[xiaoI+3]*pd.DxVx[Gindex] + CoVx[xiaoI+4]*pd.DxVy[Gindex] + CoVx[xiaoI+5]*pd.DxVz[Gindex]
								+ CoVy[xiaoI+3]*pd.DyVx[Gindex] + CoVy[xiaoI+4]*pd.DyVy[Gindex] + CoVy[xiaoI+5]*pd.DyVz[Gindex];
						pd.DzVz[Gindex] = CoVx[xiaoI+6]*pd.DxVx[Gindex] + CoVx[xiaoI+7]*pd.DxVy[Gindex] + CoVx[xiaoI+8]*pd.DxVz[Gindex]
								+ CoVy[xiaoI+6]*pd.DyVx[Gindex] + CoVy[xiaoI+7]*pd.DyVy[Gindex] + CoVy[xiaoI+8]*pd.DyVz[Gindex];
							}
							else if(idz == ipam[8]+LenFD-2)//one layer inner surface
							{
								//201
								pd.DzVx[Gindex] = M22FD( W.Vx, Gindex, zstep, zinc);
								pd.DzVy[Gindex] = M22FD( W.Vy, Gindex, zstep, zinc);
								pd.DzVz[Gindex] = M22FD( W.Vz, Gindex, zstep, zinc);
							}
							else if(idz == ipam[8]+LenFD-3)//two layer inner surface
							{
								//200
								pd.DzVx[Gindex] = M24FD( W.Vx, Gindex, zstep, zinc);
								pd.DzVy[Gindex] = M24FD( W.Vy, Gindex, zstep, zinc);
								pd.DzVz[Gindex] = M24FD( W.Vz, Gindex, zstep, zinc);
							}
							else
							{
								pd.DzVx[Gindex] = DRPFD( W.Vx, Gindex, zstep, zinc);
								pd.DzVy[Gindex] = DRPFD( W.Vy, Gindex, zstep, zinc);
								pd.DzVz[Gindex] = DRPFD( W.Vz, Gindex, zstep, zinc);
							}
#endif// end of with VLOW
#endif// end of without VUCD

#if !defined(CondFreeVLOW) && !defined(CondFreeVUCD)
#ifdef CondFree
							//if there is a free surface condition
							//the Dz in top surface must be accquired by Dx and Dy
							if(idz == ipam[8]+LenFD-1)//surface layer
							{
						pd.DzVx[Gindex] = CoVx[xiaoI+0]*pd.DxVx[Gindex] + CoVx[xiaoI+1]*pd.DxVy[Gindex] + CoVx[xiaoI+2]*pd.DxVz[Gindex]
								+ CoVy[xiaoI+0]*pd.DyVx[Gindex] + CoVy[xiaoI+1]*pd.DyVy[Gindex] + CoVy[xiaoI+2]*pd.DyVz[Gindex];
						pd.DzVy[Gindex] = CoVx[xiaoI+3]*pd.DxVx[Gindex] + CoVx[xiaoI+4]*pd.DxVy[Gindex] + CoVx[xiaoI+5]*pd.DxVz[Gindex]
								+ CoVy[xiaoI+3]*pd.DyVx[Gindex] + CoVy[xiaoI+4]*pd.DyVy[Gindex] + CoVy[xiaoI+5]*pd.DyVz[Gindex];
						pd.DzVz[Gindex] = CoVx[xiaoI+6]*pd.DxVx[Gindex] + CoVx[xiaoI+7]*pd.DxVy[Gindex] + CoVx[xiaoI+8]*pd.DxVz[Gindex]
								+ CoVy[xiaoI+6]*pd.DyVx[Gindex] + CoVy[xiaoI+7]*pd.DyVy[Gindex] + CoVy[xiaoI+8]*pd.DyVz[Gindex];
							}
							else
							{
								pd.DzVx[Gindex] = DRPFD( W.Vx, Gindex, zstep, zinc);
								pd.DzVy[Gindex] = DRPFD( W.Vy, Gindex, zstep, zinc);
								pd.DzVz[Gindex] = DRPFD( W.Vz, Gindex, zstep, zinc);
							}
#endif//end of define CondFree(except vlow and vucd)
#endif//end of doesnot define VLOW and VUCD


						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}


__global__ void CalWaveCL(int ConIndex, derivF drv, mdparF mpa, PartialD pd, apara apr, Real *CoVx, Real *CoVy, wfield hW,
			  wfield Ax, wfield hAx, wfield Ay, wfield hAy, wfield Az, wfield hAz)
{
	//int i,j,k;
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
#ifdef HYindex	
	int Hyindex;
#endif

	Real lambda,miu,rho,lam2mu;
	Real xix,xiy,xiz, etx,ety,etz, ztx,zty,ztz;//covariants
	
	Real DxiVx,DetVx,DztVx, DxiVy,DetVy,DztVy, DxiVz,DetVz,DztVz;
	Real DxiTxx,DetTxx,DztTxx, DxiTyy,DetTyy,DztTyy, DxiTzz,DetTzz,DztTzz;
	Real DxiTxy,DetTxy,DztTxy, DxiTxz,DetTxz,DztTxz, DxiTyz,DetTyz,DztTyz;
	Real Bx,By,Bz;//absorb boundary pars

#ifdef CFSPML
	Real APDx,APDy,APDz, DBx,DBy,DBz;
	int Pidx,tempIdx;
#ifdef CondFree	
	Real DzVx1,DzVx2, DzVy1,DzVy2, DzVz1,DzVz2;
	int xiaoI;
#endif	
#endif

	//the time-domain derivative is get by two equations, the momentum equation and the genaralized hooke's equation, 
	//which is Equation 2.20 and 2.21 respectively. And those two equation will also apply to the TIMG and VUCD free 
	//surface conditions.

	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current device compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<ipam[8]+LenFD && idz>=ConIndex)//contain convers interface
						{

							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
							
							rho = mpa.rho[Gindex];
							miu = rho*mpa.beta[Gindex]*mpa.beta[Gindex];
							lam2mu = rho*mpa.alpha[Gindex]*mpa.alpha[Gindex];
							lambda = lam2mu - 2.0*miu;
							rho = 1.0/rho;

							xix = drv.xix[Gindex];
							xiy = drv.xiy[Gindex];
							xiz = drv.xiz[Gindex];
							etx = drv.etax[Gindex];
							ety = drv.etay[Gindex];
							etz = drv.etaz[Gindex];
							ztx = drv.zetax[Gindex];
							zty = drv.zetay[Gindex];
							ztz = drv.zetaz[Gindex];

							
							//useful for float, double error at 1e-13
#ifdef HYindex	
							Hyindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+LenFD-ConIndex) 
								+ idy*(ipam[8]+LenFD-ConIndex) + idz-ConIndex;
							
							//hVx
							DxiVx = (xix*pd.DxTxx[Gindex] + xiy*pd.DxTxy[Gindex] + xiz*pd.DxTxz[Gindex])*rho;
							DetVx = (etx*pd.DyTxx[Hyindex] + ety*pd.DyTxy[Gindex] + etz*pd.DyTxz[Hyindex])*rho;//HYGRID
							DztVx = (ztx*pd.DzTxx[Hyindex] + zty*pd.DzTxy[Hyindex] + ztz*pd.DzTxz[Gindex])*rho;

							//hVy
							DxiVy = (xix*pd.DxTxy[Gindex] + xiy*pd.DxTyy[Hyindex] + xiz*pd.DxTyz[Hyindex])*rho;//HYGRID
							DetVy = (etx*pd.DyTxy[Gindex] + ety*pd.DyTyy[Gindex] + etz*pd.DyTyz[Gindex])*rho;
							DztVy = (ztx*pd.DzTxy[Hyindex] + zty*pd.DzTyy[Hyindex] + ztz*pd.DzTyz[Gindex])*rho;

							//hVz
							DxiVz = (xix*pd.DxTxz[Gindex] + xiy*pd.DxTyz[Hyindex] + xiz*pd.DxTzz[Hyindex])*rho;//HYGRID
							DetVz = (etx*pd.DyTxz[Hyindex] + ety*pd.DyTyz[Gindex] + etz*pd.DyTzz[Hyindex])*rho;//HYGRID
							DztVz = (ztx*pd.DzTxz[Gindex] + zty*pd.DzTyz[Gindex] + ztz*pd.DzTzz[Gindex])*rho;
#else	
							//hVx
							DxiVx = (xix*pd.DxTxx[Gindex] + xiy*pd.DxTxy[Gindex] + xiz*pd.DxTxz[Gindex])*rho;
							DetVx = (etx*pd.DyTxx[Gindex] + ety*pd.DyTxy[Gindex] + etz*pd.DyTxz[Gindex])*rho;
							DztVx = (ztx*pd.DzTxx[Gindex] + zty*pd.DzTxy[Gindex] + ztz*pd.DzTxz[Gindex])*rho;

							//hVy
							DxiVy = (xix*pd.DxTxy[Gindex] + xiy*pd.DxTyy[Gindex] + xiz*pd.DxTyz[Gindex])*rho;
							DetVy = (etx*pd.DyTxy[Gindex] + ety*pd.DyTyy[Gindex] + etz*pd.DyTyz[Gindex])*rho;
							DztVy = (ztx*pd.DzTxy[Gindex] + zty*pd.DzTyy[Gindex] + ztz*pd.DzTyz[Gindex])*rho;

							//hVz
							DxiVz = (xix*pd.DxTxz[Gindex] + xiy*pd.DxTyz[Gindex] + xiz*pd.DxTzz[Gindex])*rho;
							DetVz = (etx*pd.DyTxz[Gindex] + ety*pd.DyTyz[Gindex] + etz*pd.DyTzz[Gindex])*rho;
							DztVz = (ztx*pd.DzTxz[Gindex] + zty*pd.DzTyz[Gindex] + ztz*pd.DzTzz[Gindex])*rho;
#endif

							//hTxx
							DxiTxx = lam2mu*xix*pd.DxVx[Gindex] + lambda*xiy*pd.DxVy[Gindex] + lambda*xiz*pd.DxVz[Gindex];
							DetTxx = lam2mu*etx*pd.DyVx[Gindex] + lambda*ety*pd.DyVy[Gindex] + lambda*etz*pd.DyVz[Gindex];
							DztTxx = lam2mu*ztx*pd.DzVx[Gindex] + lambda*zty*pd.DzVy[Gindex] + lambda*ztz*pd.DzVz[Gindex];

							//hTyy
							DxiTyy = lambda*xix*pd.DxVx[Gindex] + lam2mu*xiy*pd.DxVy[Gindex] + lambda*xiz*pd.DxVz[Gindex];
							DetTyy = lambda*etx*pd.DyVx[Gindex] + lam2mu*ety*pd.DyVy[Gindex] + lambda*etz*pd.DyVz[Gindex];
							DztTyy = lambda*ztx*pd.DzVx[Gindex] + lam2mu*zty*pd.DzVy[Gindex] + lambda*ztz*pd.DzVz[Gindex];

							//hTzz
							DxiTzz = lambda*xix*pd.DxVx[Gindex] + lambda*xiy*pd.DxVy[Gindex] + lam2mu*xiz*pd.DxVz[Gindex];
							DetTzz = lambda*etx*pd.DyVx[Gindex] + lambda*ety*pd.DyVy[Gindex] + lam2mu*etz*pd.DyVz[Gindex];
							DztTzz = lambda*ztx*pd.DzVx[Gindex] + lambda*zty*pd.DzVy[Gindex] + lam2mu*ztz*pd.DzVz[Gindex];

							//hTxy
							DxiTxy = (xiy*pd.DxVx[Gindex] + xix*pd.DxVy[Gindex])*miu;
							DetTxy = (ety*pd.DyVx[Gindex] + etx*pd.DyVy[Gindex])*miu;
							DztTxy = (zty*pd.DzVx[Gindex] + ztx*pd.DzVy[Gindex])*miu;

							//hTxz
							DxiTxz = (xiz*pd.DxVx[Gindex] + xix*pd.DxVz[Gindex])*miu;
							DetTxz = (etz*pd.DyVx[Gindex] + etx*pd.DyVz[Gindex])*miu;
							DztTxz = (ztz*pd.DzVx[Gindex] + ztx*pd.DzVz[Gindex])*miu;

							//hTyz
							DxiTyz = (xiz*pd.DxVy[Gindex] + xiy*pd.DxVz[Gindex])*miu;
							DetTyz = (etz*pd.DyVy[Gindex] + ety*pd.DyVz[Gindex])*miu;
							DztTyz = (ztz*pd.DzVy[Gindex] + zty*pd.DzVz[Gindex])*miu;

#ifdef CFSPML
	APDx = apr.APDx[idx];	APDy = apr.APDy[idy];	APDz = apr.APDz[idz];
	Bx = apr.Bx[idx];	By = apr.By[idy];	Bz = apr.Bz[idz];
	DBx = apr.DBx[idx];	DBy = apr.DBy[idy];	DBz = apr.DBz[idz];
#else	
	Bx = 1.0;	By = 1.0;	Bz = 1.0;
#endif

							//time domain partial derivative--->wave field
							hW.Txx[Gindex] = DxiTxx/Bx + DetTxx/By + DztTxx/Bz;
							hW.Tyy[Gindex] = DxiTyy/Bx + DetTyy/By + DztTyy/Bz;
							hW.Tzz[Gindex] = DxiTzz/Bx + DetTzz/By + DztTzz/Bz;
							hW.Txy[Gindex] = DxiTxy/Bx + DetTxy/By + DztTxy/Bz;
							hW.Txz[Gindex] = DxiTxz/Bx + DetTxz/By + DztTxz/Bz;
							hW.Tyz[Gindex] = DxiTyz/Bx + DetTyz/By + DztTyz/Bz;
							hW.Vx[Gindex] = DxiVx/Bx + DetVx/By + DztVx/Bz;
							hW.Vy[Gindex] = DxiVy/Bx + DetVy/By + DztVy/Bz;
							hW.Vz[Gindex] = DxiVz/Bx + DetVz/By + DztVz/Bz;

#ifdef DisBug
if(zbx == idx+(ipam[2]-LenFD)+ipam[9] && zby == idy+(ipam[4]-LenFD) && zbz == idz)
{
	printf("\tat PCS[%d]DEV[%d](%d,%d,%d):\n\t\tCalWave->hW.Txx=%e, DxiTxx=%e, DetTxx=%e, DztTxx=%e\n"
	       "\t\thW.Vy=%e, DxiVy=%e, DetVy=%e, DztVy=%e\n"
	       "\t\thW.Vx=%e, DxiVx=%e, DetVx=%e, DztVx=%e\n",
		ipam[2],ipam[1], zbx,zby,zbz, 
		hW.Txx[Gindex], DxiTxx,DetTxx,DztTxx, 
		hW.Vy[Gindex],DxiVy,DetVy,DztVy,
		hW.Vx[Gindex],DxiVx,DetVx,DztVx);
	printf("\tDxVx=%e, DxVy=%e, DxVz=%e\n"
	       "\tDzTxy=%e,DzTyy=%e,DzTyz=%e\n"
	       "\tDzTxx=%e,DzTxy=%e,DzTxz=%e\n",
		pd.DxVx[Gindex], pd.DxVy[Gindex], pd.DxVz[Gindex], 
		pd.DzTxy[Hyindex],pd.DzTyy[Hyindex],pd.DzTyz[Gindex],
		pd.DzTxx[Hyindex],pd.DzTxy[Hyindex],pd.DzTxz[Gindex]);
	printf("xix=%e, %e, %e, etx=%e, %e, %e, ztx=%e, %e, %e\n",xix,xiy,xiz,etx,ety,etz,ztx,zty,ztz);
	printf(" calwave-->(%d,%d,%d) Txx=%e, Tyy=%e, Tzz=%e\n\tTxy=%e, Txz=%e,Tzz=%e\n\tVx=%e Vy=%e Vz=%e\n",
		zbx,zby,idz,hW.Txx[Gindex],hW.Tyy[Gindex],hW.Tzz[Gindex],hW.Txy[Gindex],hW.Txz[Gindex],hW.Tyz[Gindex],hW.Vx[Gindex],hW.Vy[Gindex],hW.Vz[Gindex]);
}
#endif

#ifdef CFSPML
#ifdef CondFree
							//top surface partial derivative conversion
							xiaoI=idx*(ipam[5]-ipam[4]+1+2*LenFD)*SeisGeo*SeisGeo + idy*SeisGeo*SeisGeo;//valid Y
							
							if(idz == ipam[8]+LenFD-1)//surface layer
							{
						DzVx1 = CoVx[xiaoI+0]*pd.DxVx[Gindex] + CoVx[xiaoI+1]*pd.DxVy[Gindex] + CoVx[xiaoI+2]*pd.DxVz[Gindex];
						DzVx2 = CoVy[xiaoI+0]*pd.DyVx[Gindex] + CoVy[xiaoI+1]*pd.DyVy[Gindex] + CoVy[xiaoI+2]*pd.DyVz[Gindex];
						DzVy1 = CoVx[xiaoI+3]*pd.DxVx[Gindex] + CoVx[xiaoI+4]*pd.DxVy[Gindex] + CoVx[xiaoI+5]*pd.DxVz[Gindex];
						DzVy2 = CoVy[xiaoI+3]*pd.DyVx[Gindex] + CoVy[xiaoI+4]*pd.DyVy[Gindex] + CoVy[xiaoI+5]*pd.DyVz[Gindex];
						DzVz1 = CoVx[xiaoI+6]*pd.DxVx[Gindex] + CoVx[xiaoI+7]*pd.DxVy[Gindex] + CoVx[xiaoI+8]*pd.DxVz[Gindex];
						DzVz2 = CoVy[xiaoI+6]*pd.DyVx[Gindex] + CoVy[xiaoI+7]*pd.DyVy[Gindex] + CoVy[xiaoI+8]*pd.DyVz[Gindex];
							}
#endif							
							tempIdx = idx+(ipam[2]-LenFD)+ipam[9];//idx+ipam[9]
							if(tempIdx<=apr.nabs[0]+LenFD-1 || tempIdx>=ipam[10]+LenFD-apr.nabs[1])//X-dir
							{
						tempIdx<apr.nabs[0]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[10]+LenFD-apr.nabs[1])+apr.nabs[0];
						Pidx = Pidx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
						
						//wavefield attenuation (Equation 14 of Zhang 2010)
						hW.Vx[Gindex]  -= Ax.Vx[Pidx]/Bx;  
						hW.Vy[Gindex]  -= Ax.Vy[Pidx]/Bx;  
						hW.Vz[Gindex]  -= Ax.Vz[Pidx]/Bx;
						hW.Txx[Gindex] -= Ax.Txx[Pidx]/Bx;
						hW.Tyy[Gindex] -= Ax.Tyy[Pidx]/Bx;
						hW.Tzz[Gindex] -= Ax.Tzz[Pidx]/Bx;
						hW.Txy[Gindex] -= Ax.Txy[Pidx]/Bx;
						hW.Txz[Gindex] -= Ax.Txz[Pidx]/Bx;
						hW.Tyz[Gindex] -= Ax.Tyz[Pidx]/Bx;
						
						//ADE update (Equation A10 of Zhang 2010)
						hAx.Vx[Pidx]  = DxiVx*DBx - APDx*Ax.Vx[Pidx];
						hAx.Vy[Pidx]  = DxiVy*DBx - APDx*Ax.Vy[Pidx];
						hAx.Vz[Pidx]  = DxiVz*DBx - APDx*Ax.Vz[Pidx];
						hAx.Txx[Pidx] = DxiTxx*DBx - APDx*Ax.Txx[Pidx];
						hAx.Tyy[Pidx] = DxiTyy*DBx - APDx*Ax.Tyy[Pidx];
						hAx.Tzz[Pidx] = DxiTzz*DBx - APDx*Ax.Tzz[Pidx];
						hAx.Txy[Pidx] = DxiTxy*DBx - APDx*Ax.Txy[Pidx];
						hAx.Txz[Pidx] = DxiTxz*DBx - APDx*Ax.Txz[Pidx];
						hAx.Tyz[Pidx] = DxiTyz*DBx - APDx*Ax.Tyz[Pidx];

#ifdef CondFree
						//top surface 
								if(idz == ipam[8]+LenFD-1)
								{
							hAx.Txx[Pidx] += DBx*Bx*( lam2mu*ztx*DzVx1 + lambda*zty*DzVy1 + lambda*ztz*DzVz1);
							hAx.Tyy[Pidx] += DBx*Bx*( lambda*ztx*DzVx1 + lam2mu*zty*DzVy1 + lambda*ztz*DzVz1);
							hAx.Tzz[Pidx] += DBx*Bx*( lambda*ztx*DzVx1 + lambda*zty*DzVy1 + lam2mu*ztz*DzVz1);
							hAx.Txy[Pidx] += DBx*Bx*( zty*DzVx1 + ztx*DzVy1 )*miu; 
							hAx.Txz[Pidx] += DBx*Bx*( ztz*DzVx1 + ztx*DzVz1 )*miu; 
							hAx.Tyz[Pidx] += DBx*Bx*( ztz*DzVy1 + zty*DzVz1 )*miu; 
								}
#endif						

							}
							
							tempIdx = idy + (ipam[4]-LenFD);//idy
							if(tempIdx<=apr.nabs[2]+LenFD-1 || tempIdx>=ipam[7]+LenFD-apr.nabs[3])//Y-dir
							{
						tempIdx<apr.nabs[2]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[7]+LenFD-apr.nabs[3])+apr.nabs[2];		
						Pidx = Pidx*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[8]+2*LenFD) + idx*(ipam[8]+2*LenFD) + idz;		
						
						//wavefield attenuation (Equation 14 of Zhang 2010)
						hW.Vx[Gindex]  -= Ay.Vx[Pidx]/By;  
						hW.Vy[Gindex]  -= Ay.Vy[Pidx]/By;  
						hW.Vz[Gindex]  -= Ay.Vz[Pidx]/By;
						hW.Txx[Gindex] -= Ay.Txx[Pidx]/By;
						hW.Tyy[Gindex] -= Ay.Tyy[Pidx]/By;
						hW.Tzz[Gindex] -= Ay.Tzz[Pidx]/By;
						hW.Txy[Gindex] -= Ay.Txy[Pidx]/By;
						hW.Txz[Gindex] -= Ay.Txz[Pidx]/By;
						hW.Tyz[Gindex] -= Ay.Tyz[Pidx]/By;
						
						//ADE update (Equation A10 of Zhang 2010)
						hAy.Vx[Pidx]  = DetVx*DBy - APDy*Ay.Vx[Pidx];
						hAy.Vy[Pidx]  = DetVy*DBy - APDy*Ay.Vy[Pidx];
						hAy.Vz[Pidx]  = DetVz*DBy - APDy*Ay.Vz[Pidx];
						hAy.Txx[Pidx] = DetTxx*DBy - APDy*Ay.Txx[Pidx];
						hAy.Tyy[Pidx] = DetTyy*DBy - APDy*Ay.Tyy[Pidx];
						hAy.Tzz[Pidx] = DetTzz*DBy - APDy*Ay.Tzz[Pidx];
						hAy.Txy[Pidx] = DetTxy*DBy - APDy*Ay.Txy[Pidx];
						hAy.Txz[Pidx] = DetTxz*DBy - APDy*Ay.Txz[Pidx];
						hAy.Tyz[Pidx] = DetTyz*DBy - APDy*Ay.Tyz[Pidx];

#ifdef CondFree
						//top surface 
								if(idz == ipam[8]+LenFD-1)
								{
							hAy.Txx[Pidx] += DBy*By*( lam2mu*ztx*DzVx2 + lambda*zty*DzVy2 + lambda*ztz*DzVz2);
							hAy.Tyy[Pidx] += DBy*By*( lambda*ztx*DzVx2 + lam2mu*zty*DzVy2 + lambda*ztz*DzVz2);
							hAy.Tzz[Pidx] += DBy*By*( lambda*ztx*DzVx2 + lambda*zty*DzVy2 + lam2mu*ztz*DzVz2);
							hAy.Txy[Pidx] += DBy*By*( zty*DzVx2 + ztx*DzVy2 )*miu; 
							hAy.Txz[Pidx] += DBy*By*( ztz*DzVx2 + ztx*DzVz2 )*miu; 
							hAy.Tyz[Pidx] += DBy*By*( ztz*DzVy2 + zty*DzVz2 )*miu; 
								}
#endif						
							
							}


							if(idz<=apr.nabs[4]+LenFD-1 || idz>=ipam[8]+LenFD-apr.nabs[5])//Z-dir
							{
						idz<apr.nabs[4]+LenFD ? Pidx=idz-LenFD : Pidx=idz-(ipam[8]+LenFD-apr.nabs[5])+apr.nabs[4];		
						Pidx = Pidx*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[5]-ipam[4]+1+2*LenFD) + idx*(ipam[5]-ipam[4]+1+2*LenFD) + idy;		
							
						//wavefield attenuation (Equation 14 of Zhang 2010)
						hW.Vx[Gindex]  -= Az.Vx[Pidx]/Bz;  
						hW.Vy[Gindex]  -= Az.Vy[Pidx]/Bz;  
						hW.Vz[Gindex]  -= Az.Vz[Pidx]/Bz;
						hW.Txx[Gindex] -= Az.Txx[Pidx]/Bz;
						hW.Tyy[Gindex] -= Az.Tyy[Pidx]/Bz;
						hW.Tzz[Gindex] -= Az.Tzz[Pidx]/Bz;
						hW.Txy[Gindex] -= Az.Txy[Pidx]/Bz;
						hW.Txz[Gindex] -= Az.Txz[Pidx]/Bz;
						hW.Tyz[Gindex] -= Az.Tyz[Pidx]/Bz;
						
						//ADE update (Equation A10 of Zhang 2010)
						hAz.Vx[Pidx]  = DztVx*DBz - APDz*Az.Vx[Pidx];
						hAz.Vy[Pidx]  = DztVy*DBz - APDz*Az.Vy[Pidx];
						hAz.Vz[Pidx]  = DztVz*DBz - APDz*Az.Vz[Pidx];
						hAz.Txx[Pidx] = DztTxx*DBz - APDz*Az.Txx[Pidx];
						hAz.Tyy[Pidx] = DztTyy*DBz - APDz*Az.Tyy[Pidx];
						hAz.Tzz[Pidx] = DztTzz*DBz - APDz*Az.Tzz[Pidx];
						hAz.Txy[Pidx] = DztTxy*DBz - APDz*Az.Txy[Pidx];
						hAz.Txz[Pidx] = DztTxz*DBz - APDz*Az.Txz[Pidx];
						hAz.Tyz[Pidx] = DztTyz*DBz - APDz*Az.Tyz[Pidx];
							
							}

#endif

						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

__global__ void CalDiffSL(int Xvec, int Yvec, int Zvec, int ConIndex, Real steph, Real *CoVx, Real* CoVy, wfield W, PartialD pd)
{
	//int i,j,k;
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int Zclamp;

	Real xstep, ystep, zstep;
	int xinc, yinc, zinc;
	
	xstep = steph*Xvec;
	ystep = steph*Yvec;
	zstep = steph*Zvec;
	xinc = Xvec*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD);//skip cdx.ny*cdx.nz
	yinc = Yvec*(ipam[8]+2*LenFD);//skip cdx.nz
	zinc = Zvec*1;//skip 1

	ConIndex > ipam[8]+LenFD ? Zclamp = ipam[8]+LenFD : Zclamp = ConIndex;

	//generally use DRP/opt MacCormack scheme to get derivative, as Equation 2.23 and coefficients is Equation 2.24 in Thesis.
	//for the top layer transfrom the derivative of xi and eta to get zeta direction derivative, as Equation 3.4 in Thesis.

	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current device compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<Zclamp)//vaild point with one virtual bounds
						{

							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;

							pd.DxTxx[Gindex] = DRPFD( W.Txx, Gindex, xstep, xinc);//311
							pd.DxTxy[Gindex] = DRPFD( W.Txy, Gindex, xstep, xinc);//312
							pd.DxTxz[Gindex] = DRPFD( W.Txz, Gindex, xstep, xinc);//313
							pd.DxVx[Gindex] = DRPFD( W.Vx, Gindex, xstep, xinc);//31-456
							pd.DxVy[Gindex] = DRPFD( W.Vy, Gindex, xstep, xinc);//317
							pd.DxVz[Gindex] = DRPFD( W.Vz, Gindex, xstep, xinc);//318

							pd.DyTyy[Gindex] = DRPFD( W.Tyy, Gindex, ystep, yinc);//312
							pd.DyTxy[Gindex] = DRPFD( W.Txy, Gindex, ystep, yinc);//311
							pd.DyTyz[Gindex] = DRPFD( W.Tyz, Gindex, ystep, yinc);//313
							pd.DyVx[Gindex] = DRPFD( W.Vx, Gindex, ystep, yinc);//317
							pd.DyVy[Gindex] = DRPFD( W.Vy, Gindex, ystep, yinc);//31-456
							pd.DyVz[Gindex] = DRPFD( W.Vz, Gindex, ystep, yinc);//319

							//pd.DzTxx[Gindex] = DRPFD( W.Txx, Gindex, zstep, zinc);//esp
							//pd.DzTyy[Gindex] = DRPFD( W.Tyy, Gindex, zstep, zinc);//esp
							pd.DzTzz[Gindex] = DRPFD( W.Tzz, Gindex, zstep, zinc);//313
							//pd.DzTxy[Gindex] = DRPFD( W.Txy, Gindex, zstep, zinc);//esp
							pd.DzTxz[Gindex] = DRPFD( W.Txz, Gindex, zstep, zinc);//311
							pd.DzTyz[Gindex] = DRPFD( W.Tyz, Gindex, zstep, zinc);//312
							pd.DzVx[Gindex] = DRPFD( W.Vx, Gindex, zstep, zinc);//318
							pd.DzVy[Gindex] = DRPFD( W.Vy, Gindex, zstep, zinc);//319
							pd.DzVz[Gindex] = DRPFD( W.Vz, Gindex, zstep, zinc);//31-456

						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}


__global__ void CalWaveSL(int ConIndex, derivF drv, mdparF mpa, apara apr, PartialD pd, wfield hW,
			  wfield Ax, wfield hAx, wfield Ay, wfield hAy, wfield Az, wfield hAz)
{
	//int i,j,k;
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int Zclamp;

	Real lambda,miu,rho,lam2mu;
	Real xix,ety,ztz;
	//Real ztx,zty;
	
	Real DxiVx,DetVx,DztVx, DxiVy,DetVy,DztVy, DxiVz,DetVz,DztVz;
	Real DxiTxx,DetTxx,DztTxx, DxiTyy,DetTyy,DztTyy, DxiTzz,DetTzz,DztTzz;
	Real DxiTxy,DetTxy,DztTxy, DxiTxz,DetTxz,DztTxz, DxiTyz,DetTyz,DztTyz;//T6V3
	Real Bx,By,Bz;//absorb boundary pars

#ifdef CFSPML
	Real APDx,APDy,APDz, DBx,DBy,DBz;
	int Pidx,tempIdx;
#endif

	ConIndex > ipam[8]+LenFD ? Zclamp = ipam[8]+LenFD : Zclamp = ConIndex;

	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current device compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<Zclamp)//ecept convers interface
						{
							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
							
							rho = mpa.rho[Gindex];
							miu = rho*mpa.beta[Gindex]*mpa.beta[Gindex];
							lam2mu = rho*mpa.alpha[Gindex]*mpa.alpha[Gindex];
							lambda = lam2mu - 2.0*miu;
							rho = 1.0/rho;
							
							xix = drv.xix[Gindex];
							ety = drv.etay[Gindex];
							//ztx = drv.zetax[Gindex];
							//zty = drv.zetay[Gindex];
							ztz = drv.zetaz[Gindex];
							//for hy1, for hy2 should add ztx and zty
							//xiy,xiz,etax,etaz,should be 0 for straight line
							//and xix,etay,zetax,zetay,zetaz has NO rotation property, only scaling property
							//especailly in topo area Z varies in three direction, so zetax,zetay,zetaz all have scaling property

							//hVx
							DxiVx = rho*xix*pd.DxTxx[Gindex];
							DetVx = rho*ety*pd.DyTxy[Gindex];
							DztVx = rho*ztz*pd.DzTxz[Gindex];
							//DztVx = rho*(ztx*pd.DzTxx[Gindex] + zty*pd.DzTxy[Gindex] + ztz*pd.DzTxz[Gindex]);

							//hVy
							DxiVy = rho*xix*pd.DxTxy[Gindex];
							DetVy = rho*ety*pd.DyTyy[Gindex];
							DztVy = rho*ztz*pd.DzTyz[Gindex];
							//DztVy = rho*(ztx*pd.DzTxy[Gindex] + zty*pd.DzTyy[Gindex] + ztz*pd.DzTyz[Gindex]);

							//hVz
							DxiVz = rho*xix*pd.DxTxz[Gindex];
							DetVz = rho*ety*pd.DyTyz[Gindex];
							DztVz = rho*ztz*pd.DzTzz[Gindex];
							//DztVz = rho*(ztx*pd.DzTxz[Gindex] + zty*pd.DzTyz[Gindex] + ztz*pd.DzTzz[Gindex]);

							//hTxx
							DxiTxx = lam2mu*xix*pd.DxVx[Gindex];
							DetTxx = lambda*ety*pd.DyVy[Gindex];
							DztTxx = lambda*ztz*pd.DzVz[Gindex];
							//DztTxx = lam2mu*ztx*pd.DzVx[Gindex] + lambda*zty*pd.DzVy[Gindex] + lambda*ztz*pd.DzVz[Gindex];

							//hTyy
							DxiTyy = lambda*xix*pd.DxVx[Gindex];
							DetTyy = lam2mu*ety*pd.DyVy[Gindex];
							DztTyy = lambda*ztz*pd.DzVz[Gindex];
							//DztTyy = lambda*ztx*pd.DzVx[Gindex] + lam2mu*zty*pd.DzVy[Gindex] + lambda*ztz*pd.DzVz[Gindex];

							//hTzz
							DxiTzz = lambda*xix*pd.DxVx[Gindex];
							DetTzz = lambda*ety*pd.DyVy[Gindex];
							DztTzz = lam2mu*ztz*pd.DzVz[Gindex];
							//DztTzz = lambda*ztx*pd.DzVx[Gindex] + lambda*zty*pd.DzVy[Gindex] + lam2mu*ztz*pd.DzVz[Gindex];

							//hTxy
							DxiTxy = miu*xix*pd.DxVy[Gindex];
							DetTxy = miu*ety*pd.DyVx[Gindex];
							DztTxy = 0.0;
							//DztTxy = miu*(zty*pd.DzVx[Gindex] + ztx*pd.DzVy[Gindex]);

							//hTxz
							DxiTxz = miu*xix*pd.DxVz[Gindex];
							DetTxz = 0.0;
							DztTxz = miu*ztz*pd.DzVx[Gindex];
							//DztTxz = miu*(ztz*pd.DzVx[Gindex] + ztx*pd.DzVz[Gindex]);

							//hTyz
							DxiTyz = 0.0;
							DetTyz = miu*ety*pd.DyVz[Gindex];
							DztTyz = miu*ztz*pd.DzVy[Gindex];
							//DztTyz = miu*(ztz*pd.DzVy[Gindex] + zty*pd.DzVz[Gindex]);
	
#ifdef CFSPML
	APDx = apr.APDx[idx];	APDy = apr.APDy[idy];	APDz = apr.APDz[idz];
	Bx = apr.Bx[idx];	By = apr.By[idy];	Bz = apr.Bz[idz];
	DBx = apr.DBx[idx];	DBy = apr.DBy[idy];	DBz = apr.DBz[idz];
#else	
	Bx = 1.0;	By = 1.0;	Bz = 1.0;
#endif

							//time domain partial derivative--->wave field
							hW.Txx[Gindex] = DxiTxx/Bx + DetTxx/By + DztTxx/Bz;
							hW.Tyy[Gindex] = DxiTyy/Bx + DetTyy/By + DztTyy/Bz;
							hW.Tzz[Gindex] = DxiTzz/Bx + DetTzz/By + DztTzz/Bz;
							hW.Txy[Gindex] = DxiTxy/Bx + DetTxy/By + DztTxy/Bz;
							hW.Txz[Gindex] = DxiTxz/Bx + DetTxz/By + DztTxz/Bz;
							hW.Tyz[Gindex] = DxiTyz/Bx + DetTyz/By + DztTyz/Bz;
							hW.Vx[Gindex] = DxiVx/Bx + DetVx/By + DztVx/Bz;
							hW.Vy[Gindex] = DxiVy/Bx + DetVy/By + DztVy/Bz;
							hW.Vz[Gindex] = DxiVz/Bx + DetVz/By + DztVz/Bz;
							
#ifdef CFSPML
							tempIdx = idx + (ipam[2]-LenFD) + ipam[9];//idx+ipam[9]
							if(tempIdx<=apr.nabs[0]+LenFD-1 || tempIdx>=ipam[10]+LenFD-apr.nabs[1])//X-dir
							{
						tempIdx<apr.nabs[0]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[10]+LenFD-apr.nabs[1])+apr.nabs[0];
						Pidx = Pidx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
						
						//wavefield attenuation (Equation 14 of Zhang 2010)
						hW.Vx[Gindex]  -= Ax.Vx[Pidx]/Bx;  
						hW.Vy[Gindex]  -= Ax.Vy[Pidx]/Bx;  
						hW.Vz[Gindex]  -= Ax.Vz[Pidx]/Bx;
						hW.Txx[Gindex] -= Ax.Txx[Pidx]/Bx;
						hW.Tyy[Gindex] -= Ax.Tyy[Pidx]/Bx;
						hW.Tzz[Gindex] -= Ax.Tzz[Pidx]/Bx;
						hW.Txy[Gindex] -= Ax.Txy[Pidx]/Bx;
						hW.Txz[Gindex] -= Ax.Txz[Pidx]/Bx;
						hW.Tyz[Gindex] -= Ax.Tyz[Pidx]/Bx;
						
						//ADE update (Equation A10 of Zhang 2010)
						hAx.Vx[Pidx]  = DxiVx*DBx - APDx*Ax.Vx[Pidx];
						hAx.Vy[Pidx]  = DxiVy*DBx - APDx*Ax.Vy[Pidx];
						hAx.Vz[Pidx]  = DxiVz*DBx - APDx*Ax.Vz[Pidx];
						hAx.Txx[Pidx] = DxiTxx*DBx - APDx*Ax.Txx[Pidx];
						hAx.Tyy[Pidx] = DxiTyy*DBx - APDx*Ax.Tyy[Pidx];
						hAx.Tzz[Pidx] = DxiTzz*DBx - APDx*Ax.Tzz[Pidx];
						hAx.Txy[Pidx] = DxiTxy*DBx - APDx*Ax.Txy[Pidx];
						hAx.Txz[Pidx] = DxiTxz*DBx - APDx*Ax.Txz[Pidx];
						hAx.Tyz[Pidx] = DxiTyz*DBx - APDx*Ax.Tyz[Pidx];

							}
							
							tempIdx = idy + (ipam[4]-LenFD);//idy
							if(tempIdx<=apr.nabs[2]+LenFD-1 || tempIdx>=ipam[7]+LenFD-apr.nabs[3])//Y-dir
							{
						tempIdx<apr.nabs[2]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[7]+LenFD-apr.nabs[3])+apr.nabs[2];		
						Pidx = Pidx*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[8]+2*LenFD) + idx*(ipam[8]+2*LenFD) + idz;		
						
						//wavefield attenuation (Equation 14 of Zhang 2010)
						hW.Vx[Gindex]  -= Ay.Vx[Pidx]/By;  
						hW.Vy[Gindex]  -= Ay.Vy[Pidx]/By;  
						hW.Vz[Gindex]  -= Ay.Vz[Pidx]/By;
						hW.Txx[Gindex] -= Ay.Txx[Pidx]/By;
						hW.Tyy[Gindex] -= Ay.Tyy[Pidx]/By;
						hW.Tzz[Gindex] -= Ay.Tzz[Pidx]/By;
						hW.Txy[Gindex] -= Ay.Txy[Pidx]/By;
						hW.Txz[Gindex] -= Ay.Txz[Pidx]/By;
						hW.Tyz[Gindex] -= Ay.Tyz[Pidx]/By;
						
						//ADE update (Equation A10 of Zhang 2010)
						hAy.Vx[Pidx]  = DetVx*DBy - APDy*Ay.Vx[Pidx];
						hAy.Vy[Pidx]  = DetVy*DBy - APDy*Ay.Vy[Pidx];
						hAy.Vz[Pidx]  = DetVz*DBy - APDy*Ay.Vz[Pidx];
						hAy.Txx[Pidx] = DetTxx*DBy - APDy*Ay.Txx[Pidx];
						hAy.Tyy[Pidx] = DetTyy*DBy - APDy*Ay.Tyy[Pidx];
						hAy.Tzz[Pidx] = DetTzz*DBy - APDy*Ay.Tzz[Pidx];
						hAy.Txy[Pidx] = DetTxy*DBy - APDy*Ay.Txy[Pidx];
						hAy.Txz[Pidx] = DetTxz*DBy - APDy*Ay.Txz[Pidx];
						hAy.Tyz[Pidx] = DetTyz*DBy - APDy*Ay.Tyz[Pidx];
							
							}


							if(idz<=apr.nabs[4]+LenFD-1 || idz>=ipam[8]+LenFD-apr.nabs[5])//Z1
							{
						idz<apr.nabs[4]+LenFD ? Pidx=idz-LenFD : Pidx=idz-(ipam[8]+LenFD-apr.nabs[5])+apr.nabs[4];		
						Pidx = Pidx*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[5]-ipam[4]+1+2*LenFD) + idx*(ipam[5]-ipam[4]+1+2*LenFD) + idy;		
							
						//wavefield attenuation (Equation 14 of Zhang 2010)
						hW.Vx[Gindex]  -= Az.Vx[Pidx]/Bz;  
						hW.Vy[Gindex]  -= Az.Vy[Pidx]/Bz;  
						hW.Vz[Gindex]  -= Az.Vz[Pidx]/Bz;
						hW.Txx[Gindex] -= Az.Txx[Pidx]/Bz;
						hW.Tyy[Gindex] -= Az.Tyy[Pidx]/Bz;
						hW.Tzz[Gindex] -= Az.Tzz[Pidx]/Bz;
						hW.Txy[Gindex] -= Az.Txy[Pidx]/Bz;
						hW.Txz[Gindex] -= Az.Txz[Pidx]/Bz;
						hW.Tyz[Gindex] -= Az.Tyz[Pidx]/Bz;
						
						//ADE update (Equation A10 of Zhang 2010)
						hAz.Vx[Pidx]  = DztVx*DBz - APDz*Az.Vx[Pidx];
						hAz.Vy[Pidx]  = DztVy*DBz - APDz*Az.Vy[Pidx];
						hAz.Vz[Pidx]  = DztVz*DBz - APDz*Az.Vz[Pidx];
						hAz.Txx[Pidx] = DztTxx*DBz - APDz*Az.Txx[Pidx];
						hAz.Tyy[Pidx] = DztTyy*DBz - APDz*Az.Tyy[Pidx];
						hAz.Tzz[Pidx] = DztTzz*DBz - APDz*Az.Tzz[Pidx];
						hAz.Txy[Pidx] = DztTxy*DBz - APDz*Az.Txy[Pidx];
						hAz.Txz[Pidx] = DztTxz*DBz - APDz*Az.Txz[Pidx];
						hAz.Tyz[Pidx] = DztTyz*DBz - APDz*Az.Tyz[Pidx];
							
							}

#endif
						
						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

__global__ void CalTIMG(int Xvec, int Yvec, int Zvec, Real steph, Real *rho, derivF drv, wfield W, wfield hW,
			wfield Ax, wfield hAx, wfield Ay, wfield hAy, apara apr)
{
	//  <<<BPG.y, BPG.x>>>
	//gridDim.x<=cdx.nx  blockDim.x<=cdx.ny
	int i,j;
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int Relidx;//relative index

	Real VecTx[LenFD*2+1], VecTy[LenFD*2+1], VecTz[LenFD*2+1];
	Real DxTx,DyTy,DzTz;
	Real Bx,By;//absorb boundary pars
	Real rhojac;
	Real xstep, ystep, zstep;
	int xinc, yinc;
	Real T3Src=0.0;//initial value

#ifdef CFSPML
	Real APDx,DBx, APDy,DBy;
	int Pidx,tempIdx;
#endif
	
	xstep = steph*Xvec;
	ystep = steph*Yvec;
	zstep = steph*Zvec;
	//none direction
	xinc = (ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD);//skip cdx.ny*cdx.nz
	yinc = ipam[8]+2*LenFD;//skip cdx.nz

	//the Traction Image method for free surface condition, use the conservative form momentum equation as Equation 3.11 in Thesis
	
	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.x)//loop in current compute range with step of Launch Par
	{
		idx = countX + blockIdx.x + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=blockDim.x)
			{
				idy = countY + threadIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{  
					for(countZ=0; countZ<LenFD; countZ++)
					{
						idz = ipam[8] + countZ;//valid point number + LenFD = last location of valid point

						Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;

						rhojac = 1.0/rho[Gindex]/drv.jac[Gindex];

#ifdef CFSPML
	APDx = apr.APDx[idx];	APDy = apr.APDy[idy];
	Bx = apr.Bx[idx];	By = apr.By[idy];
	DBx = apr.DBx[idx];	DBy = apr.DBy[idy];
#else	
	Bx = 1.0;	By = 1.0;
#endif

						//X-dir
						//vector of covariant pars multiplied directional stress
						for(i=0;i<LenFD*2+1;i++)
						{
							Relidx = Gindex - LenFD*xinc;//X-dir
							VecTx[i] = drv.jac[Relidx+i*xinc]*(drv.xix[Relidx+i*xinc]*W.Txx[Relidx+i*xinc]+
											   drv.xiy[Relidx+i*xinc]*W.Txy[Relidx+i*xinc]+
											   drv.xiz[Relidx+i*xinc]*W.Txz[Relidx+i*xinc]);

							Relidx = Gindex - LenFD*yinc;//Y-dir
							VecTy[i] = drv.jac[Relidx+i*yinc]*(drv.etax[Relidx+i*yinc]*W.Txx[Relidx+i*yinc]+
											   drv.etay[Relidx+i*yinc]*W.Txy[Relidx+i*yinc]+
											   drv.etaz[Relidx+i*yinc]*W.Txz[Relidx+i*yinc]);

							Relidx = Gindex - LenFD;//Z-dir
							VecTz[i] = drv.jac[Relidx+i]*(drv.zetax[Relidx+i]*W.Txx[Relidx+i]+
										      drv.zetay[Relidx+i]*W.Txy[Relidx+i]+
										      drv.zetaz[Relidx+i]*W.Txz[Relidx+i]);
						}

						//traction image
						for(j=1;j<=LenFD-(2-countZ);j++)
							VecTz[LenFD + (2-countZ) + j] = 2.0*T3Src - VecTz[LenFD + (2-countZ) - j];//TxSrc
						VecTz[LenFD + (2-countZ)] = T3Src;

						//partial derivative vector
						//the data has been already extracted from orignal array and put into new array point by point,
						//so it doesn't need big step skip when do differential work, and only need direction information.
						//the differential center is vector center.
						DxTx = rhojac*strF( VecTx, LenFD, xstep, Xvec);
						DyTy = rhojac*strF( VecTy, LenFD, ystep, Yvec);
						DzTz = rhojac*strF( VecTz, LenFD, zstep, Zvec);

						//time domain partial derivative--->wave field
						hW.Vx[Gindex] = DxTx/Bx + DyTy/By + DzTz;

#ifdef CFSPML
				//if apply free surface condition, in top area of Z-dir will not apply absorbtion
				//so nabs[5] equals 0, and related ADE wavefield(Az,hAz) will be none
				//absorbing pars will be default APDz=0, DBz=0, Bz=1
				//so the Z-top absorption is eliminated
						//X dir absorption
						tempIdx = idx + (ipam[2]-LenFD) + ipam[9];
						if(tempIdx<=apr.nabs[0]+LenFD-1 || tempIdx>=ipam[10]+LenFD-apr.nabs[1])
						{
						tempIdx<apr.nabs[0]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[10]+LenFD-apr.nabs[1])+apr.nabs[0];
						Pidx = Pidx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
						
						hW.Vx[Gindex] -= Ax.Vx[Pidx]/Bx;
						hAx.Vx[Pidx] = DBx*DxTx - APDx*Ax.Vx[Pidx];
						}


						//Y dir absorption
						tempIdx = idy + (ipam[4]-LenFD);
						if(tempIdx<=apr.nabs[2]+LenFD-1 || tempIdx>=ipam[7]+LenFD-apr.nabs[3])
						{
						tempIdx<apr.nabs[2]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[7]+LenFD-apr.nabs[3])+apr.nabs[2];
						Pidx = Pidx*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[8]+2*LenFD) + idx*(ipam[8]+2*LenFD) + idz;

						hW.Vx[Gindex] -= Ay.Vx[Pidx]/By;
						hAy.Vx[Pidx] = DBy*DyTy - APDy*Ay.Vx[Pidx];
						}


#endif
						
						//Y-dir
						for(i=0;i<LenFD*2+1;i++)
						{
							Relidx = Gindex - LenFD*xinc;
							VecTx[i] = drv.jac[Relidx+i*xinc]*(drv.xix[Relidx+i*xinc]*W.Txy[Relidx+i*xinc]+
											   drv.xiy[Relidx+i*xinc]*W.Tyy[Relidx+i*xinc]+
											   drv.xiz[Relidx+i*xinc]*W.Tyz[Relidx+i*xinc]);
							
							Relidx = Gindex - LenFD*yinc;
							VecTy[i] = drv.jac[Relidx+i*yinc]*(drv.etax[Relidx+i*yinc]*W.Txy[Relidx+i*yinc]+
											   drv.etay[Relidx+i*yinc]*W.Tyy[Relidx+i*yinc]+
											   drv.etaz[Relidx+i*yinc]*W.Tyz[Relidx+i*yinc]);
							
							Relidx = Gindex - LenFD;
							VecTz[i] = drv.jac[Relidx+i]*(drv.zetax[Relidx+i]*W.Txy[Relidx+i]+
										      drv.zetay[Relidx+i]*W.Tyy[Relidx+i]+
										      drv.zetaz[Relidx+i]*W.Tyz[Relidx+i]);
						}

						for(j=1;j<=LenFD-(2-countZ);j++)
							VecTz[LenFD + (2-countZ) + j] = 2.0*T3Src - VecTz[LenFD + (2-countZ) - j];//TySrc
						VecTz[LenFD + (2-countZ)] = T3Src;

						DxTx = rhojac*strF( VecTx, LenFD, xstep, Xvec);//LenFD means center
						DyTy = rhojac*strF( VecTy, LenFD, ystep, Yvec);
						DzTz = rhojac*strF( VecTz, LenFD, zstep, Zvec);

						hW.Vy[Gindex] = DxTx/Bx + DyTy/By + DzTz;
					
#ifdef CFSPML
						//X dir absorption
						tempIdx = idx + (ipam[2]-LenFD) + ipam[9];
						if(tempIdx<=apr.nabs[0]+LenFD-1 || tempIdx>=ipam[10]+LenFD-apr.nabs[1])
						{
						tempIdx<apr.nabs[0]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[10]+LenFD-apr.nabs[1])+apr.nabs[0];
						Pidx = Pidx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
						
						hW.Vy[Gindex] -= Ax.Vy[Pidx]/Bx;
						hAx.Vy[Pidx] = DBx*DxTx - APDx*Ax.Vy[Pidx];
						}
						//Y dir absorption
						tempIdx = idy + (ipam[4]-LenFD);
						if(tempIdx<=apr.nabs[2]+LenFD-1 || tempIdx>=ipam[7]+LenFD-apr.nabs[3])
						{
						tempIdx<apr.nabs[2]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[7]+LenFD-apr.nabs[3])+apr.nabs[2];
						Pidx = Pidx*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[8]+2*LenFD) + idx*(ipam[8]+2*LenFD) + idz;

						hW.Vy[Gindex] -= Ay.Vy[Pidx]/By;
						hAy.Vy[Pidx] = DBy*DyTy - APDy*Ay.Vy[Pidx];
						}

#endif
						//Z-dir
						for(i=0;i<LenFD*2+1;i++)
						{
							Relidx = Gindex - LenFD*xinc;
							VecTx[i] = drv.jac[Relidx+i*xinc]*(drv.xix[Relidx+i*xinc]*W.Txz[Relidx+i*xinc]+
											   drv.xiy[Relidx+i*xinc]*W.Tyz[Relidx+i*xinc]+
											   drv.xiz[Relidx+i*xinc]*W.Tzz[Relidx+i*xinc]);
							
							Relidx = Gindex - LenFD*yinc;
							VecTy[i] = drv.jac[Relidx+i*yinc]*(drv.etax[Relidx+i*yinc]*W.Txz[Relidx+i*yinc]+
											   drv.etay[Relidx+i*yinc]*W.Tyz[Relidx+i*yinc]+
											   drv.etaz[Relidx+i*yinc]*W.Tzz[Relidx+i*yinc]);
							
							Relidx = Gindex - LenFD;
							VecTz[i] = drv.jac[Relidx+i]*(drv.zetax[Relidx+i]*W.Txz[Relidx+i]+
										      drv.zetay[Relidx+i]*W.Tyz[Relidx+i]+
										      drv.zetaz[Relidx+i]*W.Tzz[Relidx+i]);
						}

						for(j=1;j<=LenFD-(2-countZ);j++)
							VecTz[LenFD + (2-countZ) + j] = 2.0*T3Src - VecTz[LenFD + (2-countZ) - j];//TzSrc
						VecTz[LenFD +(2-countZ)] = T3Src;

						DxTx = rhojac*strF( VecTx, LenFD, xstep, Xvec);
						DyTy = rhojac*strF( VecTy, LenFD, ystep, Yvec);
						DzTz = rhojac*strF( VecTz, LenFD, zstep, Zvec);

						hW.Vz[Gindex] = DxTx/Bx + DyTy/By + DzTz;

#ifdef CFSPML
						//X dir absorption
						tempIdx = idx+(ipam[2]-LenFD)+ipam[9];
						if(tempIdx<=apr.nabs[0]+LenFD-1 || tempIdx>=ipam[10]+LenFD-apr.nabs[1])
						{
						tempIdx<apr.nabs[0]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[10]+LenFD-apr.nabs[1])+apr.nabs[0];
						Pidx = Pidx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
						
						hW.Vz[Gindex] -= Ax.Vz[Pidx]/Bx;
						hAx.Vz[Pidx] = DBx*DxTx - APDx*Ax.Vz[Pidx];
						}
						//Y dir absorption
						tempIdx = idy+(ipam[4]-LenFD);
						if(tempIdx<=apr.nabs[2]+LenFD-1 || tempIdx>=ipam[7]+LenFD-apr.nabs[3])
						{
						tempIdx<apr.nabs[2]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[7]+LenFD-apr.nabs[3])+apr.nabs[2];
						Pidx = Pidx*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[8]+2*LenFD) + idx*(ipam[8]+2*LenFD) + idz;

						hW.Vz[Gindex] -= Ay.Vz[Pidx]/By;
						hAy.Vz[Pidx] = DBy*DyTy - APDy*Ay.Vz[Pidx];
						}

#endif


					}//loop countZ(LenFD)
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

__global__ void CalVUCD(int Xvec, int Yvec, int Zvec, Real steph, Real *CoVx, Real* CoVy, mdparF mpa, derivF drv, wfield W, wfield hW,
			wfield Ax, wfield hAx, wfield Ay, wfield hAy, apara apr)
{//wrong
	//  <<<BPG.y, BPG.x>>>
	//gridDim.x<=cdx.nx  blockDim.x<=cdx.ny
	int n;
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int xiaoI;

	Real lambda,miu,rho,lam2mu;
	Real xix,xiy,xiz, etx,ety,etz, ztx,zty,ztz;//covariants
	Real DxVx[LenFD+1],DxVy[LenFD+1],DxVz[LenFD+1];
	Real DyVx[LenFD+1],DyVy[LenFD+1],DyVz[LenFD+1];
	Real DzVx[LenFD+1],DzVy[LenFD+1],DzVz[LenFD+1];
	Real DxiTxx,DetTxx,DztTxx, DxiTyy,DetTyy,DztTyy, DxiTzz,DetTzz,DztTzz;//vector of covariants mutiply space derivative in xi,eta and zeta direction, respectively.
	Real DxiTxy,DetTxy,DztTxy, DxiTxz,DetTxz,DztTxz, DxiTyz,DetTyz,DztTyz;//T6V3
	Real Bx,By;//absorb boundary pars
	Real xstep, ystep, zstep;
	int xinc, yinc, zinc;
	Real V3Src=0.0;//initial value
	
	xstep = steph*Xvec;
	ystep = steph*Yvec;
	zstep = steph*Zvec;
	//none direction
	xinc = Xvec*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD);//skip cdx.ny*cdx.nz
	yinc = Yvec*(ipam[8]+2*LenFD);//skip cdx.nz
	zinc = Zvec*1;

#ifdef CFSPML
	Real APDx,DBx, APDy,DBy;
	int Pidx,tempIdx;
#endif
	
	//This is computing for the velocity free surface condition by the unilateral compact MacCormack type difference scheme
	//Hixion & Turkel, 2000. And in Thesis is Equation 2.25

	//computational sequence
	//			FORTRAN			|	GPU
	//		SN	index	direction
	//  DzV3	4	nk2	F	-	|	3	nk2-1
	//		3	nk2-1	F	B	|	2	nk2-2
	//		2	nk2-2	F	B	|	1	nk2-3
	//		1	nk2-3	-	B	|	0	nk2-4


	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.x)//loop in current compute range with step of Launch Par
	{
		idx = countX + blockIdx.x + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=blockDim.x)
			{
				idy = countY + threadIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{  
					//case 1: nk2-4 layer, needed for UCDFD
					n = 0; countZ = -4;
					idz = ipam[8] + countZ;//valid point number + LenFD = last location of valid point
					Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;

					DzVx[n] = DRPFD( W.Vx, Gindex, zstep, zinc );
					DzVy[n] = DRPFD( W.Vy, Gindex, zstep, zinc );
					DzVz[n] = DRPFD( W.Vz, Gindex, zstep, zinc );


					//case 2: nk2-1 layer, top layer, needed for UCDFD
					n = 3; countZ = -1;
					idz = ipam[8] + countZ;//valid point number + LenFD = last location of valid point
					Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
					xiaoI = idx*(ipam[5]-ipam[4]+1+2*LenFD)*SeisGeo*SeisGeo + idy*SeisGeo*SeisGeo;//valid Y

					DxVx[n] = DRPFD( W.Vx, Gindex, xstep, xinc );
					DxVy[n] = DRPFD( W.Vy, Gindex, xstep, xinc );
					DxVz[n] = DRPFD( W.Vz, Gindex, xstep, xinc );

					DyVx[n] = DRPFD( W.Vx, Gindex, ystep, yinc );
					DyVy[n] = DRPFD( W.Vy, Gindex, ystep, yinc );
					DyVz[n] = DRPFD( W.Vz, Gindex, ystep, yinc );

					DzVx[n] = CoVx[xiaoI+0]*DxVx[n] + CoVx[xiaoI+1]*DxVy[n] + CoVx[xiaoI+2]*DxVz[n]
						+ CoVy[xiaoI+0]*DyVx[n] + CoVy[xiaoI+1]*DyVy[n] + CoVy[xiaoI+2]*DyVz[n];
					DzVy[n] = CoVx[xiaoI+3]*DxVx[n] + CoVx[xiaoI+4]*DxVy[n] + CoVx[xiaoI+5]*DxVz[n]
						+ CoVy[xiaoI+3]*DyVx[n] + CoVy[xiaoI+4]*DyVy[n] + CoVy[xiaoI+5]*DyVz[n];
					DzVz[n] = CoVx[xiaoI+6]*DxVx[n] + CoVx[xiaoI+7]*DxVy[n] + CoVx[xiaoI+8]*DxVz[n]
						+ CoVy[xiaoI+6]*DyVx[n] + CoVy[xiaoI+7]*DyVy[n] + CoVy[xiaoI+8]*DyVz[n];
					
					DzVx[n] = DzVx[n] + V3Src;
					DzVy[n] = DzVy[n] + V3Src;
					DzVz[n] = DzVz[n] + V3Src;
					

					//case 3: nk2-2 and nk2-3 layer, need to apply UCDFD
					for(n=1;n<=2;n++)
					{
						countZ = n-4;//-3 -2
						idz = ipam[8] + countZ;//valid point number + LenFD = last location of valid point
						Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
						
						DxVx[n] = DRPFD( W.Vx, Gindex, xstep, xinc );
						DxVy[n] = DRPFD( W.Vy, Gindex, xstep, xinc );
						DxVz[n] = DRPFD( W.Vz, Gindex, xstep, xinc );

						DyVx[n] = DRPFD( W.Vx, Gindex, ystep, yinc );
						DyVy[n] = DRPFD( W.Vy, Gindex, ystep, yinc );
						DyVz[n] = DRPFD( W.Vz, Gindex, ystep, yinc );

						DzVx[n] = UCDFD_R( W.Vx, Gindex, zstep, zinc ) - UCDFD_L( DzVx, n, zinc );
						DzVy[n] = UCDFD_R( W.Vy, Gindex, zstep, zinc ) - UCDFD_L( DzVy, n, zinc );
						DzVz[n] = UCDFD_R( W.Vz, Gindex, zstep, zinc ) - UCDFD_L( DzVz, n, zinc );


					}//loop for layer

					//compute time-domain partial derivative
					for(n=1;n<=3;n++)
					{
						countZ = n-4;
						idz = ipam[8] + countZ;//valid point number + LenFD = last location of valid point
						Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;

						rho = mpa.rho[Gindex];
						miu = rho*mpa.beta[Gindex]*mpa.beta[Gindex];
						lam2mu = rho*mpa.alpha[Gindex]*mpa.alpha[Gindex];
						lambda = lam2mu - 2.0*miu;

						xix = drv.xix[Gindex];
						xiy = drv.xiy[Gindex];
						xiz = drv.xiz[Gindex];
						etx = drv.etax[Gindex];
						ety = drv.etay[Gindex];
						etz = drv.etaz[Gindex];
						ztx = drv.zetax[Gindex];
						zty = drv.zetay[Gindex];
						ztz = drv.zetaz[Gindex];

#ifdef CFSPML
	APDx = apr.APDx[idx];	APDy = apr.APDy[idy];
	Bx = apr.Bx[idx];	By = apr.By[idy];
	DBx = apr.DBx[idx];	DBy = apr.DBy[idy];
#else	
	Bx = 1.0;	By = 1.0;
#endif
						
						DxiTxx = lam2mu*xix*DxVx[n] + lambda*xiy*DxVy[n] + lambda*xiz*DxVz[n];
						DetTxx = lam2mu*etx*DyVx[n] + lambda*ety*DyVy[n] + lambda*etz*DyVz[n];
						DztTxx = lam2mu*ztx*DzVx[n] + lambda*zty*DzVy[n] + lambda*ztz*DzVz[n];

						DxiTyy = lambda*xix*DxVx[n] + lam2mu*xiy*DxVy[n] + lambda*xiz*DxVz[n];
						DetTyy = lambda*etx*DyVx[n] + lam2mu*ety*DyVy[n] + lambda*etz*DyVz[n];
						DztTyy = lambda*ztx*DzVx[n] + lam2mu*zty*DzVy[n] + lambda*ztz*DzVz[n];
						
						DxiTzz = lambda*xix*DxVx[n] + lambda*xiy*DxVy[n] + lam2mu*xiz*DxVz[n];
						DetTzz = lambda*etx*DyVx[n] + lambda*ety*DyVy[n] + lam2mu*etz*DyVz[n];
						DztTzz = lambda*ztx*DzVx[n] + lambda*zty*DzVy[n] + lam2mu*ztz*DzVz[n];
						
						DxiTxy = miu*(xiy*DxVx[n] + xix*DxVy[n]);
						DetTxy = miu*(ety*DyVx[n] + etx*DyVy[n]);
						DztTxy = miu*(zty*DzVx[n] + ztx*DzVy[n]);

						DxiTxz = miu*(xiz*DxVx[n] + xix*DxVz[n]);
						DetTxz = miu*(etz*DyVx[n] + etx*DyVz[n]);
						DztTxz = miu*(ztz*DzVx[n] + ztx*DzVz[n]);
						
						DxiTyz = miu*(xiz*DxVy[n] + xiy*DxVz[n]);
						DetTyz = miu*(etz*DyVy[n] + ety*DyVz[n]);
						DztTyz = miu*(ztz*DzVy[n] + zty*DzVz[n]);
						
						hW.Txx[Gindex] = DxiTxx/Bx + DetTxx/By + DztTxx;
						hW.Tyy[Gindex] = DxiTyy/Bx + DetTyy/By + DztTyy;
						hW.Tzz[Gindex] = DxiTzz/Bx + DetTzz/By + DztTzz;
						hW.Txy[Gindex] = DxiTxy/Bx + DetTxy/By + DztTxy;
						hW.Txz[Gindex] = DxiTxz/Bx + DetTxz/By + DztTxz;
						hW.Tyz[Gindex] = DxiTyz/Bx + DetTyz/By + DztTyz;

#ifdef CFSPML
						tempIdx = idx+(ipam[2]-LenFD)+ipam[9];//idx+ipam[9]
						if(tempIdx<=apr.nabs[0]+LenFD-1 || tempIdx>=ipam[10]+LenFD-apr.nabs[1])//X-dir
						{
							tempIdx<apr.nabs[0]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[10]+LenFD-apr.nabs[1])+apr.nabs[0];
						        Pidx = Pidx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;

							hW.Txx[Gindex] -= Ax.Txx[Pidx]/Bx;
							hW.Tyy[Gindex] -= Ax.Tyy[Pidx]/Bx;
							hW.Tzz[Gindex] -= Ax.Tzz[Pidx]/Bx;
							hW.Txy[Gindex] -= Ax.Txy[Pidx]/Bx;
							hW.Txz[Gindex] -= Ax.Txz[Pidx]/Bx;
							hW.Tyz[Gindex] -= Ax.Tyz[Pidx]/Bx;

							hAx.Txx[Pidx] = DxiTxx*DBx - APDx*Ax.Txx[Pidx];
							hAx.Tyy[Pidx] = DxiTyy*DBx - APDx*Ax.Tyy[Pidx];
							hAx.Tzz[Pidx] = DxiTzz*DBx - APDx*Ax.Tzz[Pidx];
							hAx.Txy[Pidx] = DxiTxy*DBx - APDx*Ax.Txy[Pidx];
							hAx.Txz[Pidx] = DxiTxz*DBx - APDx*Ax.Txz[Pidx];
							hAx.Tyz[Pidx] = DxiTyz*DBx - APDx*Ax.Tyz[Pidx];
						}

						tempIdx = idy + (ipam[4]-LenFD);//idy
						if(tempIdx<=apr.nabs[2]+LenFD-1 || tempIdx>=ipam[7]+LenFD-apr.nabs[3])//Y-dir
						{
							tempIdx<apr.nabs[2]+LenFD ? Pidx=tempIdx-LenFD : Pidx=tempIdx-(ipam[7]+LenFD-apr.nabs[3])+apr.nabs[2];
							Pidx = Pidx*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[8]+2*LenFD) + idx*(ipam[8]+2*LenFD) + idz;

							hW.Txx[Gindex] -= Ay.Txx[Pidx]/By;
							hW.Tyy[Gindex] -= Ay.Tyy[Pidx]/By;
							hW.Tzz[Gindex] -= Ay.Tzz[Pidx]/By;
							hW.Txy[Gindex] -= Ay.Txy[Pidx]/By;
							hW.Txz[Gindex] -= Ay.Txz[Pidx]/By;
							hW.Tyz[Gindex] -= Ay.Tyz[Pidx]/By;

							hAy.Txx[Pidx] = DetTxx*DBy - APDy*Ay.Txx[Pidx];
							hAy.Tyy[Pidx] = DetTyy*DBy - APDy*Ay.Tyy[Pidx];
							hAy.Tzz[Pidx] = DetTzz*DBy - APDy*Ay.Tzz[Pidx];
							hAy.Txy[Pidx] = DetTxy*DBy - APDy*Ay.Txy[Pidx];
							hAy.Txz[Pidx] = DetTxz*DBy - APDy*Ay.Txz[Pidx];
							hAy.Tyz[Pidx] = DetTyz*DBy - APDy*Ay.Tyz[Pidx];
						}

#endif

					}

				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

__global__ void LoadForce(int Tindex, cindx cdx, Real steph, int nfrc, int nstf, forceF frc, Real *jac, Real *rho, wfield hW)
{
	int i;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int modX,modY,modZ;//modify to global index

	Real stf,A,d;
	int mid;
	for(i=0;i<nfrc;i+=gridDim.y*gridDim.x*blockDim.x)
	{
		mid = i + blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
		if( mid<nfrc && frc.locx[i]-ipam[9] >= ipam[2] && frc.locx[i]-ipam[9] <=ipam[3] && frc.locy[i] >=ipam[4] && frc.locy[i] <=ipam[5] )
		{
			modX = frc.locx[mid]-ipam[9]-(ipam[2]-LenFD);
			modY = frc.locy[mid]-(ipam[4]-LenFD);
			modZ = frc.locz[mid];
#ifdef SrcSmooth
			for(idx = modX-LenFD; idx<=modX+LenFD; idx++)
				for(idy = modY-LenFD; idy<=modY+LenFD; idy++)
					for(idz = modZ-LenFD; idz<=modZ+LenFD; idz++)
					{
						Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
						if( idx>=cdx.ni1 && idx<cdx.ni2 && idy>=cdx.nj1 && idy<cdx.nj2 && idz>=cdx.nk1 && idz<cdx.nk2 )
						{
							stf = frc.stf[mid*nstf + Tindex];
							d = frc.dnorm[mid*LenNorm*LenNorm*LenNorm + (idx - (modX-LenFD))*LenNorm*LenNorm +
								(idy - (modY-LenFD))*LenNorm + (idz - (modZ-LenFD))];
							
							A = steph*steph*steph*jac[Gindex];
							A = 1/(A*rho[Gindex])*d;
							
							if(idz == ipam[8]+LenFD-1)
								A = A*2.0;

							hW.Vx[Gindex] += stf*frc.fx[mid]*A;
							hW.Vy[Gindex] += stf*frc.fy[mid]*A;
							hW.Vz[Gindex] += stf*frc.fz[mid]*A;
						}
					}
#else
			idx = modX;	idy = modY;	idz = modZ;
			Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
			if( idx>=cdx.ni1 && idx<cdx.ni2 && idy>=cdx.nj1 && idy<cdx.nj2 && idz>=cdx.nk1 && idz<cdx.nk2 )
			{
				stf = frc.stf[mid*nstf + Tindex];
				d = 1.0;

				A = steph*steph*steph*jac[Gindex];
				A = 1/(A*rho[Gindex])*d;

				if(idz == ipam[8]+LenFD-1)
					A = A*2.0;

				hW.Vx[Gindex] += stf*frc.fx[mid]*A;
				hW.Vy[Gindex] += stf*frc.fy[mid]*A;
				hW.Vz[Gindex] += stf*frc.fz[mid]*A;
			}
#endif
		}//loop modXYZ
	}//loop i

}

__global__ void LoadMoment(int Tindex, cindx cdx, Real steph, int nmnt, int nstf, momentF mnt, Real *jac, wfield hW)
{
	int i;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int modX,modY,modZ;//modify to global index
	Real stf,A;

	int mid;
	for(i=0;i<nmnt;i+=gridDim.y*gridDim.x*blockDim.x)
	{
		mid = i + blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
		if( mid<nmnt && mnt.locx[i]-ipam[9] >= ipam[2] && mnt.locx[i]-ipam[9] <=ipam[3] && mnt.locy[i] >=ipam[4] && mnt.locy[i] <=ipam[5] )
		{
			modX = mnt.locx[mid]-ipam[9]-(ipam[2]-LenFD);
			modY = mnt.locy[mid]-(ipam[4]-LenFD);
			modZ = mnt.locz[mid];
#ifdef SrcSmooth
			for(idx = modX-LenFD; idx<=modX+LenFD; idx++)
				for(idy = modY-LenFD; idy<=modY+LenFD; idy++)
					for(idz = modZ-LenFD; idz<=modZ+LenFD; idz++)
					{
			//printf("PCS[%d]DEV[%d]loc(%d,%d,%d),modx=%d, mody=%d, modzz=%d, idx=%d, idy=%d, idz=%d\n",
			//	ipam[1],ipam[0], mnt.locx[mid],mnt.locy[mid],mnt.locz[mid], modX, modY, modZ, idx,idy,idz);
						Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
						if( idx>=cdx.ni1 && idx<cdx.ni2 && idy>=cdx.nj1 && idy<cdx.nj2 && idz>=cdx.nk1 && idz<cdx.nk2 )
						{
							stf = mnt.stf[mid*nstf + Tindex];
							A = mnt.dnorm[mid*LenNorm*LenNorm*LenNorm + (idx - (modX-LenFD))*LenNorm*LenNorm +
								(idy - (modY-LenFD))*LenNorm + (idz - (modZ-LenFD))];
							
							A = stf*A/(steph*steph*steph*jac[Gindex]);

							hW.Txx[Gindex] -= mnt.mxx[mid]*A;
							hW.Tyy[Gindex] -= mnt.myy[mid]*A;
							hW.Tzz[Gindex] -= mnt.mzz[mid]*A;
							hW.Txy[Gindex] -= mnt.mxy[mid]*A;
							hW.Txz[Gindex] -= mnt.mxz[mid]*A;
							hW.Tyz[Gindex] -= mnt.myz[mid]*A;
						}
					}
#else
			idx = modX;	idy = modY;	idz = modZ;
			Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
			if( idx>=cdx.ni1 && idx<cdx.ni2 && idy>=cdx.nj1 && idy<cdx.nj2 && idz>=cdx.nk1 && idz<cdx.nk2 )
			{
				stf = mnt.stf[mid*nstf + Tindex];
				A = 1.0;

				A = stf*A/(steph*steph*steph*jac[Gindex]);

				hW.Txx[Gindex] -= mnt.mxx[mid]*A;
				hW.Tyy[Gindex] -= mnt.myy[mid]*A;
				hW.Tzz[Gindex] -= mnt.mzz[mid]*A;
				hW.Txy[Gindex] -= mnt.mxy[mid]*A;
				hW.Txz[Gindex] -= mnt.mxz[mid]*A;
				hW.Tyz[Gindex] -= mnt.myz[mid]*A;
			}
#endif
		}//loop modXYZ
	}//loop i

}

__global__ void LoadRmom(cindx cdx, Real steph, int Dfpn, RmomF mnt, Real *jac, wfield hW)
{
	int i;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int modX,modY,modZ;//modify to global index

	Real A;
	int mid;

	for(i=0;i<Dfpn;i+=gridDim.y*gridDim.x*blockDim.x)
	{
		mid = i + blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
		if( mid<Dfpn && mnt.locx[i]-ipam[9] >= ipam[2] && mnt.locx[i]-ipam[9] <=ipam[3] && mnt.locy[i] >=ipam[4] && mnt.locy[i] <=ipam[5] )
		{
			modX = mnt.locx[mid]-ipam[9]-(ipam[2]-LenFD);
			modY = mnt.locy[mid]-(ipam[4]-LenFD);
			modZ = mnt.locz[mid];
#ifdef SrcSmooth
			for(idx = modX-LenFD; idx<=modX+LenFD; idx++)
				for(idy = modY-LenFD; idy<=modY+LenFD; idy++)
					for(idz = modZ-LenFD; idz<=modZ+LenFD; idz++)
					{
						Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
						if( idx>=cdx.ni1 && idx<cdx.ni2 && idy>=cdx.nj1 && idy<cdx.nj2 && idz>=cdx.nk1 && idz<cdx.nk2 )
						{
							A = mnt.dnorm[mid*LenNorm*LenNorm*LenNorm + (idx - (modX-LenFD))*LenNorm*LenNorm +
								(idy - (modY-LenFD))*LenNorm + (idz - (modZ-LenFD))];
							
							A = A/(steph*steph*steph*jac[Gindex]);

							hW.Txx[Gindex] -= mnt.mxx[mid]*A;
							hW.Tyy[Gindex] -= mnt.myy[mid]*A;
							hW.Tzz[Gindex] -= mnt.mzz[mid]*A;
							hW.Txy[Gindex] -= mnt.mxy[mid]*A;
							hW.Txz[Gindex] -= mnt.mxz[mid]*A;
							hW.Tyz[Gindex] -= mnt.myz[mid]*A;
						}
					}
#else
			idx = modX;	idy = modY;	idz = modZ;
			Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
			//printf("PCS[%d]DEV[%d]loc(%d,%d,%d),modx=%d, mody=%d, modzz=%d, idx=%d, idy=%d, idz=%d\n",
			//	ipam[1],ipam[0], mnt.locx[mid],mnt.locy[mid],mnt.locz[mid], modX, modY, modZ, idx,idy,idz);
			if( idx>=cdx.ni1 && idx<cdx.ni2 && idy>=cdx.nj1 && idy<cdx.nj2 && idz>=cdx.nk1 && idz<cdx.nk2 )
			{
				A = 1.0;
				
				A = A/(steph*steph*steph*jac[Gindex]);

				hW.Txx[Gindex] -= mnt.mxx[mid]*A;
				hW.Tyy[Gindex] -= mnt.myy[mid]*A;
				hW.Tzz[Gindex] -= mnt.mzz[mid]*A;
				hW.Txy[Gindex] -= mnt.mxy[mid]*A;
				hW.Txz[Gindex] -= mnt.mxz[mid]*A;
				hW.Tyz[Gindex] -= mnt.myz[mid]*A;
			}
#endif
		}//loop modXYZ
	}//loop i

}

__global__ void IterationBegin(Real stept, Real alpha, Real beta, wfield FW, wfield hW, wfield tW, wfield W,
		    int *nabs, wfield FAx, wfield hAx, wfield tAx, wfield Ax,	wfield FAy, wfield hAy, wfield tAy, wfield Ay,
			       wfield FAz, wfield hAz, wfield tAz, wfield Az)
{
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int Pidx;

	alpha *= stept;
	beta *= stept;

	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<ipam[8]+LenFD)//vaild point with one virtual bounds
						{
							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;

							W.Txx[Gindex] = FW.Txx[Gindex] + alpha*hW.Txx[Gindex];
							W.Tyy[Gindex] = FW.Tyy[Gindex] + alpha*hW.Tyy[Gindex];
							W.Tzz[Gindex] = FW.Tzz[Gindex] + alpha*hW.Tzz[Gindex];
							W.Txy[Gindex] = FW.Txy[Gindex] + alpha*hW.Txy[Gindex];
							W.Txz[Gindex] = FW.Txz[Gindex] + alpha*hW.Txz[Gindex];
							W.Tyz[Gindex] = FW.Tyz[Gindex] + alpha*hW.Tyz[Gindex];
							W.Vx[Gindex] = FW.Vx[Gindex] + alpha*hW.Vx[Gindex];
							W.Vy[Gindex] = FW.Vy[Gindex] + alpha*hW.Vy[Gindex];
							W.Vz[Gindex] = FW.Vz[Gindex] + alpha*hW.Vz[Gindex];

							tW.Txx[Gindex] = FW.Txx[Gindex] + beta*hW.Txx[Gindex];
							tW.Tyy[Gindex] = FW.Tyy[Gindex] + beta*hW.Tyy[Gindex];
							tW.Tzz[Gindex] = FW.Tzz[Gindex] + beta*hW.Tzz[Gindex];
							tW.Txy[Gindex] = FW.Txy[Gindex] + beta*hW.Txy[Gindex];
							tW.Txz[Gindex] = FW.Txz[Gindex] + beta*hW.Txz[Gindex];
							tW.Tyz[Gindex] = FW.Tyz[Gindex] + beta*hW.Tyz[Gindex];
							tW.Vx[Gindex] = FW.Vx[Gindex] + beta*hW.Vx[Gindex];
							tW.Vy[Gindex] = FW.Vy[Gindex] + beta*hW.Vy[Gindex];
							tW.Vz[Gindex] = FW.Vz[Gindex] + beta*hW.Vz[Gindex];

#ifdef DisBug
if(zbx == idx+(ipam[2]-LenFD)+ipam[9] && zby == idy+(ipam[4]-LenFD) && zbz == idz)
{
	printf(" IteBeg W-->(%d,%d,%d) Txx=%e, Tyy=%e, Tzz=%e\n\tTxy=%e, Txz=%e,Tzz=%e\n\tVx=%e Vy=%e Vz=%e\n",
		zbx,zby,idz,W.Txx[Gindex],W.Tyy[Gindex],W.Tzz[Gindex],W.Txy[Gindex],W.Txz[Gindex],W.Tyz[Gindex],W.Vx[Gindex],W.Vy[Gindex],W.Vz[Gindex]);
	printf(" IteBeg tW-->(%d,%d,%d) Txx=%e, Tyy=%e, Tzz=%e\n\tTxy=%e, Txz=%e,Tzz=%e\n\tVx=%e Vy=%e Vz=%e\n",
		zbx,zby,idz,tW.Txx[Gindex],tW.Tyy[Gindex],tW.Tzz[Gindex],tW.Txy[Gindex],tW.Txz[Gindex],tW.Tyz[Gindex],tW.Vx[Gindex],tW.Vy[Gindex],tW.Vz[Gindex]);
}
#endif

#ifdef CFSPML
							//X dir absorption
							if(idx <= nabs[0] + nabs[1] + LenFD -1 )
							{
								Pidx = (idx - LenFD)*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
								
								Ax.Txx[Pidx] = FAx.Txx[Pidx] + alpha*hAx.Txx[Pidx];
								Ax.Tyy[Pidx] = FAx.Tyy[Pidx] + alpha*hAx.Tyy[Pidx];
								Ax.Tzz[Pidx] = FAx.Tzz[Pidx] + alpha*hAx.Tzz[Pidx];
								Ax.Txy[Pidx] = FAx.Txy[Pidx] + alpha*hAx.Txy[Pidx];
								Ax.Txz[Pidx] = FAx.Txz[Pidx] + alpha*hAx.Txz[Pidx];
								Ax.Tyz[Pidx] = FAx.Tyz[Pidx] + alpha*hAx.Tyz[Pidx];
								Ax.Vx[Pidx] = FAx.Vx[Pidx] + alpha*hAx.Vx[Pidx];
								Ax.Vy[Pidx] = FAx.Vy[Pidx] + alpha*hAx.Vy[Pidx];
								Ax.Vz[Pidx] = FAx.Vz[Pidx] + alpha*hAx.Vz[Pidx];

								tAx.Txx[Pidx] = FAx.Txx[Pidx] + beta*hAx.Txx[Pidx];
								tAx.Tyy[Pidx] = FAx.Tyy[Pidx] + beta*hAx.Tyy[Pidx];
								tAx.Tzz[Pidx] = FAx.Tzz[Pidx] + beta*hAx.Tzz[Pidx];
								tAx.Txy[Pidx] = FAx.Txy[Pidx] + beta*hAx.Txy[Pidx];
								tAx.Txz[Pidx] = FAx.Txz[Pidx] + beta*hAx.Txz[Pidx];
								tAx.Tyz[Pidx] = FAx.Tyz[Pidx] + beta*hAx.Tyz[Pidx];
								tAx.Vx[Pidx] = FAx.Vx[Pidx] + beta*hAx.Vx[Pidx];
								tAx.Vy[Pidx] = FAx.Vy[Pidx] + beta*hAx.Vy[Pidx];
								tAx.Vz[Pidx] = FAx.Vz[Pidx] + beta*hAx.Vz[Pidx];
							}

							//Y dir absorption
							if(idy <= nabs[2] + nabs[3] +LenFD-1)
							{
								Pidx = (idy-LenFD)*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[8]+2*LenFD) + idx*(ipam[8]+2*LenFD) + idz;

								Ay.Txx[Pidx] = FAy.Txx[Pidx] + alpha*hAy.Txx[Pidx];
								Ay.Tyy[Pidx] = FAy.Tyy[Pidx] + alpha*hAy.Tyy[Pidx];
								Ay.Tzz[Pidx] = FAy.Tzz[Pidx] + alpha*hAy.Tzz[Pidx];
								Ay.Txy[Pidx] = FAy.Txy[Pidx] + alpha*hAy.Txy[Pidx];
								Ay.Txz[Pidx] = FAy.Txz[Pidx] + alpha*hAy.Txz[Pidx];
								Ay.Tyz[Pidx] = FAy.Tyz[Pidx] + alpha*hAy.Tyz[Pidx];
								Ay.Vx[Pidx] = FAy.Vx[Pidx] + alpha*hAy.Vx[Pidx];
								Ay.Vy[Pidx] = FAy.Vy[Pidx] + alpha*hAy.Vy[Pidx];
								Ay.Vz[Pidx] = FAy.Vz[Pidx] + alpha*hAy.Vz[Pidx];

								tAy.Txx[Pidx] = FAy.Txx[Pidx] + beta*hAy.Txx[Pidx];
								tAy.Tyy[Pidx] = FAy.Tyy[Pidx] + beta*hAy.Tyy[Pidx];
								tAy.Tzz[Pidx] = FAy.Tzz[Pidx] + beta*hAy.Tzz[Pidx];
								tAy.Txy[Pidx] = FAy.Txy[Pidx] + beta*hAy.Txy[Pidx];
								tAy.Txz[Pidx] = FAy.Txz[Pidx] + beta*hAy.Txz[Pidx];
								tAy.Tyz[Pidx] = FAy.Tyz[Pidx] + beta*hAy.Tyz[Pidx];
								tAy.Vx[Pidx] = FAy.Vx[Pidx] + beta*hAy.Vx[Pidx];
								tAy.Vy[Pidx] = FAy.Vy[Pidx] + beta*hAy.Vy[Pidx];
								tAy.Vz[Pidx] = FAy.Vz[Pidx] + beta*hAy.Vz[Pidx];
							}
							
							//Z dir absorption
							if(idz <= nabs[4] + nabs[5] +LenFD-1)
							{
								Pidx = (idz-LenFD)*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[5]-ipam[4]+1+2*LenFD) 
									+ idx*(ipam[5]-ipam[4]+1+2*LenFD) + idy;

								Az.Txx[Pidx] = FAz.Txx[Pidx] + alpha*hAz.Txx[Pidx];
								Az.Tyy[Pidx] = FAz.Tyy[Pidx] + alpha*hAz.Tyy[Pidx];
								Az.Tzz[Pidx] = FAz.Tzz[Pidx] + alpha*hAz.Tzz[Pidx];
								Az.Txy[Pidx] = FAz.Txy[Pidx] + alpha*hAz.Txy[Pidx];
								Az.Txz[Pidx] = FAz.Txz[Pidx] + alpha*hAz.Txz[Pidx];
								Az.Tyz[Pidx] = FAz.Tyz[Pidx] + alpha*hAz.Tyz[Pidx];
								Az.Vx[Pidx] = FAz.Vx[Pidx] + alpha*hAz.Vx[Pidx];
								Az.Vy[Pidx] = FAz.Vy[Pidx] + alpha*hAz.Vy[Pidx];
								Az.Vz[Pidx] = FAz.Vz[Pidx] + alpha*hAz.Vz[Pidx];

								tAz.Txx[Pidx] = FAz.Txx[Pidx] + beta*hAz.Txx[Pidx];
								tAz.Tyy[Pidx] = FAz.Tyy[Pidx] + beta*hAz.Tyy[Pidx];
								tAz.Tzz[Pidx] = FAz.Tzz[Pidx] + beta*hAz.Tzz[Pidx];
								tAz.Txy[Pidx] = FAz.Txy[Pidx] + beta*hAz.Txy[Pidx];
								tAz.Txz[Pidx] = FAz.Txz[Pidx] + beta*hAz.Txz[Pidx];
								tAz.Tyz[Pidx] = FAz.Tyz[Pidx] + beta*hAz.Tyz[Pidx];
								tAz.Vx[Pidx] = FAz.Vx[Pidx] + beta*hAz.Vx[Pidx];
								tAz.Vy[Pidx] = FAz.Vy[Pidx] + beta*hAz.Vy[Pidx];
								tAz.Vz[Pidx] = FAz.Vz[Pidx] + beta*hAz.Vz[Pidx];
							}
#endif

						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

__global__ void IterationInner(Real stept, Real alpha, Real beta, wfield FW, wfield hW, wfield tW, wfield W,
		    int *nabs, wfield FAx, wfield hAx, wfield tAx, wfield Ax,	wfield FAy, wfield hAy, wfield tAy, wfield Ay,
			       wfield FAz, wfield hAz, wfield tAz, wfield Az)
{
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int Pidx;

	alpha *= stept;
	beta *= stept;

	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<ipam[8]+LenFD)//vaild point with one virtual bounds
						{
							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;

							W.Txx[Gindex] = FW.Txx[Gindex] + alpha*hW.Txx[Gindex];
							W.Tyy[Gindex] = FW.Tyy[Gindex] + alpha*hW.Tyy[Gindex];
							W.Tzz[Gindex] = FW.Tzz[Gindex] + alpha*hW.Tzz[Gindex];
							W.Txy[Gindex] = FW.Txy[Gindex] + alpha*hW.Txy[Gindex];
							W.Txz[Gindex] = FW.Txz[Gindex] + alpha*hW.Txz[Gindex];
							W.Tyz[Gindex] = FW.Tyz[Gindex] + alpha*hW.Tyz[Gindex];
							W.Vx[Gindex] = FW.Vx[Gindex] + alpha*hW.Vx[Gindex];
							W.Vy[Gindex] = FW.Vy[Gindex] + alpha*hW.Vy[Gindex];
							W.Vz[Gindex] = FW.Vz[Gindex] + alpha*hW.Vz[Gindex];

							tW.Txx[Gindex] = tW.Txx[Gindex] + beta*hW.Txx[Gindex];
							tW.Tyy[Gindex] = tW.Tyy[Gindex] + beta*hW.Tyy[Gindex];
							tW.Tzz[Gindex] = tW.Tzz[Gindex] + beta*hW.Tzz[Gindex];
							tW.Txy[Gindex] = tW.Txy[Gindex] + beta*hW.Txy[Gindex];
							tW.Txz[Gindex] = tW.Txz[Gindex] + beta*hW.Txz[Gindex];
							tW.Tyz[Gindex] = tW.Tyz[Gindex] + beta*hW.Tyz[Gindex];
							tW.Vx[Gindex] = tW.Vx[Gindex] + beta*hW.Vx[Gindex];
							tW.Vy[Gindex] = tW.Vy[Gindex] + beta*hW.Vy[Gindex];
							tW.Vz[Gindex] = tW.Vz[Gindex] + beta*hW.Vz[Gindex];

#ifdef DisBug
if(zbx == idx+(ipam[2]-LenFD)+ipam[9] && zby == idy+(ipam[4]-LenFD) && zbz == idz)
{
	printf(" IteInn W-->(%d,%d,%d) Txx=%e, Tyy=%e, Tzz=%e\n\tTxy=%e, Txz=%e,Tzz=%e\n\tVx=%e Vy=%e Vz=%e\n",
		zbx,zby,idz,W.Txx[Gindex],W.Tyy[Gindex],W.Tzz[Gindex],W.Txy[Gindex],W.Txz[Gindex],W.Tyz[Gindex],W.Vx[Gindex],W.Vy[Gindex],W.Vz[Gindex]);
	printf(" IteInn tW-->(%d,%d,%d) Txx=%e, Tyy=%e, Tzz=%e\n\tTxy=%e, Txz=%e,Tzz=%e\n\tVx=%e Vy=%e Vz=%e\n",
		zbx,zby,idz,tW.Txx[Gindex],tW.Tyy[Gindex],tW.Tzz[Gindex],tW.Txy[Gindex],tW.Txz[Gindex],tW.Tyz[Gindex],tW.Vx[Gindex],tW.Vy[Gindex],tW.Vz[Gindex]);
}
#endif

#ifdef CFSPML
							//X dir absorption
							if(idx <= nabs[0] + nabs[1] + LenFD -1 )
							{
								Pidx = (idx - LenFD)*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
								
								Ax.Txx[Pidx] = FAx.Txx[Pidx] + alpha*hAx.Txx[Pidx];
								Ax.Tyy[Pidx] = FAx.Tyy[Pidx] + alpha*hAx.Tyy[Pidx];
								Ax.Tzz[Pidx] = FAx.Tzz[Pidx] + alpha*hAx.Tzz[Pidx];
								Ax.Txy[Pidx] = FAx.Txy[Pidx] + alpha*hAx.Txy[Pidx];
								Ax.Txz[Pidx] = FAx.Txz[Pidx] + alpha*hAx.Txz[Pidx];
								Ax.Tyz[Pidx] = FAx.Tyz[Pidx] + alpha*hAx.Tyz[Pidx];
								Ax.Vx[Pidx] = FAx.Vx[Pidx] + alpha*hAx.Vx[Pidx];
								Ax.Vy[Pidx] = FAx.Vy[Pidx] + alpha*hAx.Vy[Pidx];
								Ax.Vz[Pidx] = FAx.Vz[Pidx] + alpha*hAx.Vz[Pidx];

								tAx.Txx[Pidx] = tAx.Txx[Pidx] + beta*hAx.Txx[Pidx];
								tAx.Tyy[Pidx] = tAx.Tyy[Pidx] + beta*hAx.Tyy[Pidx];
								tAx.Tzz[Pidx] = tAx.Tzz[Pidx] + beta*hAx.Tzz[Pidx];
								tAx.Txy[Pidx] = tAx.Txy[Pidx] + beta*hAx.Txy[Pidx];
								tAx.Txz[Pidx] = tAx.Txz[Pidx] + beta*hAx.Txz[Pidx];
								tAx.Tyz[Pidx] = tAx.Tyz[Pidx] + beta*hAx.Tyz[Pidx];
								tAx.Vx[Pidx] = tAx.Vx[Pidx] + beta*hAx.Vx[Pidx];
								tAx.Vy[Pidx] = tAx.Vy[Pidx] + beta*hAx.Vy[Pidx];
								tAx.Vz[Pidx] = tAx.Vz[Pidx] + beta*hAx.Vz[Pidx];
							}

							//Y dir absorption
							if(idy <= nabs[2] + nabs[3] +LenFD-1)
							{
								Pidx = (idy-LenFD)*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[8]+2*LenFD) + idx*(ipam[8]+2*LenFD) + idz;

								Ay.Txx[Pidx] = FAy.Txx[Pidx] + alpha*hAy.Txx[Pidx];
								Ay.Tyy[Pidx] = FAy.Tyy[Pidx] + alpha*hAy.Tyy[Pidx];
								Ay.Tzz[Pidx] = FAy.Tzz[Pidx] + alpha*hAy.Tzz[Pidx];
								Ay.Txy[Pidx] = FAy.Txy[Pidx] + alpha*hAy.Txy[Pidx];
								Ay.Txz[Pidx] = FAy.Txz[Pidx] + alpha*hAy.Txz[Pidx];
								Ay.Tyz[Pidx] = FAy.Tyz[Pidx] + alpha*hAy.Tyz[Pidx];
								Ay.Vx[Pidx] = FAy.Vx[Pidx] + alpha*hAy.Vx[Pidx];
								Ay.Vy[Pidx] = FAy.Vy[Pidx] + alpha*hAy.Vy[Pidx];
								Ay.Vz[Pidx] = FAy.Vz[Pidx] + alpha*hAy.Vz[Pidx];

								tAy.Txx[Pidx] = tAy.Txx[Pidx] + beta*hAy.Txx[Pidx];
								tAy.Tyy[Pidx] = tAy.Tyy[Pidx] + beta*hAy.Tyy[Pidx];
								tAy.Tzz[Pidx] = tAy.Tzz[Pidx] + beta*hAy.Tzz[Pidx];
								tAy.Txy[Pidx] = tAy.Txy[Pidx] + beta*hAy.Txy[Pidx];
								tAy.Txz[Pidx] = tAy.Txz[Pidx] + beta*hAy.Txz[Pidx];
								tAy.Tyz[Pidx] = tAy.Tyz[Pidx] + beta*hAy.Tyz[Pidx];
								tAy.Vx[Pidx] = tAy.Vx[Pidx] + beta*hAy.Vx[Pidx];
								tAy.Vy[Pidx] = tAy.Vy[Pidx] + beta*hAy.Vy[Pidx];
								tAy.Vz[Pidx] = tAy.Vz[Pidx] + beta*hAy.Vz[Pidx];
							}
							
							//Z dir absorption
							if(idz <= nabs[4] + nabs[5] +LenFD-1)
							{
								Pidx = (idz-LenFD)*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[5]-ipam[4]+1+2*LenFD) 
									+ idx*(ipam[5]-ipam[4]+1+2*LenFD) + idy;

								Az.Txx[Pidx] = FAz.Txx[Pidx] + alpha*hAz.Txx[Pidx];
								Az.Tyy[Pidx] = FAz.Tyy[Pidx] + alpha*hAz.Tyy[Pidx];
								Az.Tzz[Pidx] = FAz.Tzz[Pidx] + alpha*hAz.Tzz[Pidx];
								Az.Txy[Pidx] = FAz.Txy[Pidx] + alpha*hAz.Txy[Pidx];
								Az.Txz[Pidx] = FAz.Txz[Pidx] + alpha*hAz.Txz[Pidx];
								Az.Tyz[Pidx] = FAz.Tyz[Pidx] + alpha*hAz.Tyz[Pidx];
								Az.Vx[Pidx] = FAz.Vx[Pidx] + alpha*hAz.Vx[Pidx];
								Az.Vy[Pidx] = FAz.Vy[Pidx] + alpha*hAz.Vy[Pidx];
								Az.Vz[Pidx] = FAz.Vz[Pidx] + alpha*hAz.Vz[Pidx];

								tAz.Txx[Pidx] = tAz.Txx[Pidx] + beta*hAz.Txx[Pidx];
								tAz.Tyy[Pidx] = tAz.Tyy[Pidx] + beta*hAz.Tyy[Pidx];
								tAz.Tzz[Pidx] = tAz.Tzz[Pidx] + beta*hAz.Tzz[Pidx];
								tAz.Txy[Pidx] = tAz.Txy[Pidx] + beta*hAz.Txy[Pidx];
								tAz.Txz[Pidx] = tAz.Txz[Pidx] + beta*hAz.Txz[Pidx];
								tAz.Tyz[Pidx] = tAz.Tyz[Pidx] + beta*hAz.Tyz[Pidx];
								tAz.Vx[Pidx] = tAz.Vx[Pidx] + beta*hAz.Vx[Pidx];
								tAz.Vy[Pidx] = tAz.Vy[Pidx] + beta*hAz.Vy[Pidx];
								tAz.Vz[Pidx] = tAz.Vz[Pidx] + beta*hAz.Vz[Pidx];
							}
#endif

							
						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

__global__ void IterationFinal(Real stept, Real beta, wfield hW, wfield tW, wfield W,
		    int *nabs, wfield hAx, wfield tAx, wfield Ax,	wfield hAy, wfield tAy, wfield Ay,
			       wfield hAz, wfield tAz, wfield Az)
{
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int Pidx;

	beta *= stept;

	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<ipam[8]+LenFD)//vaild point with one virtual bounds
						{
							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;

							W.Txx[Gindex] = tW.Txx[Gindex] + beta*hW.Txx[Gindex];
							W.Tyy[Gindex] = tW.Tyy[Gindex] + beta*hW.Tyy[Gindex];
							W.Tzz[Gindex] = tW.Tzz[Gindex] + beta*hW.Tzz[Gindex];
							W.Txy[Gindex] = tW.Txy[Gindex] + beta*hW.Txy[Gindex];
							W.Txz[Gindex] = tW.Txz[Gindex] + beta*hW.Txz[Gindex];
							W.Tyz[Gindex] = tW.Tyz[Gindex] + beta*hW.Tyz[Gindex];
							W.Vx[Gindex] = tW.Vx[Gindex] + beta*hW.Vx[Gindex];
							W.Vy[Gindex] = tW.Vy[Gindex] + beta*hW.Vy[Gindex];
							W.Vz[Gindex] = tW.Vz[Gindex] + beta*hW.Vz[Gindex];

#ifdef DisBug
if(zbx == idx+(ipam[2]-LenFD)+ipam[9] && zby == idy+(ipam[4]-LenFD) && zbz == idz)
{
	printf(" IteFin W-->(%d,%d,%d) Txx=%e, Tyy=%e, Tzz=%e\n\tTxy=%e, Txz=%e,Tzz=%e\n\tVx=%e Vy=%e Vz=%e\n",
		zbx,zby,idz,W.Txx[Gindex],W.Tyy[Gindex],W.Tzz[Gindex],W.Txy[Gindex],W.Txz[Gindex],W.Tyz[Gindex],W.Vx[Gindex],W.Vy[Gindex],W.Vz[Gindex]);
}
#endif

#ifdef CFSPML
							//X dir absorption
							if(idx <= nabs[0] + nabs[1] + LenFD -1 )
							{
								Pidx = (idx - LenFD)*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
								
								Ax.Txx[Pidx] = tAx.Txx[Pidx] + beta*hAx.Txx[Pidx];
								Ax.Tyy[Pidx] = tAx.Tyy[Pidx] + beta*hAx.Tyy[Pidx];
								Ax.Tzz[Pidx] = tAx.Tzz[Pidx] + beta*hAx.Tzz[Pidx];
								Ax.Txy[Pidx] = tAx.Txy[Pidx] + beta*hAx.Txy[Pidx];
								Ax.Txz[Pidx] = tAx.Txz[Pidx] + beta*hAx.Txz[Pidx];
								Ax.Tyz[Pidx] = tAx.Tyz[Pidx] + beta*hAx.Tyz[Pidx];
								Ax.Vx[Pidx] = tAx.Vx[Pidx] + beta*hAx.Vx[Pidx];
								Ax.Vy[Pidx] = tAx.Vy[Pidx] + beta*hAx.Vy[Pidx];
								Ax.Vz[Pidx] = tAx.Vz[Pidx] + beta*hAx.Vz[Pidx];
							}

							//Y dir absorption
							if(idy <= nabs[2] + nabs[3] +LenFD-1)
							{
								Pidx = (idy-LenFD)*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[8]+2*LenFD) + idx*(ipam[8]+2*LenFD) + idz;

								Ay.Txx[Pidx] = tAy.Txx[Pidx] + beta*hAy.Txx[Pidx];
								Ay.Tyy[Pidx] = tAy.Tyy[Pidx] + beta*hAy.Tyy[Pidx];
								Ay.Tzz[Pidx] = tAy.Tzz[Pidx] + beta*hAy.Tzz[Pidx];
								Ay.Txy[Pidx] = tAy.Txy[Pidx] + beta*hAy.Txy[Pidx];
								Ay.Txz[Pidx] = tAy.Txz[Pidx] + beta*hAy.Txz[Pidx];
								Ay.Tyz[Pidx] = tAy.Tyz[Pidx] + beta*hAy.Tyz[Pidx];
								Ay.Vx[Pidx] = tAy.Vx[Pidx] + beta*hAy.Vx[Pidx];
								Ay.Vy[Pidx] = tAy.Vy[Pidx] + beta*hAy.Vy[Pidx];
								Ay.Vz[Pidx] = tAy.Vz[Pidx] + beta*hAy.Vz[Pidx];
							}
							
							//Z dir absorption
							if(idz <= nabs[4] + nabs[5] +LenFD-1)
							{
								Pidx = (idz-LenFD)*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[5]-ipam[4]+1+2*LenFD) 
									+ idx*(ipam[5]-ipam[4]+1+2*LenFD) + idy;

								Az.Txx[Pidx] = tAz.Txx[Pidx] + beta*hAz.Txx[Pidx];
								Az.Tyy[Pidx] = tAz.Tyy[Pidx] + beta*hAz.Tyy[Pidx];
								Az.Tzz[Pidx] = tAz.Tzz[Pidx] + beta*hAz.Tzz[Pidx];
								Az.Txy[Pidx] = tAz.Txy[Pidx] + beta*hAz.Txy[Pidx];
								Az.Txz[Pidx] = tAz.Txz[Pidx] + beta*hAz.Txz[Pidx];
								Az.Tyz[Pidx] = tAz.Tyz[Pidx] + beta*hAz.Tyz[Pidx];
								Az.Vx[Pidx] = tAz.Vx[Pidx] + beta*hAz.Vx[Pidx];
								Az.Vy[Pidx] = tAz.Vy[Pidx] + beta*hAz.Vy[Pidx];
								Az.Vz[Pidx] = tAz.Vz[Pidx] + beta*hAz.Vz[Pidx];
							}
#endif


						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

__global__ void IterationFinalPV(Real stept, Real beta, PeakVel Dpv, wfield hW, wfield tW, wfield W,
		    int *nabs, wfield hAx, wfield tAx, wfield Ax,	wfield hAy, wfield tAy, wfield Ay,
			       wfield hAz, wfield tAz, wfield Az)
{
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int xiaoI;
	int Pidx;

	beta *= stept;

	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<ipam[8]+LenFD)//vaild point with one virtual bounds
						{
							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
							xiaoI = idx*(ipam[5]-ipam[4]+1+2*LenFD) + idy;

							W.Txx[Gindex] = tW.Txx[Gindex] + beta*hW.Txx[Gindex];
							W.Tyy[Gindex] = tW.Tyy[Gindex] + beta*hW.Tyy[Gindex];
							W.Tzz[Gindex] = tW.Tzz[Gindex] + beta*hW.Tzz[Gindex];
							W.Txy[Gindex] = tW.Txy[Gindex] + beta*hW.Txy[Gindex];
							W.Txz[Gindex] = tW.Txz[Gindex] + beta*hW.Txz[Gindex];
							W.Tyz[Gindex] = tW.Tyz[Gindex] + beta*hW.Tyz[Gindex];
							W.Vx[Gindex] = tW.Vx[Gindex] + beta*hW.Vx[Gindex];
							W.Vy[Gindex] = tW.Vy[Gindex] + beta*hW.Vy[Gindex];
							W.Vz[Gindex] = tW.Vz[Gindex] + beta*hW.Vz[Gindex];
							
							if(idz == ipam[8]+LenFD-1)
							{
								Dpv.Vx[xiaoI] = MAX( ABS(W.Vx[Gindex]), ABS(Dpv.Vx[xiaoI]) );
								Dpv.Vy[xiaoI] = MAX( ABS(W.Vy[Gindex]), ABS(Dpv.Vy[xiaoI]) );
								Dpv.Vz[xiaoI] = MAX( ABS(W.Vz[Gindex]), ABS(Dpv.Vz[xiaoI]) );
							}


#ifdef DisBug
if(zbx == idx+(ipam[2]-LenFD)+ipam[9] && zby == idy+(ipam[4]-LenFD) && zbz == idz)
{
	printf(" IteFin W-->(%d,%d,%d) Txx=%e, Tyy=%e, Tzz=%e\n\tTxy=%e, Txz=%e,Tzz=%e\n\tVx=%e Vy=%e Vz=%e\n",
		zbx,zby,idz,W.Txx[Gindex],W.Tyy[Gindex],W.Tzz[Gindex],W.Txy[Gindex],W.Txz[Gindex],W.Tyz[Gindex],W.Vx[Gindex],W.Vy[Gindex],W.Vz[Gindex]);
}
#endif

#ifdef CFSPML
							//X dir absorption
							if(idx <= nabs[0] + nabs[1] + LenFD -1 )
							{
								Pidx = (idx - LenFD)*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
								
								Ax.Txx[Pidx] = tAx.Txx[Pidx] + beta*hAx.Txx[Pidx];
								Ax.Tyy[Pidx] = tAx.Tyy[Pidx] + beta*hAx.Tyy[Pidx];
								Ax.Tzz[Pidx] = tAx.Tzz[Pidx] + beta*hAx.Tzz[Pidx];
								Ax.Txy[Pidx] = tAx.Txy[Pidx] + beta*hAx.Txy[Pidx];
								Ax.Txz[Pidx] = tAx.Txz[Pidx] + beta*hAx.Txz[Pidx];
								Ax.Tyz[Pidx] = tAx.Tyz[Pidx] + beta*hAx.Tyz[Pidx];
								Ax.Vx[Pidx] = tAx.Vx[Pidx] + beta*hAx.Vx[Pidx];
								Ax.Vy[Pidx] = tAx.Vy[Pidx] + beta*hAx.Vy[Pidx];
								Ax.Vz[Pidx] = tAx.Vz[Pidx] + beta*hAx.Vz[Pidx];
							}

							//Y dir absorption
							if(idy <= nabs[2] + nabs[3] +LenFD-1)
							{
								Pidx = (idy-LenFD)*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[8]+2*LenFD) + idx*(ipam[8]+2*LenFD) + idz;

								Ay.Txx[Pidx] = tAy.Txx[Pidx] + beta*hAy.Txx[Pidx];
								Ay.Tyy[Pidx] = tAy.Tyy[Pidx] + beta*hAy.Tyy[Pidx];
								Ay.Tzz[Pidx] = tAy.Tzz[Pidx] + beta*hAy.Tzz[Pidx];
								Ay.Txy[Pidx] = tAy.Txy[Pidx] + beta*hAy.Txy[Pidx];
								Ay.Txz[Pidx] = tAy.Txz[Pidx] + beta*hAy.Txz[Pidx];
								Ay.Tyz[Pidx] = tAy.Tyz[Pidx] + beta*hAy.Tyz[Pidx];
								Ay.Vx[Pidx] = tAy.Vx[Pidx] + beta*hAy.Vx[Pidx];
								Ay.Vy[Pidx] = tAy.Vy[Pidx] + beta*hAy.Vy[Pidx];
								Ay.Vz[Pidx] = tAy.Vz[Pidx] + beta*hAy.Vz[Pidx];
							}
							
							//Z dir absorption
							if(idz <= nabs[4] + nabs[5] +LenFD-1)
							{
								Pidx = (idz-LenFD)*(ipam[3]-ipam[2]+1+2*LenFD)*(ipam[5]-ipam[4]+1+2*LenFD) 
									+ idx*(ipam[5]-ipam[4]+1+2*LenFD) + idy;

								Az.Txx[Pidx] = tAz.Txx[Pidx] + beta*hAz.Txx[Pidx];
								Az.Tyy[Pidx] = tAz.Tyy[Pidx] + beta*hAz.Tyy[Pidx];
								Az.Tzz[Pidx] = tAz.Tzz[Pidx] + beta*hAz.Tzz[Pidx];
								Az.Txy[Pidx] = tAz.Txy[Pidx] + beta*hAz.Txy[Pidx];
								Az.Txz[Pidx] = tAz.Txz[Pidx] + beta*hAz.Txz[Pidx];
								Az.Tyz[Pidx] = tAz.Tyz[Pidx] + beta*hAz.Tyz[Pidx];
								Az.Vx[Pidx] = tAz.Vx[Pidx] + beta*hAz.Vx[Pidx];
								Az.Vy[Pidx] = tAz.Vy[Pidx] + beta*hAz.Vy[Pidx];
								Az.Vz[Pidx] = tAz.Vz[Pidx] + beta*hAz.Vz[Pidx];
							}
#endif


						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

__global__ void ErrorSta(wfield W, int *flag)
{
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index


	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<ipam[8]+LenFD)//vaild point with one virtual bounds
						{
							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;


							if( W.Txx[Gindex]!=0 && ABS( W.Txx[Gindex] ) < 1E-10 ) 
							{
								atomicAdd_system(flag,1);
								printf("\t\tat PCS[%d]DEV[%d] RELpoint(%d,%d,%d) ABSpoint(%d,%d,%d),W.Txx[Gindex]=%e\n",
									ipam[1],ipam[0], idx,idy,idz, 
									idx+(ipam[2]-LenFD)+ipam[9], idy+(ipam[4]-LenFD), idz,
									W.Txx[Gindex]);	
							}

							if( W.Tyy[Gindex]!=0 && ABS( W.Tyy[Gindex] ) < 1E-10 ) 
							{
								atomicAdd_system(flag,1);
								printf("\t\tat PCS[%d]DEV[%d] RELpoint(%d,%d,%d) ABSpoint(%d,%d,%d),W.Tyy[Gindex]=%e\n",
									ipam[1],ipam[0], idx,idy,idz, 
									idx+(ipam[2]-LenFD)+ipam[9], idy+(ipam[4]-LenFD), idz,
									W.Tyy[Gindex]);	
							}

							if( W.Tzz[Gindex]!=0 && ABS( W.Tzz[Gindex] ) < 1E-10 ) 
							{
								atomicAdd_system(flag,1);
								printf("\t\tat PCS[%d]DEV[%d] RELpoint(%d,%d,%d) ABSpoint(%d,%d,%d),W.Tzz[Gindex]=%e\n",
									ipam[1],ipam[0], idx,idy,idz, 
									idx+(ipam[2]-LenFD)+ipam[9], idy+(ipam[4]-LenFD), idz,
									W.Txx[Gindex]);	
							}

							if( W.Txy[Gindex]!=0 && ABS( W.Txy[Gindex] ) < 1E-10 ) 
							{
								atomicAdd_system(flag,1);
								printf("\t\tat PCS[%d]DEV[%d] RELpoint(%d,%d,%d) ABSpoint(%d,%d,%d),W.Txy[Gindex]=%e\n",
									ipam[1],ipam[0], idx,idy,idz, 
									idx+(ipam[2]-LenFD)+ipam[9], idy+(ipam[4]-LenFD), idz,
									W.Txy[Gindex]);	
							}

							if( W.Txz[Gindex]!=0 && ABS( W.Txz[Gindex] ) < 1E-10 ) 
							{
								atomicAdd_system(flag,1);
								printf("\t\tat PCS[%d]DEV[%d] RELpoint(%d,%d,%d) ABSpoint(%d,%d,%d),W.Txz[Gindex]=%e\n",
									ipam[1],ipam[0], idx,idy,idz, 
									idx+(ipam[2]-LenFD)+ipam[9], idy+(ipam[4]-LenFD), idz,
									W.Txz[Gindex]);	
							}

							if( W.Tyz[Gindex]!=0 && ABS( W.Tyz[Gindex] ) < 1E-10 ) 
							{
								atomicAdd_system(flag,1);
								printf("\t\tat PCS[%d]DEV[%d] RELpoint(%d,%d,%d) ABSpoint(%d,%d,%d),W.Tyz[Gindex]=%e\n",
									ipam[1],ipam[0], idx,idy,idz, 
									idx+(ipam[2]-LenFD)+ipam[9], idy+(ipam[4]-LenFD), idz,
									W.Tyz[Gindex]);	
							}

							if( W.Vx[Gindex]!=0 && ABS( W.Vx[Gindex] ) < 1E-15 ) 
							{
								atomicAdd_system(flag,1);
								printf("\t\tat PCS[%d]DEV[%d] RELpoint(%d,%d,%d) ABSpoint(%d,%d,%d),W.Vx[Gindex]=%e\n",
									ipam[1],ipam[0], idx,idy,idz, 
									idx+(ipam[2]-LenFD)+ipam[9], idy+(ipam[4]-LenFD), idz,
									W.Vx[Gindex]);	
							}

							if( W.Vy[Gindex]!=0 && ABS( W.Vy[Gindex] ) < 1E-15 ) 
							{
								atomicAdd_system(flag,1);
								printf("\t\tat PCS[%d]DEV[%d] RELpoint(%d,%d,%d) ABSpoint(%d,%d,%d),W.Vy[Gindex]=%e\n",
									ipam[1],ipam[0], idx,idy,idz, 
									idx+(ipam[2]-LenFD)+ipam[9], idy+(ipam[4]-LenFD), idz,
									W.Vy[Gindex]);	
							}

							if( W.Vz[Gindex]!=0 && ABS( W.Vz[Gindex] ) < 1E-15 ) 
							{
								atomicAdd_system(flag,1);
								printf("\t\tat PCS[%d]DEV[%d] RELpoint(%d,%d,%d) ABSpoint(%d,%d,%d),W.Vz[Gindex]=%e\n",
									ipam[1],ipam[0], idx,idy,idz, 
									idx+(ipam[2]-LenFD)+ipam[9], idy+(ipam[4]-LenFD), idz,
									W.Vz[Gindex]);	
							}

						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}


__global__ void AbsExp(Real *Ex, Real *Ey, Real *Ez, int *nabs, wfield W)
{
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point indexa
	Real D=1.0;

	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<ipam[8]+LenFD)//vaild point with one virtual bounds
						{
							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
							
							//X1 && X2
							//if(idx+Cstart<=AbsLoc[0*6+1] || idx+Cstart>=AbsLoc[1*6+0])
							if(idx+(ipam[2]-LenFD)+ipam[9]<=nabs[0]+LenFD-1 || idx+(ipam[2]-LenFD)+ipam[9]>=ipam[10]+LenFD-nabs[1])
							{
								D = MIN( MIN(Ex[idx],Ey[idy]), Ez[idz]);

								W.Txx[Gindex] = D*W.Txx[Gindex];
								W.Tyy[Gindex] = D*W.Tyy[Gindex];
								W.Tzz[Gindex] = D*W.Tzz[Gindex];
								W.Txy[Gindex] = D*W.Txy[Gindex];
								W.Txz[Gindex] = D*W.Txz[Gindex];
								W.Tyz[Gindex] = D*W.Tyz[Gindex];
								W.Vx[Gindex]  = D*W.Vx[Gindex];
								W.Vy[Gindex]  = D*W.Vy[Gindex];
								W.Vz[Gindex]  = D*W.Vz[Gindex];
							}
							else
							{
								//Y1 && Y2
								//if(idy<=AbsLoc[2*6+3] || idy>=AbsLoc[3*6+2] )
								if(idy+(ipam[4]-LenFD)<=nabs[2]+LenFD-1 || idy+(ipam[4]-LenFD)>=ipam[7]+LenFD-nabs[3])
								{
									D = MIN( MIN(Ex[idx],Ey[idy]), Ez[idz]);

									W.Txx[Gindex] = D*W.Txx[Gindex];
									W.Tyy[Gindex] = D*W.Tyy[Gindex];
									W.Tzz[Gindex] = D*W.Tzz[Gindex];
									W.Txy[Gindex] = D*W.Txy[Gindex];
									W.Txz[Gindex] = D*W.Txz[Gindex];
									W.Tyz[Gindex] = D*W.Tyz[Gindex];
									W.Vx[Gindex]  = D*W.Vx[Gindex];
									W.Vy[Gindex]  = D*W.Vy[Gindex];
									W.Vz[Gindex]  = D*W.Vz[Gindex];
								}
								else
								{
									//Z1 && Z2
									//if(idz<=AbsLoc[4*6+5] || idz>=AbsLoc[5*6+4] )
									if(idz<=nabs[4]+LenFD-1 || idz>=ipam[8]+LenFD-nabs[5])
									{
										D = MIN( MIN(Ex[idx],Ey[idy]), Ez[idz]);

										W.Txx[Gindex] = D*W.Txx[Gindex];
										W.Tyy[Gindex] = D*W.Tyy[Gindex];
										W.Tzz[Gindex] = D*W.Tzz[Gindex];
										W.Txy[Gindex] = D*W.Txy[Gindex];
										W.Txz[Gindex] = D*W.Txz[Gindex];
										W.Tyz[Gindex] = D*W.Tyz[Gindex];
										W.Vx[Gindex]  = D*W.Vx[Gindex];
										W.Vy[Gindex]  = D*W.Vy[Gindex];
										W.Vz[Gindex]  = D*W.Vz[Gindex];
									}//end Z1
								}//end Y1 
							}//end X1


						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}


__global__ void WavefieldPick(wfield W, wfield DPW, PointIndexBufferF Dpt, int currT, int np, int nt)
{
	int i;
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int dst;
	int modX,modY;
	
	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<ipam[8]+LenFD)//vaild point with one virtual bounds
						{
							//idx = idx + Cstart;//wrong, it could chang the absolute access index
							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
							modX = idx+(ipam[2]-LenFD)+ipam[9];//modify to absolute index
							modY = idy+(ipam[4]-LenFD);
							
							for(i=0;i<np;i++)
							{
								dst = Dpt.Rsn[i]*nt+currT;
								
								if( modX == Dpt.locx[i] && modY == Dpt.locy[i] && idz == Dpt.locz[i] )
								{
									
									DPW.Vx[dst] = W.Vx[Gindex];
									DPW.Vy[dst] = W.Vy[Gindex];
									DPW.Vz[dst] = W.Vz[Gindex];
									DPW.Txx[dst] = W.Txx[Gindex];
									DPW.Tyy[dst] = W.Tyy[Gindex];
									DPW.Tzz[dst] = W.Tzz[Gindex];
									DPW.Txy[dst] = W.Txy[Gindex];
									DPW.Txz[dst] = W.Txz[Gindex];
									DPW.Tyz[dst] = W.Tyz[Gindex];

								}//restrict source location
							
							}//loop point in device
						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}
		
__global__ void SnapWavefieldPick(wfield W, wfield DSW, SnapIndexBufferF DSpt, int currT, int np, int nTime)
{
	//wave pick by kernel, Abandoned
	int i;
	int countX,countY,countZ;
	int idx,idy,idz;
	int Gindex;//valid physical point index
	int dst;
	int modX,modY;
	
	for(countX=0; countX<=ipam[3]-ipam[2]; countX+=gridDim.y)//loop in current compute range with step of Launch Par
	{
		idx = countX + blockIdx.y + LenFD;
		if(idx<=ipam[3]-ipam[2]+LenFD)//restrict to last index
		{
			for(countY=0; countY<=ipam[5]-ipam[4]; countY+=gridDim.x)
			{
				idy = countY + blockIdx.x + LenFD;
				if(idy<=ipam[5]-ipam[4]+LenFD)
				{
					for(countZ=0; countZ<ipam[8]; countZ+=blockDim.x)
					{
						idz = countZ + threadIdx.x + LenFD;
						if(idz<ipam[8]+LenFD)//vaild point with one virtual bounds
						{
							//idx = idx + Cstart;//wrong, it could chang the absolute access index
							Gindex = idx*(ipam[5]-ipam[4]+1+2*LenFD)*(ipam[8]+2*LenFD) + idy*(ipam[8]+2*LenFD) + idz;
							modX = idx+(ipam[2]-LenFD)+ipam[9];//modify to absolute index
							modY = idy+(ipam[4]-LenFD);
							
							for(i=0;i<np;i++)
							{
								dst = DSpt.Rsn[i]*nTime+currT;
								
								if( modX == DSpt.locx[i] && modY == DSpt.locy[i] && idz == DSpt.locz[i] )
								{
									//printf("at PCS=%d DEV=%d,Rsn=%d,Gsn=%d,(%d,%d,%d)\n",ipam[1],ipam[0],
									//	DSpt.Rsn[i],DSpt.Gsn[i],DSpt.locx[i],DSpt.locy[i],DSpt.locz[i]);
									if(DSpt.cmp==1 || DSpt.cmp==3)
									{
										DSW.Vx[dst] = W.Vx[Gindex];
										DSW.Vy[dst] = W.Vy[Gindex];
										DSW.Vz[dst] = W.Vz[Gindex];
									}
									if(DSpt.cmp==2 || DSpt.cmp==3)
									{
										DSW.Txx[dst] = W.Txx[Gindex];
										DSW.Tyy[dst] = W.Tyy[Gindex];
										DSW.Tzz[dst] = W.Tzz[Gindex];
										DSW.Txy[dst] = W.Txy[Gindex];
										DSW.Txz[dst] = W.Txz[Gindex];
										DSW.Tyz[dst] = W.Tyz[Gindex];
									}
									
								}//restrict source location
								
							}//loop point in device

						}//restrict idz;
					}//loop countZ
				}//restrict idy
			}//loop countY
		}//restrict idx
	}//loop countX

}

