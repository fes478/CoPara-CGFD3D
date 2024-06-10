//concentional
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<iostream>
#include<netcdf.h>

//check
#define zbx (103)
#define zby (103)
#define zbz (187)

//double or float
#ifndef TypeDouble
#define Real float
#define Intg int
#define SeisNCType NC_FLOAT
#define NFillType NF4FLOAT
#define CONST(a) a##F
#define Rformat "%f"
#define MpiType MPI_FLOAT
#else
#define Real double
#define Intg long int
#define SeisNCType NC_DOUBLE
#define NFillType NF4DOUBLE
#define CONST(a) a
#define Rformat "%lf"
#define MpiType MPI_DOUBLE
#endif

//differential template
#define LenFD 3

//source smooth
#ifdef SrcSmooth
#define LenNorm (2*LenFD+1)
#else
#define LenNorm 0
#endif

//simple math
#define ABS(a) (a>0?a:-(a))
#define MAX(a,b) (a>=b?a:b)
#define MIN(a,b) (a<=b?a:b)
#define SIZE(a) (sizeof(a)/sizeof(a[0]))
#define ISEQSTR(a,b) (!strcmp(a,b))

//coefficients
//Normal use
//one side difference operators (Zhang 2006, equation 10)
#define drp_A_1 (-0.30874)
#define drp_A0 (-0.6326)
#define drp_A1 1.23300
#define drp_A2 (-0.3334)
#define drp_A3 0.04168

#define strF(v,i,dh,inc) ((drp_A_1*v[i-1*inc] + drp_A0*v[i] + drp_A1*v[i+1*inc] + drp_A2*v[i+2*inc] + drp_A3*v[i+3*inc])/dh)
#define strB(v,i,dh,inc) ((drp_A3*v[i+3*inc] + drp_A2*v[i+2*inc] + drp_A1*v[i+1*inc] + drp_A0*v[i] + drp_A_1*v[i-1*inc])/dh)
#define DRPFD(v,i,dh,inc) ( dh>0 ? strF(v,i,dh,inc) : strB(v,i,dh,inc) ) 
//#define DRPFD(v,i,dh,inc) ((drp_A_1*v[i-1*inc] + drp_A0*v[i] + drp_A1*v[i+1*inc] + drp_A2*v[i+2*inc] + drp_A3*v[i+3*inc])/dh)
/*
#define DRPFD(v,i,dh,inc) FD(v,i,dh,inc)
inline Real FD( Real *v, int i, Real dh, int inc)
{
        if(dh>0)//forward
                return (drp_A_1*v[i-1*inc] + drp_A0*v[i] + drp_A1*v[i+1*inc] + drp_A2*v[i+2*inc] + drp_A3*v[i+3*inc])/dh;
        else
                return (drp_A3*v[i+3*inc] + drp_A2*v[i+2*inc] + drp_A1*v[i+1*inc] + drp_A0*v[i] + drp_A_1*v[i-1*inc])/dh;
}
*/


//VLOW use
//Gottlieb 1976 2-4 MacCormack
#define m24_A0 (-1.1666666666666667) //(-7/6)
#define m24_A1 1.3333333333333333    //(8/6)
#define m24_A2 (-0.16666666666666666) //(-1/6)


#define strFm24(v,i,dh,inc) ((m24_A0*v[i] + m24_A1*v[i+1*inc] + m24_A2*v[i+2*inc])/dh)
#define strBm24(v,i,dh,inc) ((m24_A2*v[i+2*inc] + m24_A1*v[i+1*inc] + m24_A0*v[i])/dh)
#define M24FD(v,i,dh,inc) ( dh>0 ? strFm24(v,i,dh,inc) : strBm24(v,i,dh,inc) )
//#define M24FD(v,i,dh,inc) ((m24_A0*v[i] + m24_A1*v[i+1*inc] + m24_A2*v[i+2*inc])/dh)
#define strFm22(v,i,dh,inc) ( (v[i+1*inc]-v[i])/dh )
#define strBm22(v,i,dh,inc) ( -1.0*(v[i]-v[i+1*inc])/dh )
#define M22FD(v,i,dh,inc) ( dh>0 ? strFm22(v,i,dh,inc) : strBm22(v,i,dh,inc) )
//#define M22FD(v,i,dh,inc) ((v[i+1*inc]-v[i])/dh) //true for two point

//Hixon 2000 4-4 compact MacCormack
//forward one
#define cm44_A_1 (-0.25)
#define cm44_A0 (-1.0)
#define cm44_A1 (1.25)
#define cm44_A3 (-0.5)  //represents the [j+1] element in right side

//Velocity free surface condition coefficients
//Hixon & Turkel, 2000.  
//Unilateral compact difference scheme 
#define LA0 (0.6666666666666667) //(2/3)
#define LA1 (0.3333333333333333) //(1/3)
#define RA_1 (-0.1666666666666667) //(-1/6)
#define RA0 (-0.6666666666666667) //(-2/3)
#define RA1 (0.8333333333333333) //(5/6)
#define UCDFD_R(v,i,dh,inc) ((RA_1*v[i-1*inc] + RA0*v[i] + RA1*v[i+1*inc])/LA0/dh)
#define UCDFD_L(v,i,inc) ((LA1*v[i+1*inc])/LA0)

//=========================data struct================================================
namespace constant
{
	const Real PI=3.141592653589793;
	const Real SeisZero = 1.e-20;
	const Real SeisEqual = 1.e-5;
	const Real SeisInf = 1.e+20;

	const int SeisGeo=3;
	const int SeisStrLen=128;
	const int SeisStrLen2=32;
	const int SeisNcNum=10;//V3+T6+time1

	const int Fail2Open=11;
	const int Fail2Close=12;
	const int Fail2Read=13;
	const int Fail2Write=14;
	const int Fail2File=15;

	const int Fail2Check=21;
	const int Fail2Cuda=999;
	const int Fail2Kernel=888;
	const int NoFatalError=0;
}

namespace defstruct
{
	typedef struct
	{
		int nx,ny,nz;
		int ni,nj,nk;
		int nx1,nx2,ny1,ny2,nz1,nz2;
		int ni1,ni2,nj1,nj2,nk1,nk2;
	}cindx;

	typedef struct
	{
		Real ***x,***y,***z;
	}coord;

	typedef struct
	{
		Real ***xi_x,***xi_y,***xi_z;
		Real ***eta_x,***eta_y,***eta_z;
		Real ***zeta_x,***zeta_y,***zeta_z;
		Real ***jac;
	}deriv;

	typedef struct
	{
		Real ***alpha,***beta,***rho;
	}mdpar;

	typedef struct
	{
		bool *interface_flat, **interface_const, *layer_const;
		int ****indexZ;
		Real **layerZrange,**interfaceZrange;
		Real *Xpos,*Ypos,***Zpos;
		Real ****Vp,****Vs,****Dc;
		int nx_in,ny_in,x0_in,y0_in,dx_in,dy_in;
		int nInterface;
	}interface;

	typedef struct
	{
		int ni,nj,nk;
		Real *Xpos, *Ypos, *Zpos;
		Real ***vp, ***vs, ***den;
	}volume;
	
	typedef struct
	{
		int *locx,*locy,*locz;//index
		Real *posx,*posy,*posz;//physical position
		Real *fx,*fy,*fz;
		Real **stf;//source time function
#ifdef SrcSmooth
		Real ****dnorm;//normalization
#endif
	}force;

	typedef struct
	{
		int *locx,*locy,*locz;//index
		Real *posx,*posy,*posz;//physical position
		Real *mxx,*myy,*mzz,*mxy,*mxz,*myz;
		Real **stf;//source time function
#ifdef SrcSmooth
		Real ****dnorm;//normalization
#endif
	}moment;
	
	typedef struct
	{
		int np,nt;
		Real dt;
		int *SN;
		int *locx,*locy,*locz;//index
		Real *posx,*posy,*posz;//physical position
		Real **mxx,**myy,**mzz,**mxy,**mxz,**myz;//STF
#ifdef SrcSmooth
		Real ****dnorm;//normalization
#endif
	}Rmom;

	typedef struct
	{
		int tinv;
		int *SN;
		int *Pnum,*Lnum;
		int *locx,*locy,*locz;//index with bounds
		Real *posx,*posy,*posz;
		char file[constant::SeisStrLen];
		int ncid;
		int vid[constant::SeisNcNum];
	}point;
	
	typedef struct
	{
		int **Rsn,**Gsn;
		int **locx,**locy,**locz;
	}FocalIndexBuffer;

	typedef struct
	{
		int **Rsn,**Gsn;
		int **locx,**locy,**locz;
	}PointIndexBuffer;
	
	typedef struct
	{
		int tinv,cmp;
		int **Rsn,**Gsn;
		int **locx,**locy,**locz;
	}SnapIndexBuffer;

	typedef struct
	{
		int *xs,*xn,*xi;
		int *ys,*yn,*yi;
		int *zs,*zn,*zi;
		int *tinv;
		int *cmp;
		char **file;
		int *ncid;
		int **vid;
		int **SN;
	}snap;
	
	typedef struct
	{
		char file[constant::SeisStrLen];
		int ncid;
		int vid[constant::SeisNcNum];
	}wavebuffer;

	typedef struct
	{
		Real *Vx,*Vy,*Vz;
		Real *Txx,*Tyy,*Tzz;
		Real *Txy,*Txz,*Tyz;
	}wfield;
	
	typedef struct
	{
		Real *Vx,*Vy,*Vz;
	}PeakVel;

	typedef struct
	{
		int *nabs,*ELoc,*CLoc;
		Real *APDx,*APDy,*APDz,*Bx,*By,*Bz,*DBx,*DBy,*DBz;//shifting,scaling,damping factors(ADE CFS PML)
		Real *Ex,*Ey,*Ez;//damping factor(sponge layer)
	}apara;


}

//===============================function class================
class common
{
	public:
		void errorprint(const char*,int,int,const char *);
		void get_conf(FILE*, const char*, int, int*);
		void get_conf(FILE*, const char*, int, long int*);
		void get_conf(FILE*, const char*, int, double*);
		void get_conf(FILE*, const char*, int, float*);
		void get_conf(FILE*, const char*, int, char*);
		void get_conf(FILE*, const char*, int, bool*);
		void setchunk(FILE*, const char*);
		void interpolated_extend(Real***, defstruct::cindx);
		void mirror_extend(Real***, defstruct::cindx);
		void equivalence_extend(Real***, defstruct::cindx);
		void flatten21D(Real***, Real*, defstruct::cindx, int);//flag=0, apply for full size
		void compress23D(Real*, Real***, defstruct::cindx, int);
		void reposition(Real, Real, Real, defstruct::cindx, defstruct::coord, Real, int*, int*, int*, Real*);
	
	private:
		void getConfStr(FILE*, const char*, int, char*);

};

class mathfunc
{
	public:
		int max(int*, int); //max in N length array, Output=value
		float max(float*, int);
		double max(double*, int);
		int min(int*, int); //min in N length array, Output=value
		float min(float*, int);
		double min(double*, int);
		int sum(int*, int); // sum the N length array, Output=value
		float sum(float*, int);
		double sum(double*, int);
		long int accumulation(long int*, int); //accumulate the N length array, Output=value
		long int accumulation(int*, int);
		long int accumulation(size_t*, int);
		double accumulation(float*, int);
		long double accumulation(double*, int);
		int dotproduct(int*, int*, int); //dotproduct of two N length array, Output=value
		float dotproduct(float*,float*, int);
		double dotproduct(double*,double*, int);
		int crossproduct(int*, int*, int*); //crossproduct of two 3 elements array, Output=array
		float crossproduct(float*, float*, float*);
		double crossproduct(double*, double*, double*);
		float vectornorm(float*, int); //norm of the N length vector, Output=value
		double vectornorm(double*, int);
		
		void matmul(Real**,Real**,Real**); //3*3matrix multiply, Output=array
		void matmul(Real[3][3], Real[3][3], Real[3][3]);
		Real determinant(Real**); //determinant of the 3*3 array, Output=value
		Real determinant(Real[3][3]);
		void matinv(Real**); //invert of the 3*3 array, Output=array
		void matinv(Real[3][3]);
		
		long int locmin(Real*, int); //positioning the min location in N length array, Output=value
		long int locmax(Real*, int); //positioning the max location in N length array, Output=value
		void LocValue1d(Real, Real*, int, int*, int*, Real*); //confirm the value and boundary in N length 
		               //array for a given value, Output= 1 nearst value and 2 location
		
		Real interp1d(Real*, Real*, int, Real); // Lagrangian interpolation method, Output=value
		Real interp2d(Real*, Real*, Real**, int, int, Real, Real);
		Real interp3d(Real*, Real*, Real*, Real***, int, int, int, Real, Real, Real);
		
		Real distanceP2L(Real, Real, Real*, Real*); //Output=value
		Real distanceP2S(Real, Real, Real, Real*, Real*, Real*);
		
		Real Gauss(Real, Real, Real);
		Real gauss(Real, Real, Real);
		Real Ricker(Real, Real, Real);
		Real ricker(Real, Real, Real);
		Real Bell(Real, Real);
		Real bell(Real, Real);
		Real BELL(Real, Real);
		Real Triangle(Real, Real);
		Real TRIANGLE(Real, Real);
		Real Bshift(Real, Real, Real);
		Real Step(Real);
		Real Delta(Real, Real);

};

class seisnc
{
	private:
		mathfunc mathf;
		common com;
		int nf4int;
		float nf4float;
		double nf4double;
		long int getVarSize(const char *, int, const char *, int);
		long int getVarSize(const char *, int, const char *, int, size_t *);
	public:
		seisnc();
		void errorprint(const char *, int, const char *, const char *, int, int);
		void errorprint(const char *, int, const char *, const char *, const char *, int, int);
		size_t dimsize(const char*, const char*);
		void attrget(const char*, const char*, int*);
		void attrget(const char*, const char*, float*);
		void attrget(const char*, const char*, double*);
		void varget(const char*, const char*, int*);
		void varget(const char*, const char*, float*);
		void varget(const char*, const char*, double*);
		void varget(const char*, const char*, int*, size_t*, size_t*, ptrdiff_t*);
		void varget(const char*, const char*, float*, size_t*, size_t*, ptrdiff_t*);
		void varget(const char*, const char*, double*, size_t*, size_t*, ptrdiff_t*);
		void varput(const char*, const char*, int*);
		void varput(const char*, const char*, float*);
		void varput(const char*, const char*, double*);
		void varput(const char*, const char*, int*, size_t*, size_t*, ptrdiff_t*, bool);
		void varput(const char*, const char*, float*, size_t*, size_t*, ptrdiff_t*, bool);
		void varput(const char*, const char*, double*, size_t*, size_t*, ptrdiff_t*, bool);
		void grid_export(defstruct::coord, defstruct::deriv, defstruct::cindx, const char*, const char*);
		void grid_import(defstruct::coord, defstruct::deriv, defstruct::cindx, const char*, const char*);
		void media_export(defstruct::mdpar, defstruct::cindx, defstruct::coord, const char*);
		void media_import(defstruct::mdpar, defstruct::cindx, defstruct::coord, const char*);
		void force_export(defstruct::force, const char*, int, int);
		void force_import(defstruct::force, const char*, int, int);
		void moment_export(defstruct::moment, const char*, int, int);
		void moment_import(defstruct::moment, const char*, int, int);
		void focal_export(defstruct::Rmom, const char*);
		void focal_import(defstruct::Rmom, const char*);
		void point_export_def(int, defstruct::point*);
		void point_export(int, Real, int, defstruct::point, defstruct::wfield);
		void point_import(defstruct::point*);
		void point_export_end(defstruct::point);
		void snap_export_def(int, defstruct::snap);
		void snap_export(int, Real, int, defstruct::snap, defstruct::cindx, defstruct::wfield);//full size
		void snap_export(int, int, Real, int, defstruct::snap, Real**);//valid size
		void snap_import(int, defstruct::snap);
		void snap_export_end(int, defstruct::snap);
		void wavebuffer_def(defstruct::wavebuffer*, defstruct::cindx);
		void wavebuffer_export(int, Real, defstruct::wavebuffer, defstruct::cindx, defstruct::wfield);
		void wavebuffer_import(int*, Real, defstruct::wavebuffer*, defstruct::cindx, defstruct::wfield);
		void wavebuffer_end(defstruct::wavebuffer);
		void PV_export(defstruct::PeakVel, const char*, defstruct::cindx);

};

class gridmesh
{
	private:
		common com;
		mathfunc mathf;
		seisnc snc;
		bool Rwork,Mflag;
		int myid;
		int Dnx,Dny,Dnz;
		Real ConversDepth;
		char gridtype[constant::SeisStrLen2], gridfile[constant::SeisStrLen];
		void getConf(const char*);
		void read_vmap(defstruct::cindx, const char*, Real);
		void read_vmapnew(defstruct::cindx, const char*, Real);
		void read_point(defstruct::cindx);
		void smooth_zdepth(Real*, Real, Real, Real, Real, int);
		void set_twoside_space(Real, int, int, Real, Real, Real, Real, Real*, Real*);
		void GridCheck(defstruct::cindx, Real steph);

	public:
		defstruct::coord crd;
		defstruct::deriv drv;
		gridmesh(const char*, defstruct::cindx, const int, const int);
		~gridmesh();
		int ConIndex,HyGrid;
		void readdata(defstruct::cindx, const char*, Real);
		void calderivative(Real, defstruct::cindx);//6 point
		void calmetric(Real, defstruct::cindx);//2 point
		void export_data(const char*, defstruct::cindx);

};

class mediapar
{
	private:
		common com;
		mathfunc mathf;
		seisnc snc;
		bool Rwork,Mflag;
		int myid;
		int Dnx,Dny,Dnz;
		int sampx,sampy,sampz;
		Real crit_value, crit_perc;
		
		char mediatype[constant::SeisStrLen2], mediafile[constant::SeisStrLen];
		void getConf(const char*);
		void read_interface(defstruct::cindx, defstruct::coord);
		void read_interface3D(defstruct::cindx, defstruct::coord);
		void read_volume_new(defstruct::cindx, defstruct::coord);//read only
		void read_volume_old(defstruct::cindx, defstruct::coord);//read and interpolate
		void interface3d_discrete(Real, Real, Real, Real*, Real*, Real*);
		void volume_discrete(Real, Real, Real, Real*, Real*, Real*);
		void TwoinOne(Real*, Real*, Real*, Real*, Real*, Real*, int*);
		void average(int, int, int, Real, Real, Real, int);
		void get2in1(int, int, int, Real*, Real*, Real*);
		void set_vel3d(int, int, int, Real*);
		void ApplySmooth(defstruct::cindx, defstruct::coord);
		bool OverLimits(Real, Real);
		void MediaStatistics(defstruct::cindx);
	public:
		defstruct::mdpar mpa;
		defstruct::interface i3d;
		defstruct::volume Vnc;
		Real ***lambda, ***miu, ***density;//temp transfer use
		Real ***vel3d;//check use
		mediapar(const char*, defstruct::cindx, const int, const int);
		~mediapar();
		void readdata(defstruct::cindx, defstruct::coord);
		void timecheck(Real, defstruct::cindx, defstruct::coord);
		void export_data(const char*, defstruct::cindx, defstruct::coord);

};

class source
{
	private:
		common com;
		mathfunc mathf;
		seisnc snc;
		char momtype[constant::SeisStrLen2];
		bool Rwork,Mflag;
		int myid;
		Real dst2m,hyp2g;
		char srcfile[constant::SeisStrLen];
		char fNCfile[constant::SeisStrLen];
		void getConf(const char*);
		void cal_stf_force(Real, const char*, Real, Real, int, Real, Real*);
		void cal_stf_moment(Real, const char*, Real, Real, int, Real, Real*);
		void angle2moment(Real, Real, Real, Real*, Real*, Real*, Real*, Real*, Real*); 
		void eighth_locate(Real, Real, Real, Real***, Real***, Real***, int*, int*, int*);
		void MomentShiftPosition(defstruct::coord, int*, int*, int*, Real, Real, Real, Real*, Real*, Real*); 
		void cal_norm(Real, Real, Real, int, int, Real***);

	public:
		int nfrc,nmnt,nstf;
		int CPN;
		
		defstruct::force frc;
		defstruct::moment mnt;
		defstruct::Rmom Rmnt;
		defstruct::FocalIndexBuffer Fpt;//point buffer parameters-Var 
		
		source(const char*, int, const int, const int, int);
		~source();
		void readdata(Real);
		void cal_index(defstruct::cindx, defstruct::coord, defstruct::mdpar);
		void BoundJug(defstruct::cindx, defstruct::coord, int*);
		void export_data(const char*, defstruct::force, defstruct::moment);
		void M2CPointPick(defstruct::cindx, int*, int*, int*, int);

};

class seisplot
{
	private:
		common com;
		mathfunc mathf;
		seisnc snc;
		bool Rwork,Mflag;
		int myid;
		Real hyp2g;
		char sptfile[constant::SeisStrLen];
		void getConf(const char*);

	public:
		int CPN,nt;
		int npnt,nrecv,nline,nsnap;
		int FWSI;//full wave field storage interval
		int PVflag;//output peak velocity
		
		int *Snp;//snapshot point number [CPN*nsnap]
		defstruct::point pnt;//point and line type parameters-Var 
		defstruct::PointIndexBuffer Hpt;//point buffer parameters-Var 
		defstruct::snap snp;//snap type parameters-Var
		defstruct::SnapIndexBuffer *HSpt;//snap buffer parameters-Var 
		
		defstruct::wavebuffer wbuffer;//full wave field buffer parameter-Var
		defstruct::wfield wpoint;//point and line wave field buffer data
		defstruct::wfield MPW, *MSW;//point data buffer, snap data buffer(valid size)
		defstruct::wfield wsnap;//snap wave field buffer data(fullsize)
		defstruct::PeakVel pv;
		
		seisplot(const char*, defstruct::cindx, const int, const int, int, int);
		~seisplot();
		void readdata(const char*);
		void locpoint(defstruct::cindx, defstruct::coord);
		void M2CPointPick(defstruct::cindx, int*, int*, int*, int);
		void M2CSnapPick(defstruct::cindx, int*, int*, int);
		void point_extract(defstruct::point, defstruct::cindx, defstruct::wfield, defstruct::wfield);//extract point from fullwave field
		void data_export_def(int*, Real, defstruct::cindx, defstruct::point*, defstruct::snap, defstruct::wavebuffer*, defstruct::wfield);
		void data_export(int, int, Real, defstruct::cindx, defstruct::point, defstruct::snap, defstruct::wavebuffer, defstruct::wfield, defstruct::wfield);
		void data_export_end(defstruct::point, defstruct::snap, defstruct::wavebuffer);
		void point_export(int, int, Real); 
		void snap_export(int, int, Real); 
		void point_extract(int, defstruct::wfield, defstruct::wfield);//extract point from point wave buffer
		void export_pv(const char*, defstruct::cindx);

};

class absorb
{
	private:
		common com;
		mathfunc mathf;
		int numblk,pmlblk;
		bool Rwork,Mflag;
		int myid;
		Real *CFS_bmax,*CFS_amax;
		char absfile[constant::SeisStrLen];
		void getConf(const char*);
		void GridCovariant(defstruct::deriv, int, int, int, int, Real*);
		Real CalExp(int, Real, Real, int, Real);

	public:
		int *nabs;
		Real *velabs;
		int *ELoc,*CLoc;//damping scale(include bounds)
		defstruct::apara apr;//parameters
		absorb(const char*, defstruct::cindx, const int, const int);
		~absorb();
		void CalSLDamping(defstruct::cindx, defstruct::deriv, Real, Real);
		void CalCFSfactors(defstruct::cindx, defstruct::deriv, Real, Real);

};

void generatewave(int*, int, defstruct::wfield, defstruct::wfield, defstruct::cindx);
void MpiErrorPrint(const char*, int, int, const char*);


















