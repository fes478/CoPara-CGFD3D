#include "typenew.h"
#include<math.h>
#include<typeinfo>

using namespace std;
using namespace defstruct;
using namespace constant;

#define errprt(...) com.errorprint(__FILE__, __LINE__, __VA_ARGS__)

//-------------------private--------------------------------
void source::getConf(const char *filename)
{
	char parpath[SeisStrLen];
	char name[SeisStrLen2];
	char errstr[SeisStrLen];
	FILE *fp;

	fp = fopen(filename,"r");
	if(!fp)
	{
		sprintf(errstr,"Fail to open main par file %s in source.cpp", filename);
		errprt(Fail2Open,errstr);
	}
	com.get_conf(fp, "seispath", 3, parpath);
	com.get_conf(fp, "source_filename", 3, name);
	fclose(fp);
	
	sprintf(srcfile,"%s/%s",parpath,name);

	fp = fopen(srcfile,"r");
	if(!fp)
	{
		sprintf(errstr,"Fail to open source configure file %s in source.cpp", srcfile);
		errprt(Fail2Open,errstr);
	}
	
	//init
	memset(name, '\0', SeisStrLen2);
	nfrc=0;	nmnt=0;	Rmnt.np=0; Rmnt.nt=0; Rmnt.dt=0.0;//par init

	int Stype;
	com.get_conf(fp,"source_type",3,&Stype);
	if( Stype==1 )
	{
		//com.get_conf(fp,"focal_point_number",3,&Rmnt.np);
		//com.get_conf(fp,"focal_time_number",3,&Rmnt.nt);
		//com.get_conf(fp,"focal_time_interval",3,&Rmnt.dt);
		com.get_conf(fp,"focal_process_NC_file",3,name);
		sprintf(fNCfile,"%s/%s",parpath,name);
		size_t dim;
		dim = snc.dimsize(fNCfile, "NT");	Rmnt.nt = dim;
		dim = snc.dimsize(fNCfile, "NP");	Rmnt.np = dim;
		snc.attrget(fNCfile, "DT", &Rmnt.dt);
		fprintf(stdout,"will apply focal process with %d points of %f interval and %d time steps\n",Rmnt.np, Rmnt.dt, Rmnt.nt);

		nfrc = 0;
		nmnt = 0;
	}
	else if( Stype==2 || Stype==3 ) 
	{
		com.get_conf(fp,"number_of_force_source",3,&nfrc);
		com.get_conf(fp,"number_of_moment_source",3,&nmnt);

		Rmnt.np = 0;
	}
	fclose(fp);

	fprintf(stdout,"will apply focus %d, force %d and moment %d\n",Rmnt.np, nfrc, nmnt);
}

void source::cal_stf_force(Real start, const char *name, Real t0, Real f0, int lengthtime, Real dt, Real *stf)
{
	Real t;
	for(int i=0;i<lengthtime;i++)
	{
		t = i*dt - start;
		if(ISEQSTR(name,"Gauss"))
			stf[i] = mathf.Gauss(t,f0,t0);
		else if(ISEQSTR(name,"gauss"))
			stf[i] = mathf.gauss(t,f0,t0);
		else if(ISEQSTR(name,"Ricker"))
			stf[i] = mathf.Ricker(t,f0,t0);
		else if(ISEQSTR(name,"ricker"))
			stf[i] = mathf.ricker(t,f0,t0);
		else if(ISEQSTR(name,"Bell"))
			stf[i] = mathf.Bell(t-t0,f0);
		else if(ISEQSTR(name,"bell"))
			stf[i] = mathf.bell(t-t0,f0);
		else if(ISEQSTR(name,"BELL"))
			stf[i] = mathf.BELL(t-t0,f0);
		else if(ISEQSTR(name,"Step"))
			stf[i] = mathf.Step(t);
		else
		{
			char errstr[SeisStrLen];
			sprintf(errstr,"Wrong type(%s) for source time function FORCE",name);
			errprt(Fail2Check,errstr);
		}
	}
}
void source::cal_stf_moment(Real start, const char *name, Real t0, Real f0, int lengthtime, Real dt, Real *stf)
{
	Real t, s;
	for(int i=0;i<lengthtime;i++)
	{
		t = i*dt - start;
		if(ISEQSTR(name,"Gauss"))
			s = mathf.Gauss(t,f0,t0);
		else if(ISEQSTR(name,"gauss"))
			s = mathf.gauss(t,f0,t0);
		else if(ISEQSTR(name,"Ricker"))
			s = mathf.Ricker(t,f0,t0);
		else if(ISEQSTR(name,"ricker"))
			s = mathf.ricker(t,f0,t0);
		else if(ISEQSTR(name,"Bell"))
			s = mathf.Bell(t-t0,f0);
		else if(ISEQSTR(name,"bell"))
			s = mathf.bell(t-t0,f0);
		else if(ISEQSTR(name,"BELL"))
			s = mathf.BELL(t-t0,f0);
		else if(ISEQSTR(name,"Step"))
			s = mathf.Step(t);
		else
		{
			char errstr[SeisStrLen];
			sprintf(errstr,"Wrong type(%s) for source time function MOMENT",name);
			errprt(Fail2Check,errstr);
		}
		(ABS(s) < SeisZero) ? stf[i] = 0.0 : stf[i] = s;
	}
}

void source::angle2moment(Real strike, Real dip, Real rake, Real *mxx, Real *myy, Real *mzz, Real *mxy, Real *mxz, Real *myz)
{
	//transform angle to moment
	//Modern Global Seismology-Academic press
	//Chapter 8 Section 5---the seismic moment tensor
	//Equation 8.83
	Real S,D,R;//arc system
	Real m11,m22,m33,m12,m13,m23;

	S = strike/180.0*PI;
	D = dip/180.0*PI;
	R = rake/180.0*PI;

	m11 = -( sin(D)*cos(R)*sin(2.0*S) + sin(2.0*D)*sin(R)*sin(S)*sin(S) );
	m22 =    sin(D)*cos(R)*sin(2.0*S) - sin(2.0*D)*sin(R)*cos(S)*cos(S);
	m33 = -( m11+m22 );
	m12 =    sin(D)*cos(R)*cos(2.0*S) + sin(2.0*D)*sin(R)*sin(2.0*S)*0.5;
	m13 = -( cos(D)*cos(R)*cos(S) + cos(2.0*D)*sin(R)*sin(S) );
	m23 = -( cos(D)*cos(R)*sin(S) - cos(2.0*D)*sin(R)*cos(S) );
	*mxx = m11; *myy = m22; *mzz = m33;
	*mxy = m12; *mxz = m13; *myz = m23;
}

void source::MomentShiftPosition(coord crd, int *ix, int *iy, int *iz, Real px, Real py, Real pz, Real *shx, Real *shy, Real *shz)
{
	//on X and Y direction is one-point smoothing on both sides
	//while on Z direction is three-point smoothing on both side.
	//A cuboid smooth.
	int si,sj,sk;
	int i,j,k;
	int nx,ny,nz,ns;
	int mi,mj,mk, mi0,mj0,mk0;
	int i1,i2,j1,j2,k1,k2;
	Real p,dist;
	Real ***x,***y,***z;

	nx=1; ny=1; nz=LenFD; ns=8;

	Real mx[2*nx*ns][2*ny*ns][2*nz*ns];
	Real my[2*nx*ns][2*ny*ns][2*nz*ns];
	Real mz[2*nx*ns][2*ny*ns][2*nz*ns];

	x = new Real **[2];
	y = new Real **[2];
	z = new Real **[2];
	for(i=0;i<2;i++)
	{
		x[i] = new Real *[2];
		y[i] = new Real *[2];
		z[i] = new Real *[2];
		for(j=0;j<2;j++)
		{
			x[i][j] = new Real [2];
			y[i][j] = new Real [2];
			z[i][j] = new Real [2];
		}
	}

	Real infun[2]={0.0,1.0};

	for(si=-nx;si<=nx-1;si++)
		for(sj=-ny;sj<=ny-1;sj++)
			for(sk=-nz;sk<=nz-1;sk++)
			{
				mi0 = (si+nx)*ns+1;
				mj0 = (sj+ny)*ns+1;
				mk0 = (sk+nz)*ns+1;

				i1 = *ix+si;
				j1 = *iy+sj;
				k1 = *iz+sk;

				for(i=i1,i2=0;i<=i1+1,i2<=1;i++,i2++)
					for(j=j1,j2=0;j<=j1+1,j2<=1;j++,j2++)
						for(k=k1,k2=0;k<=k1+1,k2<=1;k++,k2++)
						{
							x[i2][j2][k2] = crd.x[i][j][k];
							y[i2][j2][k2] = crd.y[i][j][k];
							z[i2][j2][k2] = crd.z[i][j][k];
						}

				for(mi=0;mi<ns;mi++)
					for(mj=0;mj<ns;mj++)
						for(mk=0;mk<ns;mk++)
						{
							i = mi+mi0-1;
							j = mj+mj0-1;
							k = mk+mk0-1;
							mx[i][j][k] = mathf.interp3d(infun,infun,infun, x, 2,2,2, (Real)mi/ns, (Real)mj/ns, (Real)mk/ns);
							my[i][j][k] = mathf.interp3d(infun,infun,infun, y, 2,2,2, (Real)mi/ns, (Real)mj/ns, (Real)mk/ns);
							mz[i][j][k] = mathf.interp3d(infun,infun,infun, z, 2,2,2, (Real)mi/ns, (Real)mj/ns, (Real)mk/ns);
						}

			}

	dist = SeisInf;
	p = -1.0*SeisInf;
	for(i=0;i<2*ns*nx;i++)
		for(j=0;j<2*ns*ny;j++)
			for(k=0;k<2*ns*nz;k++)
			{
				p = (mx[i][j][k] - px)*(mx[i][j][k] - px)+
				    (my[i][j][k] - py)*(my[i][j][k] - py)+
				    (mz[i][j][k] - pz)*(mz[i][j][k] - pz);
				if(p < dist)
				{
					dist = p;
					i1 = i; 
					j1 = j;
					k1 = k;
				}
			}

	si = (i1+ns/2)/ns;
	sj = (j1+ns/2)/ns;
	sk = (k1+ns/2)/ns;
	
	*ix = *ix+si-nx;
	*iy = *iy+sj-ny;
	*iz = *iz+sk-nz;

	*shx = (i1-si*ns)*1.0/ns;
	*shy = (j1-sj*ns)*1.0/ns;
	*shz = (k1-sk*ns)*1.0/ns;

	for(i=0;i<2;i++)
	{
		for(j=0;j<2;j++)
		{
			delete [] x[i][j];
			delete [] y[i][j];
			delete [] z[i][j];
		}
		delete [] x[i];
		delete [] y[i];
		delete [] z[i];
	}
	delete [] x;
	delete [] y;
	delete [] z;

}

void source::eighth_locate(Real px, Real py, Real pz, Real***matx, Real ***maty, Real ***matz, int *si, int *sj, int *sk)
{
	Real X[5][5][5]{},Y[5][5][5]{},Z[5][5][5]{};
	bool flag[5][5][5];
	int i,j,k,i1,j1,k1;
	Real infun[3]={0.0,2.0,4.0};
	Real dist,p;
	
	for(i=0;i<5;i++)
		for(j=0;j<5;j++)
			for(k=0;k<5;k++)
				flag[i][j][k] = false;

	for(i=0;i<3;i++)
		for(j=0;j<3;j++)
			for(k=0;k<3;k++)
			{
				X[i*2][j*2][k*2] = matx[i][j][k];
				Y[i*2][j*2][k*2] = maty[i][j][k];
				Z[i*2][j*2][k*2] = matz[i][j][k];
				flag[i*2][j*2][k*2] = true;
			}

	for(i=0;i<5;i++)
		for(j=0;j<5;j++)
			for(k=0;k<5;k++)
			{
				if(flag[i][j][k])
					continue;
				X[i][j][k] = mathf.interp3d(infun,infun,infun,matx,3,3,3,(Real)i,(Real)j,(Real)k);
				Y[i][j][k] = mathf.interp3d(infun,infun,infun,maty,3,3,3,(Real)i,(Real)j,(Real)k);
				Z[i][j][k] = mathf.interp3d(infun,infun,infun,matz,3,3,3,(Real)i,(Real)j,(Real)k);
			}
	
	dist = SeisInf;
	p = -1.0*SeisInf;
	for(i=0;i<5;i++)
		for(j=0;j<5;j++)
			for(k=0;k<5;k++)
			{
				p = (X[i][j][k] - px)*(X[i][j][k] - px)+
				    (Y[i][j][k] - py)*(Y[i][j][k] - py)+
				    (Z[i][j][k] - pz)*(Z[i][j][k] - pz);
				if(p < dist)
				{
					dist = p;
					i1 = i; 
					j1 = j;
					k1 = k;
				}
			}
	
	*si = i1 - 2;
	*sj = j1 - 2;
	*sk = k1 - 2;

	for(i=0;i<3;i++)
		for(j=0;j<3;j++)
			for(k=0;k<3;k++)
			{
				matx[i][j][k] = X[i1+i-1][j1+j-1][k1+k-1];
				maty[i][j][k] = Y[i1+i-1][j1+j-1][k1+k-1];
				matz[i][j][k] = Z[i1+i-1][j1+j-1][k1+k-1];
			}


};

void source::cal_norm(Real shx, Real shy, Real shz, int iz, int nk2, Real ***delta)
{
	int i,j,k;
	Real d1,d2,d3;
	Real dsum;
	dsum = 0.0;
	
	for(i=-LenFD;i<=LenFD;i++)
		for(j=-LenFD;j<=LenFD;j++)
			for(k=-LenFD;k<=LenFD;k++)
			{
				d1 = mathf.Gauss(i-shx, LenFD/2.0, 0.0);
				d2 = mathf.Gauss(j-shy, LenFD/2.0, 0.0);
				d3 = mathf.Gauss(k-shz, LenFD/2.0, 0.0);
				delta[i+LenFD][j+LenFD][k+LenFD] = d1*d2*d3;
				dsum += d1*d2*d3; 
			}
	
	if(dsum < SeisZero)
		errprt(Fail2Check,"NormDelta is zero when smooth source in source.cpp");
	else
		for(i=0;i<LenNorm;i++)
			for(j=0;j<LenNorm;j++)
				for(k=0;k<LenNorm;k++)
					delta[i][j][k] /= dsum;

	dsum = 0.0;
	if(iz+LenFD>nk2-1)
	{
		for(i=0;i<LenNorm;i++)
			for(j=0;j<LenNorm;j++)
				for(k=0;k<nk2-iz+LenFD;k++)//nk2-1 is surface
					dsum += delta[i][j][k];
		for(i=0;i<LenNorm;i++)
			for(j=0;j<LenNorm;j++)
				for(k=0;k<LenNorm;k++)
					delta[i][j][k] /= dsum;
	}

};


//---------------------public-------------------------------------
source::source(const char *filename, int nt, const int restart, const int Myid, int cpn)
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
	
	getConf(filename);

	if(restart==1)
		Rwork = true;
	else
		Rwork = false;
	
	this->CPN = cpn; 
	this->nstf = 2*(nt + 1);

	// explaination
	// for each time step, doing 4-th Runge Kutta time iteration;
	// in every iteration, using time 0, 1/2, 1/2, and 1.
	// so for accurate simulation, should store pervious, half and current time source function.
	// in detail, expand total time step to 2 times ( 2*nt ), 
	//            reduce time stride to 1/2 ( stept/2 ),
	//            the even time point store interger point value,
	//            the odd time point store half point value, which should use twice time in one iteration.

	int i,j,k;

	if(Rmnt.np)
	{
		fprintf(stdout,"***malloc and init focal data buffer\n");
		
		Rmnt.SN = new int[Rmnt.np]();
		for(i=0;i<Rmnt.np;i++)
			Rmnt.SN[i] = i;

		Rmnt.locx = new int[Rmnt.np]();
		Rmnt.locy = new int[Rmnt.np]();
		Rmnt.locz = new int[Rmnt.np]();
		Rmnt.posx = new Real[Rmnt.np]();
		Rmnt.posy = new Real[Rmnt.np]();
		Rmnt.posz = new Real[Rmnt.np]();
		
		Rmnt.mxx = new Real *[Rmnt.np];
		Rmnt.myy = new Real *[Rmnt.np];
		Rmnt.mzz = new Real *[Rmnt.np];
		Rmnt.mxy = new Real *[Rmnt.np];
		Rmnt.mxz = new Real *[Rmnt.np];
		Rmnt.myz = new Real *[Rmnt.np];
		for(i=0; i< Rmnt.np; i++)
		{
			Rmnt.mxx[i] = new Real[Rmnt.nt]();
			Rmnt.myy[i] = new Real[Rmnt.nt]();
			Rmnt.mzz[i] = new Real[Rmnt.nt]();
			Rmnt.mxy[i] = new Real[Rmnt.nt]();
			Rmnt.mxz[i] = new Real[Rmnt.nt]();
			Rmnt.myz[i] = new Real[Rmnt.nt]();
		}
#ifdef SrcSmooth
		Rmnt.dnorm = new Real ***[Rmnt.np];
		for(i=0;i<Rmnt.np;i++)
		{
			Rmnt.dnorm[i] = new Real **[LenNorm];
			for(j=0;j<LenNorm;j++)
			{
				Rmnt.dnorm[i][j] = new Real *[LenNorm];
				for(k=0;k<LenNorm;k++)
					Rmnt.dnorm[i][j][k] = new Real[LenNorm]();
			}
		}
#endif

		//malloc focal index buffer
		Fpt.Rsn = new int*[cpn];
		Fpt.Gsn = new int*[cpn];
		Fpt.locx = new int*[cpn];
		Fpt.locy = new int*[cpn];
		Fpt.locz = new int*[cpn];

	}
	else
		fprintf(stdout,"---There's no focal implemented\n");

	if(nfrc)
	{
		fprintf(stdout,"***malloc and init force data buffer\n");

		frc.locx = new int[nfrc]();
		frc.locy = new int[nfrc]();
		frc.locz = new int[nfrc]();
		frc.posx = new Real[nfrc]();
		frc.posy = new Real[nfrc]();
		frc.posz = new Real[nfrc]();
		frc.fx = new Real[nfrc]();
		frc.fy = new Real[nfrc]();
		frc.fz = new Real[nfrc]();
		frc.stf = new Real *[nfrc];
		for(i =0; i<nfrc; i++)
			frc.stf[i] = new Real[nstf]();
#ifdef SrcSmooth
		frc.dnorm = new Real ***[nfrc];
		for(i=0;i<nfrc;i++)
		{
			frc.dnorm[i] = new Real **[LenNorm];
			for(j=0;j<LenNorm;j++)
			{
				frc.dnorm[i][j] = new Real *[LenNorm];
				for(k=0;k<LenNorm;k++)
					frc.dnorm[i][j][k] = new Real[LenNorm]();
			}
		}
#endif
	}
	else
		fprintf(stdout,"---There's no force implemented\n");
	
	if(nmnt)
	{
		fprintf(stdout,"***malloc and init moment data buffer\n");

		mnt.locx = new int[nmnt]();
		mnt.locy = new int[nmnt]();
		mnt.locz = new int[nmnt]();
		mnt.posx = new Real[nmnt]();
		mnt.posy = new Real[nmnt]();
		mnt.posz = new Real[nmnt]();
		mnt.mxx = new Real[nmnt]();
		mnt.myy = new Real[nmnt]();
		mnt.mzz = new Real[nmnt]();
		mnt.mxy = new Real[nmnt]();
		mnt.mxz = new Real[nmnt]();
		mnt.myz = new Real[nmnt]();
		mnt.stf = new Real *[nmnt];
		for(i =0; i<nmnt; i++)
			mnt.stf[i] = new Real[nstf]();
#ifdef SrcSmooth
		mnt.dnorm = new Real ***[nmnt];
		for(i=0;i<nmnt;i++)
		{
			mnt.dnorm[i] = new Real **[LenNorm];
			for(j=0;j<LenNorm;j++)
			{
				mnt.dnorm[i][j] = new Real *[LenNorm];
				for(k=0;k<LenNorm;k++)
					mnt.dnorm[i][j][k] = new Real[LenNorm]();
			}
		}
#endif
	}
	else
		fprintf(stdout,"---There's no moment implemented\n");

}
source::~source()
{
	fprintf(stdout,"into data free at Procs[%d],in source.cpp\n",myid);
	if(this->Mflag)
	{
		int i,j,k;
		
		if(nmnt)
		{
#ifdef SrcSmooth
			for(i=0;i<nmnt;i++)
			{
				for(j=0;j<LenNorm;j++)
				{
					for(k=0;k<LenNorm;k++)
						delete [] mnt.dnorm[i][j][k];
					delete [] mnt.dnorm[i][j];
				}
				delete [] mnt.dnorm[i];
			}
			delete [] mnt.dnorm;
#endif
			for(i=0;i<nmnt;i++)
				delete [] mnt.stf[i];
			delete [] mnt.stf;
			delete [] mnt.myz;
			delete [] mnt.mxz;
			delete [] mnt.mxy;
			delete [] mnt.mzz;
			delete [] mnt.myy;
			delete [] mnt.mxx;
			delete [] mnt.posz;
			delete [] mnt.posy;
			delete [] mnt.posx;
			delete [] mnt.locz;
			delete [] mnt.locy;
			delete [] mnt.locx;
		}

		if(nfrc)
		{
#ifdef SrcSmooth
			for(i=0;i<nfrc;i++)
			{
				for(j=0;j<LenNorm;j++)
				{
					for(k=0;k<LenNorm;k++)
						delete [] frc.dnorm[i][j][k];
					delete [] frc.dnorm[i][j];
				}
				delete [] frc.dnorm[i];
			}
			delete [] frc.dnorm;
#endif
			for(i=0;i<nfrc;i++)
				delete [] frc.stf[i];
			delete [] frc.stf;
			delete [] frc.fz;
			delete [] frc.fy;
			delete [] frc.fx;
			delete [] frc.posz;
			delete [] frc.posy;
			delete [] frc.posx;
			delete [] frc.locz;
			delete [] frc.locy;
			delete [] frc.locx;
		}
		
		if(Rmnt.np)
		{
			for(i=0;i<CPN;i++)
			{
				delete [] Fpt.locz[i];
				delete [] Fpt.locy[i];
				delete [] Fpt.locx[i];
				delete [] Fpt.Gsn[i];
				delete [] Fpt.Rsn[i];
			}
			delete [] Fpt.locz;
			delete [] Fpt.locy;
			delete [] Fpt.locx;
			delete [] Fpt.Gsn;
			delete [] Fpt.Rsn;


#ifdef SrcSmooth
			for(i=0;i<Rmnt.np;i++)
			{
				for(j=0;j<LenNorm;j++)
				{
					for(k=0;k<LenNorm;k++)
						delete [] Rmnt.dnorm[i][j][k];
					delete [] Rmnt.dnorm[i][j];
				}
				delete [] Rmnt.dnorm[i];
			}
			delete [] Rmnt.dnorm;
#endif
			for(i=0;i<Rmnt.np;i++)
			{
				delete [] Rmnt.myz[i];
				delete [] Rmnt.mxz[i];
				delete [] Rmnt.mxy[i];
				delete [] Rmnt.mzz[i];
				delete [] Rmnt.myy[i];
				delete [] Rmnt.mxx[i];
			}
			delete [] Rmnt.myz;
			delete [] Rmnt.mxz;
			delete [] Rmnt.mxy;
			delete [] Rmnt.mzz;
			delete [] Rmnt.myy;
			delete [] Rmnt.mxx;
			delete [] Rmnt.posz;
			delete [] Rmnt.posy;
			delete [] Rmnt.posx;
			delete [] Rmnt.locz;
			delete [] Rmnt.locy;
			delete [] Rmnt.locx;
			delete [] Rmnt.SN;
		}
	}
	fprintf(stdout,"data free at Procs[%d],in source.cpp\n",myid);
}

void source::readdata(Real stept)
{
	if(this->Rwork)
	{
		fprintf(stdout,"---Reading the exists source time function data, there is no needs to read from ordinary one due to the restart work\n");
		return;
	}

	fprintf(stdout,"***Start to reading source configure files\n");

	int i,j,k;
	char errstr[SeisStrLen];
	Real tempstf[nstf];

	//read configure
	FILE *fp;
	fp = fopen(srcfile,"r");
	if(!fp)
	{
		sprintf(errstr,"Fail to open source configure file %s in source.cpp", srcfile);
		errprt(Fail2Open,errstr);
	}
	com.get_conf(fp,"distance2meter",3,&dst2m);
	com.get_conf(fp,"source_hyper_height",3,&hyp2g);
	hyp2g = hyp2g*dst2m;

	int nwin;
	Real df0,dm0,st0,sf0;
	Real *dst;//dominant force start time
	char stype[SeisStrLen2];
	
	if(nfrc)
	{
		dst = new Real[nfrc];
		i=0; j=0; k=0; nwin=0;

		com.get_conf(fp,"force_stf_window",3,&nwin);
		com.setchunk(fp,"<anchor_force>");
		for(i=0;i<nfrc;i++)
		{
			fscanf(fp,Rformat Rformat Rformat Rformat Rformat Rformat Rformat Rformat,
					&frc.posx[i],&frc.posy[i],&frc.posz[i],&dst[i],&df0,&frc.fx[i],&frc.fy[i],&frc.fz[i]);
			frc.posx[i] *= dst2m; frc.posy[i] *= dst2m; frc.posz[i]*= dst2m;
			frc.fx[i] *= df0; frc.fy[i] *= df0; frc.fz[i] *= df0;
		}

		for(i=0;i<nfrc;i++)
			for(k=0;k<nwin;k++)
			{
				com.get_conf(fp,"force_stf_type",3+k,stype);
				com.get_conf(fp,"force_stf_timefactor",3+k,&st0);
				com.get_conf(fp,"force_stf_freqfactor",3+k,&sf0);
				cal_stf_force(dst[i],stype,st0,sf0,nstf,stept/2.0,tempstf);
				for(j=0;j<nstf;j++)
					frc.stf[i][j] = tempstf[j];
			}
		delete [] dst;
		dst = NULL;
	}
	
	//read moment configuration
	//char momtype[SeisStrLen2];
	Real strike,dip,rake;
	
	nwin=0; st0=0.0; sf0=0.0;//refresh
	memset(stype,'\0',SeisStrLen2*sizeof(char));
	memset(tempstf,'\0',nstf*sizeof(Real));
	
	if(nmnt)
	{
		dst = new Real[nmnt];
		i=0; j=0; k=0;

		com.get_conf(fp,"moment_stf_window",3,&nwin);
		com.get_conf(fp,"moment_mech_input",3,momtype);//angle or moment
		com.setchunk(fp,"<anchor_moment>");
		for(i=0;i<nmnt;i++)
		{
			if(ISEQSTR(momtype,"moment"))
				fscanf(fp,Rformat Rformat Rformat Rformat Rformat Rformat Rformat Rformat Rformat Rformat Rformat,
						&mnt.posx[i],&mnt.posy[i],&mnt.posz[i],&dst[i],&dm0,
						&mnt.mxx[i],&mnt.myy[i],&mnt.mzz[i],&mnt.mxy[i],&mnt.mxz[i],&mnt.myz[i]);
			else if(ISEQSTR(momtype,"angle"))
			{
				fscanf(fp,Rformat Rformat Rformat Rformat Rformat Rformat Rformat Rformat,
						&mnt.posx[i],&mnt.posy[i],&mnt.posz[i],&dst[i],&dm0,&strike,&dip,&rake);
				angle2moment(strike,dip,rake,&mnt.mxx[i],&mnt.myy[i],&mnt.mzz[i],&mnt.mxy[i],&mnt.mxz[i],&mnt.myz[i]);
			}
			else
			{
			sprintf(errstr,"moment mechanics input wrong = %s",momtype);
			errprt(Fail2Check, errstr);
			}
			mnt.posx[i] *= dst2m; mnt.posy[i] *= dst2m; mnt.posz[i] *= dst2m;
			mnt.mxx[i] *= dm0; mnt.myy[i] *= dm0; mnt.mzz[i] *= dm0;
			mnt.mxy[i] *= dm0; mnt.mxz[i] *= dm0; mnt.myz[i] *= dm0;
		}

		for(i=0;i<nmnt;i++)
			for(j=0;j<nwin;j++)
			{
				com.get_conf(fp,"moment_stf_type",3+j,stype);
				com.get_conf(fp,"moment_stf_timefactor",3+j,&st0);
				com.get_conf(fp,"moment_stf_freqfactor",3+j,&sf0);
				cal_stf_moment(dst[i],stype,st0,sf0,nstf,stept/2.0,tempstf);
				for(k=0;k<nstf;k++)
					mnt.stf[i][k] = tempstf[k];
			}

		delete [] dst;
		dst = NULL;
	}


	//read focal NC file
	Real fd0;
	fd0 = 1.0;
	com.get_conf(fp,"simulate_damp",3,&fd0);
	fclose(fp);
	
	if(Rmnt.np)
	{
		size_t subs[2],subn[2];
		ptrdiff_t subi[2];
		Real *Mvar;
		Mvar = new Real[Rmnt.np*Rmnt.nt]();
		
		//get moment
		subs[0] = 0;	subn[0] = Rmnt.np;	subi[0] = 1;
		subs[1] = 0;	subn[1] = Rmnt.nt;	subi[1] = 1;

		snc.varget(fNCfile, "Mxx", Mvar, subs, subn, subi); 
		for(i=0;i<Rmnt.np;i++)
			for(j=0;j<Rmnt.nt;j++)
				Rmnt.mxx[i][j] = Mvar[i*Rmnt.nt + j]*fd0;
		
		snc.varget(fNCfile, "Myy", Mvar, subs, subn, subi); 
		for(i=0;i<Rmnt.np;i++)
			for(j=0;j<Rmnt.nt;j++)
				Rmnt.myy[i][j] = Mvar[i*Rmnt.nt + j]*fd0;

		snc.varget(fNCfile, "Mzz", Mvar, subs, subn, subi); 
		for(i=0;i<Rmnt.np;i++)
			for(j=0;j<Rmnt.nt;j++)
				Rmnt.mzz[i][j] = Mvar[i*Rmnt.nt + j]*fd0;

		snc.varget(fNCfile, "Mxy", Mvar, subs, subn, subi); 
		for(i=0;i<Rmnt.np;i++)
			for(j=0;j<Rmnt.nt;j++)
				Rmnt.mxy[i][j] = Mvar[i*Rmnt.nt + j]*fd0;

		snc.varget(fNCfile, "Mxz", Mvar, subs, subn, subi); 
		for(i=0;i<Rmnt.np;i++)
			for(j=0;j<Rmnt.nt;j++)
				Rmnt.mxz[i][j] = Mvar[i*Rmnt.nt + j]*fd0;

		snc.varget(fNCfile, "Myz", Mvar, subs, subn, subi); 
		for(i=0;i<Rmnt.np;i++)
			for(j=0;j<Rmnt.nt;j++)
				Rmnt.myz[i][j] = Mvar[i*Rmnt.nt + j]*fd0;
		
		//get postion
		snc.varget(fNCfile, "posx", Rmnt.posx);
		snc.varget(fNCfile, "posy", Rmnt.posy);
		snc.varget(fNCfile, "posz", Rmnt.posz);
		
		/*
		for(i=0;i<Rmnt.np;i++)
			for(j=0;j<Rmnt.nt;j++)
				printf("in Master Rmnt.Mxx[%d][%d]=%e\n",i,j,Rmnt.mxx[i][j]);
		*/

		delete [] Mvar;
		Mvar == NULL;
	}

	fprintf(stdout,"---accomplished reading the source configure file, will generate %d focus %d force %d moment source\n", Rmnt.np, nfrc, nmnt);

}

void source::BoundJug(cindx cdx, coord crd, int *nabs)
{
	fprintf(stdout,"***Confirm the distance between source and absorbing boundary\n");

	int i,j;
	
	fprintf(stdout,"Abs layer number is: ");
	for(i=0;i<6;i++)
		fprintf(stdout,"%d ",nabs[i]);
	fprintf(stdout,"\n");

	int x,y,z;

	if(nfrc)
	for(i=0;i<nfrc;i++)
	{
		x = nabs[0]+cdx.ni1; y = cdx.ny/2; z = cdx.nz/2;
		if( frc.posx[i] <= crd.x[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the LEFT boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, frc.locx[i],frc.locy[i],frc.locz[i], frc.posx[i], frc.posy[i], frc.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
		
		x = cdx.ni-nabs[1]; y = cdx.ny/2; z = cdx.nz/2;
		if( frc.posx[i] >=crd.x[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the RIGHT boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, frc.locx[i],frc.locy[i],frc.locz[i], frc.posx[i], frc.posy[i], frc.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
		
		x = cdx.nx/2; y = cdx.nj1+nabs[2]; z = cdx.nz/2;
		if( frc.posy[i] <= crd.y[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the FRONT boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, frc.locx[i],frc.locy[i],frc.locz[i], frc.posx[i], frc.posy[i], frc.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
		
		x = cdx.nx/2; y = cdx.nj-nabs[3]; z = cdx.nz/2;
		if( frc.posy[i] >= crd.y[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the BACK boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, frc.locx[i],frc.locy[i],frc.locz[i], frc.posx[i], frc.posy[i], frc.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
		
		x = cdx.nx/2; y = cdx.ny/2; z = cdx.nk1+nabs[4];
		if( frc.posz[i] <= crd.z[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the BOTTOM boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, frc.locx[i],frc.locy[i],frc.locz[i], frc.posx[i], frc.posy[i], frc.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);

		x = cdx.nx/2; y = cdx.ny/2; z = cdx.nk-nabs[5];
		if( frc.posz[i] >= crd.z[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the TOP boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, frc.locx[i],frc.locy[i],frc.locz[i], frc.posx[i], frc.posy[i], frc.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
	}
	if(nmnt)
	for(i=0;i<nmnt;i++)
	{
		x = nabs[0]+cdx.ni1; y = cdx.ny/2; z = cdx.nz/2;
		if( mnt.posx[i] <= crd.x[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the LEFT boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, mnt.locx[i],mnt.locy[i],mnt.locz[i], mnt.posx[i], mnt.posy[i], mnt.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
		
		x = cdx.ni-nabs[1]; y = cdx.ny/2; z = cdx.nz/2;
		if( mnt.posx[i] >=crd.x[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the RIGHT boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, mnt.locx[i],mnt.locy[i],mnt.locz[i], mnt.posx[i], mnt.posy[i], mnt.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
		
		x = cdx.nx/2; y = cdx.nj1+nabs[2]; z = cdx.nz/2;
		if( mnt.posy[i] <= crd.y[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the FRONT boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, mnt.locx[i],mnt.locy[i],mnt.locz[i], mnt.posx[i], mnt.posy[i], mnt.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
		
		x = cdx.nx/2; y = cdx.nj-nabs[3]; z = cdx.nz/2;
		if( mnt.posy[i] >= crd.y[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the BACK boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, mnt.locx[i],mnt.locy[i],mnt.locz[i], mnt.posx[i], mnt.posy[i], mnt.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
		
		x = cdx.nx/2; y = cdx.ny/2; z = cdx.nk1+nabs[4];
		if( mnt.posz[i] <= crd.z[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the BOTTOM boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, mnt.locx[i],mnt.locy[i],mnt.locz[i], mnt.posx[i], mnt.posy[i], mnt.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);

		x = cdx.nx/2; y = cdx.ny/2; z = cdx.nk-nabs[5];
		if( mnt.posz[i] >= crd.z[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the TOP boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, mnt.locx[i],mnt.locy[i],mnt.locz[i], mnt.posx[i], mnt.posy[i], mnt.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
	}
	if(Rmnt.np)
	for(i=0;i<Rmnt.np;i++)
	{
		x = nabs[0]+cdx.ni1; y = cdx.ny/2; z = cdx.nz/2;
		if( Rmnt.posx[i] <= crd.x[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the LEFT boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, Rmnt.locx[i],Rmnt.locy[i],Rmnt.locz[i], Rmnt.posx[i], Rmnt.posy[i], Rmnt.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
		
		x = cdx.ni-nabs[1]; y = cdx.ny/2; z = cdx.nz/2;
		if( Rmnt.posx[i] >=crd.x[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the RIGHT boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, Rmnt.locx[i],Rmnt.locy[i],Rmnt.locz[i], Rmnt.posx[i], Rmnt.posy[i], Rmnt.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
		
		x = cdx.nx/2; y = cdx.nj1+nabs[2]; z = cdx.nz/2;
		if( Rmnt.posy[i] <= crd.y[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the FRONT boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, Rmnt.locx[i],Rmnt.locy[i],Rmnt.locz[i], Rmnt.posx[i], Rmnt.posy[i], Rmnt.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
		
		x = cdx.nx/2; y = cdx.nj-nabs[3]; z = cdx.nz/2;
		if( Rmnt.posy[i] >= crd.y[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the BACK boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, Rmnt.locx[i],Rmnt.locy[i],Rmnt.locz[i], Rmnt.posx[i], Rmnt.posy[i], Rmnt.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
		
		x = cdx.nx/2; y = cdx.ny/2; z = cdx.nk1+nabs[4];
		if( Rmnt.posz[i] <= crd.z[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the BOTTOM boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, Rmnt.locx[i],Rmnt.locy[i],Rmnt.locz[i], Rmnt.posx[i], Rmnt.posy[i], Rmnt.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);

		x = cdx.nx/2; y = cdx.ny/2; z = cdx.nk-nabs[5];
		if( Rmnt.posz[i] >= crd.z[x][y][z] )
		printf("for source %d, locate at (%d,%d,%d)->position at (%f,%f,%f), exceeds the TOP boundary (%d,%d,%d)->position(%f,%f,%f)\n",
			i, Rmnt.locx[i],Rmnt.locy[i],Rmnt.locz[i], Rmnt.posx[i], Rmnt.posy[i], Rmnt.posz[i],
			x,y,z, crd.x[x][y][z],crd.y[x][y][z],crd.z[x][y][z]);
	}

	fprintf(stdout,"---accomplished source boundary judgement\n");
}

void source::cal_index(cindx cdx, coord crd, mdpar mpa)
{
	if(this->Rwork)
	{
		fprintf(stdout,"---Reading the exists source index information, there is no needs to calculate the index information, due to the restart work\n");
		return;
	}

	fprintf(stdout,"***Start to calculate source location (Index)\n");

	Real px,py,pz;//position
	int ix,iy,iz;//index
	Real tempz;//reposition Z
	bool isout;
	char errstr[SeisStrLen];
	int i,j;

	isout = false;

#ifdef SrcSmooth
	int ic,jc,kc;
	Real ***matx,***maty,***matz;
	Real shx,shy,shz;//shift
	int si,sj,sk;
	Real ***delta;
	
	matx = new Real**[3];
	maty = new Real**[3];
	matz = new Real**[3];
	for(i=0;i<3;i++)
	{
		matx[i] = new Real *[3];
		maty[i] = new Real *[3];
		matz[i] = new Real *[3];
		for(j=0;j<3;j++)
		{
			matx[i][j] = new Real[3];
			maty[i][j] = new Real[3];
			matz[i][j] = new Real[3];
		}
	}
	delta = new Real**[LenNorm];
	for(i=0;i<LenNorm;i++)
	{
		delta[i] = new Real *[LenNorm];
		for(j=0;j<LenNorm;j++)
			delta[i][j] = new Real[LenNorm];
	}
#endif

	if(nfrc)
	for(i=0;i<nfrc;i++)
	{
		ix=0; iy=0; iz=0;
		px = frc.posx[i]; py = frc.posy[i]; pz = frc.posz[i];
		com.reposition(px,py,pz,cdx,crd,hyp2g,&ix,&iy,&iz,&tempz);
		frc.posz[i] = tempz;//keep original in fixed arange, clamped at boundary.
		
		//printf("initial calculated moment location at (%d,%d,%d), and ranges is (%d=<%d, %d=<%d, %d=<%d)\n",
		//	ix,iy,iz,cdx.ni1,cdx.ni2,cdx.nj1,cdx.nj2,cdx.nk1,cdx.nk2);

		if(ix>=cdx.ni1&&ix<cdx.ni2 && iy>=cdx.nj1&&iy<cdx.nj2 && iz>=cdx.nk1&&iz<cdx.nk2)
		{
			frc.locx[i] = ix; frc.locy[i] = iy; frc.locz[i] = iz;
#ifdef SrcSmooth
			for(ic=0;ic<3;ic++)
				for(jc=0;jc<3;jc++)
					for(kc=0;kc<3;kc++)
					{
						matx[ic][jc][kc] = crd.x[ix-1+ic][iy-1+jc][iz-1+kc];
						maty[ic][jc][kc] = crd.y[ix-1+ic][iy-1+jc][iz-1+kc];
						matz[ic][jc][kc] = crd.z[ix-1+ic][iy-1+jc][iz-1+kc];
					}
			shx = 0.0; shy = 0.0; shz = 0.0;

			for(j=0;j<4;j++)
			{
				eighth_locate(frc.posx[i],frc.posy[i],frc.posz[i],matx,maty,matz,&si,&sj,&sk);
				shx += si/pow(2.0,j+1);
				shy += sj/pow(2.0,j+1);
				shz += sk/pow(2.0,j+1);
			}

			cal_norm(shx,shy,shz,iz,cdx.nk2,delta);

			for(ic=0;ic<LenNorm;ic++)
				for(jc=0;jc<LenNorm;jc++)
					for(kc=0;kc<LenNorm;kc++)
						frc.dnorm[i][ic][jc][kc] = delta[ic][jc][kc];
#endif
			if( frc.locx[i]<cdx.ni1 || frc.locx[i]>=cdx.ni2 || frc.locy[i]<cdx.nj1 || frc.locy[i]>=cdx.nj2 || frc.locz[i]<cdx.nk1 || frc.locz[i]>=cdx.nk2 )
			{
				sprintf(errstr,"The %d force source at position(%f,%f,%f), exceed the coord boundary which is location(%d,%d,%d)",
						i+1,frc.posx[i],frc.posy[i],frc.posz[i],frc.locx[i],frc.locy[i],frc.locz[i]);
				errprt(Fail2Check,errstr);
			}

			fprintf(stdout,"The %d force source: Request-position(%f,%f,%f), Actually depth is %f, corresponding location(%d,%d,%d), applying at {"
				,i+1,frc.posx[i],frc.posy[i],frc.posz[i],crd.z[ix][iy][iz],frc.locx[i],frc.locy[i],frc.locz[i]);
			if(frc.fx[i])fprintf(stdout," Fx ");
			if(frc.fy[i])fprintf(stdout," Fy ");
			if(frc.fz[i])fprintf(stdout," Fz ");
			fprintf(stdout," } directions\n");
		}
		else
		{
			isout = true;
			sprintf(errstr,"The %d force source at position(%f,%f,%f), exceed the coord boundary which is location(%d,%d,%d)",
					i+1,frc.posx[i],frc.posy[i],frc.posz[i],ix,iy,iz);
			errprt(Fail2Check,errstr);
		}
	
	}
	

	if(nmnt)
	for(i=0;i<nmnt;i++)
	{
		ix=0; iy=0; iz=0;
		px = mnt.posx[i]; py = mnt.posy[i]; pz = mnt.posz[i];
		com.reposition(px,py,pz,cdx,crd,hyp2g,&ix,&iy,&iz,&tempz);
		mnt.posz[i] = tempz;

		//printf("initial calculated moment location at (%d,%d,%d), and ranges is (%d=<%d, %d=<%d, %d=<%d)\n",
		//	ix,iy,iz,cdx.ni1,cdx.ni2,cdx.nj1,cdx.nj2,cdx.nk1,cdx.nk2);

		if(ix>=cdx.ni1&&ix<cdx.ni2 && iy>=cdx.nj1&&iy<cdx.nj2 && iz>=cdx.nk1&&iz<cdx.nk2)
		{
			mnt.locx[i] = ix; mnt.locy[i] = iy; mnt.locz[i] = iz;
#ifdef SrcSmooth
			MomentShiftPosition(crd, &mnt.locx[i], &mnt.locy[i], &mnt.locz[i], mnt.posx[i], mnt.posy[i], mnt.posz[i], &shx, &shy, &shz);

			cal_norm(shx,shy,shz,iz,cdx.nk2,delta);

			for(ic=0;ic<LenNorm;ic++)
				for(jc=0;jc<LenNorm;jc++)
					for(kc=0;kc<LenNorm;kc++)
					{
						mnt.dnorm[i][ic][jc][kc] = delta[ic][jc][kc];
						//printf("dnorm at[%d][%d][%d]=%e\n",ix-1+ic,iy-1+jc,iz-1+kc,delta[ic][jc][kc]);//AAA
					}
#endif
			if( ISEQSTR(momtype,"angle") )
			{
				Real miu;
				miu = mpa.rho[ mnt.locx[i] ][ mnt.locy[i] ][ mnt.locz[i] ]*
				      mpa.beta[ mnt.locx[i] ][ mnt.locy[i] ][ mnt.locz[i] ]*
				      mpa.beta[ mnt.locx[i] ][ mnt.locy[i] ][ mnt.locz[i] ];
				mnt.mxx[i] *= miu; mnt.myy[i] *= miu; mnt.mzz[i] *= miu;
				mnt.mxy[i] *= miu; mnt.mxz[i] *= miu; mnt.myz[i] *= miu;
			}

			if( mnt.locx[i]<cdx.ni1 || mnt.locx[i]>=cdx.ni2 || mnt.locy[i]<cdx.nj1 || mnt.locy[i]>=cdx.nj2 || mnt.locz[i]<cdx.nk1 || mnt.locz[i]>=cdx.nk2 )
			{
				sprintf(errstr,"The %d moment source at position(%f,%f,%f), exceed the coord boundary which is location(%d,%d,%d)",
						i+1,mnt.posx[i],mnt.posy[i],mnt.posz[i],mnt.locx[i],mnt.locy[i],mnt.locz[i]);
				errprt(Fail2Check,errstr);
			}

			fprintf(stdout,"The %d moment source: Request-position(%f,%f,%f), Actually depth is %f, corresponding location(%d,%d,%d), applying at {"
				,i+1,mnt.posx[i],mnt.posy[i],mnt.posz[i],crd.z[ix][iy][iz],mnt.locx[i],mnt.locy[i],mnt.locz[i]);
			if(mnt.mxx[i])fprintf(stdout," Mxx ");
			if(mnt.myy[i])fprintf(stdout," Myy ");
			if(mnt.mzz[i])fprintf(stdout," Mzz ");
			if(mnt.mxy[i])fprintf(stdout," Mxy ");
			if(mnt.mxz[i])fprintf(stdout," Mxz ");
			if(mnt.myz[i])fprintf(stdout," Myz ");
			fprintf(stdout," } components\n");
		}
		else
		{
			isout = true;
			sprintf(errstr,"The %d moment source at position(%f,%f,%f), exceed the coord boundary which is location(%d,%d,%d)",
					i+1,mnt.posx[i],mnt.posy[i],mnt.posz[i],ix,iy,iz);
			errprt(Fail2Check,errstr);
		}

	}
	
	
	if(Rmnt.np)
	for(i=0;i<Rmnt.np;i++)
	{
		ix=0; iy=0; iz=0;
		px = Rmnt.posx[i]; py = Rmnt.posy[i]; pz = Rmnt.posz[i];
		com.reposition(px,py,pz,cdx,crd,hyp2g,&ix,&iy,&iz,&tempz);
		Rmnt.posz[i] = tempz;

		//printf("initial calculated moment location at (%d,%d,%d), and ranges is (%d=<%d, %d=<%d, %d=<%d)\n",
		//	ix,iy,iz,cdx.ni1,cdx.ni2,cdx.nj1,cdx.nj2,cdx.nk1,cdx.nk2);
		printf("point location is (%f, %f, %f), position location is (%f,%f,%f)\n",
			px,py,pz, crd.x[ix][iy][iz],crd.y[ix][iy][iz],crd.z[ix][iy][iz]);
		//Check OK, accurate positioning 20240416

		if(ix>=cdx.ni1&&ix<cdx.ni2 && iy>=cdx.nj1&&iy<cdx.nj2 && iz>=cdx.nk1&&iz<cdx.nk2)
		{
			Rmnt.locx[i] = ix; Rmnt.locy[i] = iy; Rmnt.locz[i] = iz;
			
			if( Rmnt.locx[i]<cdx.ni1 || Rmnt.locx[i]>=cdx.ni2 || 
			    Rmnt.locy[i]<cdx.nj1 || Rmnt.locy[i]>=cdx.nj2 || 
			    Rmnt.locz[i]<cdx.nk1 || Rmnt.locz[i]>=cdx.nk2 )
			{
				fprintf(stdout,"The %d focal source at position(%f,%f,%f), exceed the coord boundary which is location(%d,%d,%d)",
						i+1,Rmnt.posx[i],Rmnt.posy[i],Rmnt.posz[i],Rmnt.locx[i],Rmnt.locy[i],Rmnt.locz[i]);
				memset( Rmnt.mxx[i], 0, Rmnt.nt*sizeof(Real) );
				memset( Rmnt.myy[i], 0, Rmnt.nt*sizeof(Real) );
				memset( Rmnt.mzz[i], 0, Rmnt.nt*sizeof(Real) );
				memset( Rmnt.mxy[i], 0, Rmnt.nt*sizeof(Real) );
				memset( Rmnt.mxz[i], 0, Rmnt.nt*sizeof(Real) );
				memset( Rmnt.myz[i], 0, Rmnt.nt*sizeof(Real) );

				continue;

				//sprintf(errstr,"The %d focal source at position(%f,%f,%f), exceed the coord boundary which is location(%d,%d,%d)",
				//		i+1,Rmnt.posx[i],Rmnt.posy[i],Rmnt.posz[i],Rmnt.locx[i],Rmnt.locy[i],Rmnt.locz[i]);
				//errprt(Fail2Check,errstr);
			}

#ifdef SrcSmooth
			MomentShiftPosition(crd, &Rmnt.locx[i], &Rmnt.locy[i], &Rmnt.locz[i], Rmnt.posx[i], Rmnt.posy[i], Rmnt.posz[i], &shx, &shy, &shz);

			cal_norm(shx,shy,shz,iz,cdx.nk2,delta);

			for(ic=0;ic<LenNorm;ic++)
				for(jc=0;jc<LenNorm;jc++)
					for(kc=0;kc<LenNorm;kc++)
					{
						Rmnt.dnorm[i][ic][jc][kc] = delta[ic][jc][kc];
						//printf("dnorm at[%d][%d][%d][%d]=%e\n",i,ix-1+ic,iy-1+jc,iz-1+kc,delta[ic][jc][kc]);//AAA
						//printf("dnorm at[%d][%d]=%e\n",i, ic*LenNorm*LenNorm+jc*LenNorm+kc, delta[ic][jc][kc]);//AAA
					}
#endif
			
			Real miu;
			miu = mpa.rho[ Rmnt.locx[i] ][ Rmnt.locy[i] ][ Rmnt.locz[i] ]*
				mpa.beta[ Rmnt.locx[i] ][ Rmnt.locy[i] ][ Rmnt.locz[i] ]*
				mpa.beta[ Rmnt.locx[i] ][ Rmnt.locy[i] ][ Rmnt.locz[i] ];
			for(int jjj=0; jjj<Rmnt.nt; jjj++)
			{
				Rmnt.mxx[i][jjj] = miu*Rmnt.mxx[i][jjj]; Rmnt.myy[i][jjj] = miu*Rmnt.myy[i][jjj]; Rmnt.mzz[i][jjj] = miu*Rmnt.mzz[i][jjj];
				Rmnt.mxy[i][jjj] = miu*Rmnt.mxy[i][jjj]; Rmnt.mxz[i][jjj] = miu*Rmnt.mxz[i][jjj]; Rmnt.myz[i][jjj] = miu*Rmnt.myz[i][jjj];
			}

			/*
			for(int jjj=0; jjj<10;jjj++)
				printf("H-Rmnt.mxx[%d][%d]=%g\n",i,jjj,Rmnt.mxx[i][jjj]);
			fprintf(stdout,"The %d focal source: Request-position(%f,%f,%f), Actually depth is %f, corresponding location(%d,%d,%d), applying at {"
				,i+1,Rmnt.posx[i],Rmnt.posy[i],Rmnt.posz[i],crd.z[ix][iy][iz],Rmnt.locx[i],Rmnt.locy[i],Rmnt.locz[i]);
			if(Rmnt.mxx[i])fprintf(stdout," Mxx ");
			if(Rmnt.myy[i])fprintf(stdout," Myy ");
			if(Rmnt.mzz[i])fprintf(stdout," Mzz ");
			if(Rmnt.mxy[i])fprintf(stdout," Mxy ");
			if(Rmnt.mxz[i])fprintf(stdout," Mxz ");
			if(Rmnt.myz[i])fprintf(stdout," Myz ");
			fprintf(stdout," } components\n");
			*/
		}
		else
		{
			isout = true;
			memset( Rmnt.mxx[i], 0, Rmnt.nt*sizeof(Real) );
			memset( Rmnt.myy[i], 0, Rmnt.nt*sizeof(Real) );
			memset( Rmnt.mzz[i], 0, Rmnt.nt*sizeof(Real) );
			memset( Rmnt.mxy[i], 0, Rmnt.nt*sizeof(Real) );
			memset( Rmnt.mxz[i], 0, Rmnt.nt*sizeof(Real) );
			memset( Rmnt.myz[i], 0, Rmnt.nt*sizeof(Real) );

			//sprintf(errstr,"The %d focal source at position(%f,%f,%f), exceed the coord boundary which is location(%d,%d,%d)",
			//		i+1,Rmnt.posx[i],Rmnt.posy[i],Rmnt.posz[i],ix,iy,iz);
			//errprt(Fail2Check,errstr);
		}

	}


#ifdef SrcSmooth
	for(i=0;i<LenNorm;i++)
	{
		for(j=0;j<LenNorm;j++)
			delete [] delta[i][j];
		delete [] delta[i];
	}
	delete [] delta;
	for(i=0;i<3;i++)
	{
		for(j=0;j<3;j++)
		{
			delete [] matz[i][j];
			delete [] maty[i][j];
			delete [] matx[i][j];
		}
		delete [] matz[i];
		delete [] maty[i];
		delete [] matx[i];
	}
	delete [] matz;
	delete [] maty;
	delete [] matx;
#endif
	if(isout)
		errprt(Fail2Check,"Some source points are out of computing physical area");

	fprintf(stdout,"---accomplished calculating source locations\n");

}

void source::M2CPointPick(cindx cdx, int *Xs, int *Xe, int *fp, int cpn)
{
	int i,j;
	int numP;
	
	if(Rmnt.np==0)
	{
		for(i=0;i<cpn;i++)
			fp[i] = 0; //set 0 for Mid.fp
		return;//no focal applied, doesn't need to do point pick
	}

	
	for(j=0;j<cpn;j++)
	{
		numP = 0;
		for(i=0;i<Rmnt.np;i++)
			if(Rmnt.locx[i]>=Xs[j] && Rmnt.locx[i]<=Xe[j])// from Master to Child, only split X index
				numP++;
		fp[j] = numP;

		Fpt.Rsn[j] = new int[numP]();
		Fpt.Gsn[j] = new int[numP]();
		Fpt.locx[j] = new int[numP]();
		Fpt.locy[j] = new int[numP]();
		Fpt.locz[j] = new int[numP]();
	}

	for(j=0;j<cpn;j++)
	{
		numP = 0;
		for(i=0;i<Rmnt.np;i++)
			if(Rmnt.locx[i]>=Xs[j] && Rmnt.locx[i]<=Xe[j])
			{
				Fpt.Rsn[j][numP] = numP;
				Fpt.Gsn[j][numP] = Rmnt.SN[i];
				Fpt.locx[j][numP] = Rmnt.locx[i];
				Fpt.locy[j][numP] = Rmnt.locy[i];
				Fpt.locz[j][numP] = Rmnt.locz[i];
				numP++;
			}
	}
	
	/*
	for(j=0;j<cpn;j++)
	{
		for(i=0;i<fp[j];i++)
			printf("Master-Fpt:Rsn[%d],Gsn[%d]->(%d,%d,%d)\n",Fpt.Rsn[j][i],Fpt.Gsn[j][i],Fpt.locx[j][i],Fpt.locy[j][i],Fpt.locz[j][i]);
	}
	*/
	
}

void source::export_data(const char *path, force frc, moment mnt)
{
	char frcfile[SeisStrLen], mntfile[SeisStrLen], focalfile[SeisStrLen];
	sprintf(frcfile, "%s/force.nc",path);
	sprintf(mntfile, "%s/moment.nc",path);
	sprintf(focalfile, "%s/focal.nc",path);

	if(this->Rwork)
	{
		fprintf(stdout,"---Thers's no needs to store data, due to the restart work\n");
		if(nfrc) snc.force_import(frc, frcfile, nfrc, nstf);
		if(nmnt) snc.moment_import(mnt, mntfile, nmnt, nstf);
		if(Rmnt.np) snc.focal_import(Rmnt, focalfile);
	}
	else
	{
		if(nfrc) snc.force_export(frc, frcfile, nfrc, nstf);
		if(nmnt) snc.moment_export(mnt, mntfile, nmnt, nstf);
		if(Rmnt.np) snc.focal_export(Rmnt, focalfile);
	}
}
































