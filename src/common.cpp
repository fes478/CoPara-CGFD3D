#include "typenew.h"

using namespace constant;
using namespace defstruct;
using namespace std;

#define errprt(...) errorprint(__FILE__,__LINE__,__VA_ARGS__)

//----------------------------public part----------------------------------------

void common::errorprint(const char *filename, int numline, int errorcode, const char *descrp)
{
	if(errorcode)
	{
		printf("Error at %s(%d)--->",filename,numline);
		MpiErrorPrint(filename, numline, errorcode, descrp);
	}
	else
		printf("Warning at %s(%d),%s.\n",filename,numline,descrp);
}


void common::get_conf(FILE *fp, const char *keystr, int numcol, int *outval)
{
	char valstr[SeisStrLen];
	getConfStr(fp,keystr,numcol,valstr);
	*outval=atoi(valstr);
}
void common::get_conf(FILE *fp, const char *keystr, int numcol, long int *outval)
{
	char valstr[SeisStrLen];
	getConfStr(fp,keystr,numcol,valstr);
	*outval=atol(valstr);
}
void common::get_conf(FILE *fp, const char *keystr, int numcol, double *outval)
{
	char valstr[SeisStrLen];
	getConfStr(fp,keystr,numcol,valstr);
	*outval=atof(valstr);
}
void common::get_conf(FILE *fp, const char *keystr, int numcol, float *outval)
{
	char valstr[SeisStrLen];
	getConfStr(fp,keystr,numcol,valstr);
	*outval=(float)atof(valstr);
}
void common::get_conf(FILE *fp, const char *keystr, int numcol, char outval[])
{
	getConfStr(fp,keystr,numcol,outval);
}
void common::get_conf(FILE *fp, const char *keystr, int numcol, bool *outval)
{
	char valstr[SeisStrLen];
	getConfStr(fp,keystr,numcol,valstr);
	if(ISEQSTR(valstr,"0"))
		*outval=false;
	else
		*outval=true;
}

void common::setchunk(FILE *fp,const char *anchor)
{
	char valstr[SeisStrLen];
	getConfStr(fp,anchor,1,valstr);
}

void common::interpolated_extend(Real ***var, cindx cdx)
{
	//used for grid, with corner
	//
	//  choose  2=3-4   1=3-5   0=3-6   11=10-9  12=10-8  13=10-7
	int i,j,k;
	
	//Z direction
	for(i=0;i<cdx.nx;i++)
		for(j=0;j<cdx.ny;j++)
			for(k=1;k<=LenFD;k++)
			{
				var[i][j][cdx.nk1-k]=2.0*var[i][j][cdx.nk1]-var[i][j][cdx.nk1+k];
				var[i][j][cdx.nk2+k - 1]=2.0*var[i][j][cdx.nk2 - 1]-var[i][j][cdx.nk2-k - 1];
			}
	//Y direction
	for(i=0;i<cdx.nx;i++)
		for(k=0;k<cdx.nz;k++)
			for(j=1;j<=LenFD;j++)
			{
				var[i][cdx.nj1-j][k]=2.0*var[i][cdx.nj1][k]-var[i][cdx.nj1+j][k];
				var[i][cdx.nj2+j - 1][k]=2.0*var[i][cdx.nj2 - 1][k]-var[i][cdx.nj2-j - 1][k];
			}
	//X direction
	for(j=0;j<cdx.ny;j++)
		for(k=0;k<cdx.nz;k++)
			for(i=1;i<=LenFD;i++)
			{
				var[cdx.ni1-i][j][k]=2.0*var[cdx.ni1][j][k]-var[cdx.ni1+i][j][k];
				var[cdx.ni2+i - 1][j][k]=2.0*var[cdx.ni2 - 1][j][k]-var[cdx.ni2-i - 1][j][k];
			}

}
void common::mirror_extend(Real ***var, cindx cdx)
{
	//used for metric, without corner
	//   nx1            ni1                                    ni2             nx2 
        //   0    1    2    3    4    5    6    7    8    9   10   11   12   13    14
        //   F    F    F    A    A    A    A    A    A    A    A    F    F    F
	//
	//   choose 2-3   1-4   0-5   10-11  9-12  8-13   ||||   CASE1
	//   choose 2-4   1-5   0-6   9-11   8-12  7-13   ||||   CASE2(ZhangWei
	//   CAUTION ni2 should minus 1 when use
	int i,j,k;
	//X direction;
	for(i=1;i<=LenFD;i++)
		for(j=cdx.nj1;j<cdx.nj2;j++)
			for(k=cdx.nk1;k<cdx.nk2;k++)
			{
				var[cdx.ni1-i][j][k]=var[cdx.ni1+i][j][k];
				var[cdx.ni2-1+i][j][k]=var[cdx.ni2-1-i][j][k];
			}
	//Y direction;
	for(i=cdx.ni1;i<cdx.ni2;i++)
		for(k=cdx.nk1;k<cdx.nk2;k++)
			for(j=1;j<=LenFD;j++)
			{
				var[i][cdx.nj1-j][k]=var[i][cdx.nj1+j][k];
				var[i][cdx.nj2-1+j][k]=var[i][cdx.nj2-1-j][k];
			}
	//Z direction;
	for(i=cdx.ni1;i<cdx.ni2;i++)
		for(j=cdx.nj1;j<cdx.nj2;j++)
			for(k=1;k<=LenFD;k++)
			{
				var[i][j][cdx.nk1-k]=var[i][j][cdx.nk1+k];
				var[i][j][cdx.nk2-1+k]=var[i][j][cdx.nk2-1-k];
			}
}
void common::equivalence_extend(Real ***var, cindx cdx)
{
	//used for media, with corner
	//
	//  choose 3 10
	int i,j,k;
	//X direction;
	for(i=1;i<=LenFD;i++)
		for(j=0;j<cdx.ny;j++)
			for(k=0;k<cdx.nz;k++)
			{
				var[cdx.ni1-i][j][k]=var[cdx.ni1][j][k];
				var[cdx.ni2+i - 1][j][k]=var[cdx.ni2-1][j][k];
			}
	//Y direction;
	for(i=0;i<cdx.nx;i++)
		for(k=0;k<cdx.nz;k++)
			for(j=1;j<=LenFD;j++)
			{
				var[i][cdx.nj1-j][k]=var[i][cdx.nj1+j][k];
				var[i][cdx.nj2+j - 1][k]=var[i][cdx.nj2-1][k];
			}
	//Z direction;
	for(i=0;i<cdx.nx;i++)
		for(j=0;j<cdx.ny;j++)
			for(k=1;k<=LenFD;k++)
			{
				var[i][j][cdx.nk1-k]=var[i][j][cdx.nk1+k];
				var[i][j][cdx.nk2+k - 1]=var[i][j][cdx.nk2-1];
			}
}

void common::flatten21D(Real ***src, Real *dest, cindx cdx, int flag)
{
	int i,j,k;
	int Dnx,Dny,Dnz;
	if(flag)
	{
		//flag=1, use small scale, only valid point;
		Dnx=cdx.ni;
		Dny=cdx.nj;
		Dnz=cdx.nk;
	}
	else
	{
		//flag=0, use full scale, with virtual point;
		Dnx=cdx.nx;
		Dny=cdx.ny;
		Dnz=cdx.nz;
	}

	for(i=0;i<Dnx;i++)
		for(j=0;j<Dny;j++)
			for(k=0;k<Dnz;k++)
			{
				if(flag)
					dest[i*Dny*Dnz + j*Dnz +k] = src[i+cdx.ni1][j+cdx.nj1][k+cdx.nk1];
				else
					dest[i*Dny*Dnz + j*Dnz +k] = src[i][j][k];
			}
}
void common::compress23D(Real *src, Real ***dest, cindx cdx, int flag)
{
	int i,j,k;
	int Dnx,Dny,Dnz;
	if(flag)
	{
		//flag=1, use small scale, only valid point;
		Dnx=cdx.ni;
		Dny=cdx.nj;
		Dnz=cdx.nk;
	}
	else
	{
		//flag=0, use full scale, with virtual point;
		Dnx=cdx.nx;
		Dny=cdx.ny;
		Dnz=cdx.nz;
	}

	for(i=0;i<Dnx;i++)
		for(j=0;j<Dny;j++)
			for(k=0;k<Dnz;k++)
				if(flag)
					dest[i+cdx.ni1][j+cdx.nj1][k+cdx.nk1] = src[i*Dny*Dnz + j*Dnz + k];
				else
					dest[i][j][k] = src[i*Dny*Dnz + j*Dnz + k];
}

void common::reposition(Real px, Real py, Real pz, cindx cdx, coord crd, Real hyp2g, int *ix, int *iy, int *iz, Real *tempz)
{
	//tempz = position
	int i,j,k;
	Real dist,p;
	dist = SeisInf;
	p = SeisZero;
	if( pz > hyp2g)
	{
		for(i=cdx.ni1;i<cdx.ni2;i++)
			for(j=cdx.nj1;j<cdx.nj2;j++)
			{
				p = (crd.x[i][j][cdx.nk2-1]-px)*(crd.x[i][j][cdx.nk2-1]-px)+
			            (crd.y[i][j][cdx.nk2-1]-py)*(crd.y[i][j][cdx.nk2-1]-py);
				if(p<dist)
				{
					dist = p;
					*ix = i;
					*iy = j;
				}
			}
		*iz = cdx.nk2-1;
		*tempz = crd.z[*ix][*iy][*iz];
	}
	else
	{
		for(i=cdx.ni1;i<cdx.ni2;i++)
			for(j=cdx.nj1;j<cdx.nj2;j++)
				for(k=cdx.nk1;k<cdx.nk2;k++)
				{
					p = (crd.x[i][j][k]-px)*(crd.x[i][j][k]-px) + 
					    (crd.y[i][j][k]-py)*(crd.y[i][j][k]-py) +
					    (crd.z[i][j][k]-pz)*(crd.z[i][j][k]-pz);
					if(p<dist)
					{
						dist = p;
						*ix = i;
						*iy = j;
						*iz = k;
					}
				}
		*tempz = pz;
	}
}
















//------------------------------private part--------------------
void common::getConfStr(FILE *fp, const char *keystr, int numcol, char outstr[])
{
	char str[SeisStrLen],fmtstr[SeisStrLen],scnstr[SeisStrLen];
	bool isget;
	int i;

	isget=false;
	rewind(fp);

	while(fgets(str,SeisStrLen,fp))
	{
		if(str[0]=='#')
			continue;
		else
		{
			if(sscanf(str,"%s",scnstr)==1)
				if(!strcmp(scnstr,keystr))
				{
					strcpy(fmtstr,"");
					for(i=1;i<numcol;i++)
						strcat(fmtstr,"%*s");
					strcat(fmtstr,"%s");
					if(sscanf(str,fmtstr,outstr)==1)
					{
						isget=true;
						break;
					}
				}
		}
	}
	if(!isget)
	{
		char errstr[SeisStrLen];
		sprintf(errstr,"there is no value found at %d colomn for %s at %s",numcol,keystr,"common::getConfStr");
		errprt(Fail2Read,errstr);
	}
}











