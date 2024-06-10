#include "typenew.h"
#include<math.h>

using namespace constant;
using namespace defstruct;

//-----------------------public----------------
int mathfunc::max(int *p, int n)
{
	int pmax;
	pmax=p[0];
	for(int i=0;i<n;i++)
		pmax=MAX(pmax,p[i]);
	return pmax;
}
float mathfunc::max(float *p, int n)
{
	float pmax;
	pmax=p[0];
	for(int i=0;i<n;i++)
		pmax=MAX(pmax,p[i]);
	return pmax;
}
double mathfunc::max(double *p, int n)
{
	double pmax;
	pmax=p[0];
	for(int i=0;i<n;i++)
		pmax=MAX(pmax,p[i]);
	return pmax;
}


int mathfunc::min(int *p, int n)
{
	int pmin;
	pmin=p[0];
	for(int i=0;i<n;i++)
		pmin=MIN(pmin,p[i]);
	return pmin;
}
float mathfunc::min(float *p, int n)
{
	float pmin;
	pmin=p[0];
	for(int i=0;i<n;i++)
		pmin=MIN(pmin,p[i]);
	return pmin;
}
double mathfunc::min(double *p, int n)
{
	double pmin;
	pmin=p[0];
	for(int i=0;i<n;i++)
		pmin=MIN(pmin,p[i]);
	return pmin;
}

int mathfunc::sum(int *p, int n)
{
	int s;
	s=0;
	for(int i=0;i<n;i++)
		s+=p[i];
	return s;
}
float mathfunc::sum(float *p, int n)
{
	float s;
	s=0.0F;
	for(int i=0;i<n;i++)
		s+=p[i];
	return s;
}
double mathfunc::sum(double *p, int n)
{
	double s;
	s=0.0;
	for(int i=0;i<n;i++)
		s+=p[i];
	return s;
}

long int mathfunc::accumulation(int *p, int n)
{
	long int a;
	a=1L;
	for(int i=0;i<n;i++)
		a*=p[i];
	return a;
}
long int mathfunc::accumulation(size_t *p, int n)
{
	long int a;
	a=1L;
	for(int i=0;i<n;i++)
		a*=p[i];
	return a;
}
double mathfunc::accumulation(float *p, int n)
{
	double a;
	a=1.0;
	for(int i=0;i<n;i++)
		a*=p[i];
	return a;
}
long double mathfunc::accumulation(double *p, int n)
{
	long double a;
	a=1.0L;
	for(int i=0;i<n;i++)
		a*=p[i];
	return a;
}

int mathfunc::dotproduct(int *A, int *B, int n)
{
	//A(a,b,c)
	//B(m,n,p)
	//dp(A,B)=am+bn+cp;
	// cos(angle)= dp(A,B)/(norm(A)*norm(B))
	int dp;
	dp=0;
	for(int i=0;i<n;i++)
		dp+=A[i]*B[i];
	return dp;
}
float mathfunc::dotproduct(float *A, float *B, int n)
{
	float dp;
	dp=0.0F;
	for(int i=0;i<n;i++)
		dp+=A[i]*B[i];
	return dp;
}
double mathfunc::dotproduct(double *A, double *B, int n)
{
	double dp;
	dp=0.0;
	for(int i=0;i<n;i++)
		dp+=A[i]*B[i];
	return dp;
}

int mathfunc::crossproduct(int *A, int *B, int *C)
{
	C[0]=A[1]*B[2]-A[2]*B[1];
	C[1]=A[2]*B[0]-A[0]*B[2];
        C[2]=A[0]*B[1]-A[1]*B[0];
	return 0;
}
float mathfunc::crossproduct(float *A, float *B, float *C)
{
	C[0]=A[1]*B[2]-A[2]*B[1];
	C[1]=A[2]*B[0]-A[0]*B[2];
        C[2]=A[0]*B[1]-A[1]*B[0];
	return 0;
}
double mathfunc::crossproduct(double *A, double *B, double *C)
{
	C[0]=A[1]*B[2]-A[2]*B[1];
	C[1]=A[2]*B[0]-A[0]*B[2];
        C[2]=A[0]*B[1]-A[1]*B[0];
	return 0;
}

float mathfunc::vectornorm(float *A,int n)
{
	return (float)sqrt(this->dotproduct(A,A,n));
}
double mathfunc::vectornorm(double *A,int n)
{
	return sqrt(this->dotproduct(A,A,n));
}

void mathfunc::matmul(Real **A, Real **B, Real **C)
{
	int m=3,n=3;//3*3 matrix
	int i,j,k;
	for(i=0;i<m;i++)
		for(j=0;j<n;j++)
		{
			C[i][j]=0.0;
			for(k=0;k<m;k++)
				C[i][j]=C[i][j]+A[i][k]*B[k][j];
		}
}
void mathfunc::matmul(Real A[3][3], Real B[3][3], Real C[3][3])
{
	int m=3,n=3;//3*3 matrix
	int i,j,k;
	for(i=0;i<m;i++)
		for(j=0;j<n;j++)
		{
			C[i][j]=0.0;
			for(k=0;k<m;k++)
				C[i][j]=C[i][j]+A[i][k]*B[k][j];
		}
}

Real mathfunc::determinant(Real **A)
{
	Real det;
	det=A[0][0]*(A[1][1]*A[2][2]-A[2][1]*A[1][2]);
	det=det-A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]);
	det=det+A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
	return det;
}
Real mathfunc::determinant(Real A[3][3])
{
	Real det;
	det=A[0][0]*(A[1][1]*A[2][2]-A[2][1]*A[1][2]);
	det=det-A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]);
	det=det+A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
	return det;
}

void mathfunc::matinv(Real **A)
{
	int i,j,k,n;
	n=3;//3*3 matrix
	Real con;
	for(i=0;i<n;i++)
	{
		con=A[i][i];
		A[i][i]=1;
		for(j=0;j<n;j++)
			A[i][j]=A[i][j]/con;
		
		for(j=0;j<n;j++)
			if(j!=i)
			{
				con=A[j][i];
				A[j][i]=0;
				for(k=0;k<n;k++)
					A[j][k]=A[j][k]-A[i][k]*con;
			}
	}
}
void mathfunc::matinv(Real A[3][3])
{
	int i,j,k,n;
	n=3;//3*3 matrix
	Real con;
	for(i=0;i<n;i++)
	{
		con=A[i][i];
		A[i][i]=1;
		for(j=0;j<n;j++)
			A[i][j]=A[i][j]/con;
		
		for(j=0;j<n;j++)
			if(j!=i)
			{
				con=A[j][i];
				A[j][i]=0;
				for(k=0;k<n;k++)
					A[j][k]=A[j][k]-A[i][k]*con;
			}
	}
}

long int mathfunc::locmin(Real *P, const int n)
{
	long int k;
	Real pmin;
	k=0;
	pmin=P[k];
	for(int i=0;i<n;i++)
		if(pmin>P[i])
		{
			k=i;
			pmin=P[k];
		}
	return k;
}

long int mathfunc::locmax(Real *P, const int n)
{
	long int k;
	Real pmax;
	k=0;
	pmax=P[k];
	for(int i=0;i<n;i++)
		if(pmax<P[i])
		{
			k=i;
			pmax=P[k];
		}
	return k;
}

void mathfunc::LocValue1d(Real pointV, Real *array, int n, int *L, int *R, Real *value)
{
	//only valid for increase array, array[0] is small array[n-1] is big
	int i0;
	*L=0;
	*R=n-1;
	*value=pointV;
	
	if(pointV<=array[*L])
	{
		*value=array[*L];
		*R=*L+1;
	}
	else if(pointV>=array[*R])
	{
		*value=array[*R];
		*L=*R-1;
	}
	else
		while(*R-*L>1)
		{
			i0=(*R-*L)/2+*L;
			if(array[i0]==pointV)
			{
				*L=i0;
				*R=*L+1;
				break;
			}
			else if(array[i0]>pointV)
				*R=i0;
			else
				*L=i0;
		}

}

Real mathfunc::interp1d(Real *X, Real *Y, int nx, Real x0)
{
	Real y0;
	Real Lx[nx],xb[nx],xt[nx];
	int i,j;

	y0=0.0;
	for(i=0;i<nx;i++)
	{
		for(j=0;j<nx;j++)
		{
			xb[j]=X[j]-X[i];
			xt[j]=X[j]-x0;
		}
		xb[i]=1.0;
		xt[i]=1.0;
		Lx[i]=1.0;
		for(j=0;j<nx;j++)
			Lx[i] *= xt[j]/xb[j];
		y0 += Lx[i]*Y[i];
	}
	return y0;
}
Real mathfunc::interp2d(Real *X, Real *Y, Real **Z, int nx, int ny, Real x0, Real y0)
{
	Real z0;
	Real Lx[nx],xb[nx],xt[nx];
	Real Ly[ny],yb[ny],yt[ny];
	int i,j;

	for(i=0;i<nx;i++)
	{
		for(j=0;j<nx;j++)
		{
			xb[j]=X[j]-X[i];
			xt[j]=X[j]-x0;
		}
		xb[i]=1.0;
		xt[i]=1.0;
		Lx[i]=1.0;
		for(j=0;j<nx;j++)
			Lx[i]*=xt[j]/xb[j];
	}

	for(j=0;j<ny;j++)
	{
		for(i=0;i<ny;i++)
		{
			yb[i]=Y[i]-Y[j];
			yt[i]=Y[i]-y0;
		}
		yb[j]=1.0;
		yt[j]=1.0;
		Ly[j]=1.0;
		for(i=0;i<ny;i++)
			Ly[j]*=yt[i]/yb[i];
	}

	z0=0.0;
	for(i=0;i<nx;i++)
		for(j=0;j<ny;j++)
			z0 += Lx[i]*Ly[j]*Z[i][j];

	return z0;
}
Real mathfunc::interp3d(Real *X, Real *Y, Real *Z, Real ***F, int nx, int ny, int nz, Real x0, Real y0, Real z0)
{
	Real f0;
	Real Lx[nx],xb[nx],xt[nx];
	Real Ly[ny],yb[ny],yt[ny];
	Real Lz[nz],zb[nz],zt[nz];
	int i,j,k;
	for(i=0;i<nx;i++)
	{
		for(j=0;j<nx;j++)
		{
			xb[j]=X[j]-X[i];
			xt[j]=X[j]-x0;
		}
		xb[i]=1.0;
		xt[i]=1.0;
		Lx[i]=1.0;
		for(j=0;j<nx;j++)
			Lx[i]*=xt[j]/xb[j];
	}
	for(i=0;i<ny;i++)
	{
		for(j=0;j<ny;j++)
		{
			yb[j]=Y[j]-Y[i];
			yt[j]=Y[j]-y0;
		}
		yb[i]=1.0;
		yt[i]=1.0;
		Ly[i]=1.0;
		for(j=0;j<ny;j++)
			Ly[i]*=yt[j]/yb[j];
	}
	for(i=0;i<nz;i++)
	{
		for(j=0;j<nz;j++)
		{
			zb[j]=Z[j]-Z[i];
			zt[j]=Z[j]-z0;
		}
		zb[i]=1.0;
		zt[i]=1.0;
		Lz[i]=1.0;
		for(j=0;j<nz;j++)
			Lz[i]*=zt[j]/zb[j];
	}
	f0 = 0.0;
	for(i=0;i<nx;i++)
		for(j=0;j<ny;j++)
			for(k=0;k<nz;k++)
				f0+=Lx[i]*Ly[j]*Lz[k]*F[i][j][k];
	return f0;
}

Real mathfunc::distanceP2L(Real x0, Real y0, Real *p1, Real *p2)
{
	Real dist, A, B, C;
	//line:Ax+By+C=0
	A=p2[1]-p1[1];
	B=p2[0]-p1[0];
	C=1.0*p1[1]*p2[0]-1.0*p1[0]*p2[1];
	dist= ABS( (A*x0+B*y0+C)/sqrt(A*A+B*B) );
	return dist;
}
Real mathfunc::distanceP2S(Real x0, Real y0, Real z0, Real *p1, Real *p2, Real *p3)
{
	Real dist, AB[3], AC[3], P[3], D;
	//plane:Ax+By+Cz+D=0

        AB[2]=p2[2]-p1[2];
	AB[1]=p2[1]-p1[1];
	AB[0]=p2[0]-p1[0];
	AC[2]=p3[2]-p1[2];
	AC[1]=p3[1]-p1[1];
	AC[0]=p3[0]-p1[0];
	//AB and AC is two vector used to form the plane

	this->crossproduct(AB,AC,P);//P normal vector of the plane
	D=-1.0*this->dotproduct(P,p1,3); //点法式
	dist= ABS( (P[0]*x0+P[1]*y0+P[2]*z0+D)/sqrt(this->dotproduct(P,P,3)) );
	return dist;
}


//------------------ source time functon ---------------------------------------
/* Original: Name; Derivative: name; Integral: NAME */
Real mathfunc::Gauss(Real x, Real a, Real x0)
{
  Real y;
  if(ABS(x0) >= SeisZero && (x <= 0.0 || x >= 2.0*x0))
    y = 0.0;
  else
    y = exp( - (x - x0)*(x - x0)/(a*a))/(sqrt(PI)*a);
  return y;
}
Real mathfunc::gauss(Real x, Real a, Real x0)
{
  Real y;
  if(ABS(x0) >= SeisZero && (x <= 0.0 || x >= 2.0*x0))
    y = 0.0;
  else
    y = exp( - (x - x0)*(x - x0)/(a*a))/(sqrt(PI)*a) * ( - 2.0*(x - x0)/(a*a));
  return y;
}
Real mathfunc::Ricker(Real x, Real fc, Real x0)
{
  Real u, y;
  if(x <= 0.0)
    y = 0.0;
  else
  {
    u = (x - x0)*2.0*PI*fc;
    y = (u*u/4.0 - 0.5)*exp( - u*u/4.0)*sqrt(PI)/2.0;
  }
  return y;
}
Real mathfunc::ricker(Real x, Real fc, Real x0)
{
  Real u, y;
  if(x <= 0.0)
    y = 0.0;
  else
  {
    u = (x - x0)*2.0*PI*fc;
    y = u*(1.5 - u*u/4.0)*exp( - u*u/4.0)*sqrt(PI)/2.0*PI*fc;
  }
  return y;
}
Real mathfunc::Bell(Real x, Real xr)
{
  Real y;
  if(x > 0.0 && x < xr)
    y = (1.0 - cos(2.0*PI*x/xr))/xr;
  else
    y = 0.0;
  return y;
}
Real mathfunc::bell(Real x, Real xr)
{
  Real y;
  if(x > 0.0 && x < xr)
    y = 2.0*PI*sin(2.0*PI*x/xr)/xr;
  else
    y = 0.0;
  return y;
}
Real mathfunc::BELL(Real x, Real xr)
{
  Real y;
  if(x <= 0.0)
    y = 0.0;
  else if(x < xr)
    y = x/xr - sin(2.0*PI*x/xr)/(2.0*PI);
  else
    y = 1.0;
  return y;
}
Real mathfunc::Triangle(Real x, Real xr)
{
  Real y;
  if(x > xr)
    y = 0.0;
  else if(x > xr/2.0)
    y = 2.0/(xr/2.0) - x/(xr/2.0*xr/2.0);
  else if(x > 0.0)
    y = x/(xr/2.0*xr/2.0);
  else
    y = 0.0;
  return y;
}
Real mathfunc::TRIANGLE(Real x, Real xr)
{
  Real y;
  if(x > xr)
    y = 1.0;
  else if(x > xr/2.0)
    y = - 0.5*x*x/(xr/2.0*xr/2.0) + 2.0*x/(xr/2.0) - 1.0;
  else if(x > 0.0)
    y = 0.5*x*x/(xr/2.0*xr/2.0);
  else
    y = 0.0;
  return y;
}
Real mathfunc::Bshift(Real x, Real xr, Real x0)
{
  Real u, y;
  if(x > 0.0 && x < xr)
  {
    u = cosh((x - xr/2.0)/x0);
    y = 0.5/x0/u/u;
  }
  else
    y = 0.0;
  return y;
}
Real mathfunc::Step(Real x)
{
  Real y;
  if(x > 0.0)
    y = 1.0;
  else
    y = 0.0;
  return y;
}
Real mathfunc::Delta(Real t, Real t0)
{
	Real v;
	if(t>=0.0 && t<t0)
		v=1.0/t0;
	else
		v=0.0;
	return v;
}
//------------------------------------------------------------------------------
	



























