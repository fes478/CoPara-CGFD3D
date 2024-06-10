#include "typenew.h"
#include<math.h>

using namespace std;
using namespace defstruct;
using namespace constant;

#define errprt(...) com.errorprint(__FILE__, __LINE__, __VA_ARGS__)

//-------------------private--------------------------------
void mediapar::getConf(const char *filename)
{
	char parpath[SeisStrLen];
	char name[SeisStrLen2];
	
	FILE *fp;
	fp=fopen(filename,"r");//SeisFD3D
	if(!fp)
	{
		char errstr[SeisStrLen];
		sprintf(errstr,"fail to open grid conf file %s",filename);
		errprt(Fail2Open,errstr);
	}

	com.get_conf(fp, "seispath", 3, parpath);
	com.get_conf(fp, "media_type", 3, mediatype);
	com.get_conf(fp, "media_filename", 3, name);
	com.get_conf(fp, "threshold_velocity_jump_of_interface", 3, &crit_value);
	com.get_conf(fp, "threshold_percent_change_of_interface", 3, &crit_perc);
	com.get_conf(fp, "sampling_point_per_cell", 3, &sampx);
	com.get_conf(fp, "sampling_point_per_cell", 4, &sampy);
	com.get_conf(fp, "sampling_point_per_cell", 5, &sampz);

	if(ISEQSTR(mediatype,"interface3D"))
	{
		com.get_conf(fp, "interface3dpar", 3, &i3d.nx_in);
		com.get_conf(fp, "interface3dpar", 4, &i3d.ny_in);
		com.get_conf(fp, "interface3dpar", 5, &i3d.x0_in);
		com.get_conf(fp, "interface3dpar", 6, &i3d.y0_in);
		com.get_conf(fp, "interface3dpar", 7, &i3d.dx_in);
		com.get_conf(fp, "interface3dpar", 8, &i3d.dy_in);
		com.get_conf(fp, "interface3dpar", 9, &i3d.nInterface);
	}

	fclose(fp);

	fprintf(stdout,"the input media model is %s in %s type, the sampling points is %d,%d,%d\n",mediatype,name,sampx,sampy,sampz);

	sprintf(mediafile,"%s/%s",parpath,name);
}

void mediapar::read_interface3D(cindx cdx, coord crd)
{
	int i,j,k,m,n;
	char errstr[SeisStrLen];

	fprintf(stdout,"***Start to read the Interface-type media file %s in the media program\n",mediafile);

	FILE *fp;
	fp = fopen(mediafile,"r");
	if(!fp)
	{
		sprintf(errstr,"fail to open media interface file %s",mediafile);
		errprt(Fail2Open,errstr);
	}
	
	printf("%d %d %d %d %d %d %d\n",i3d.nx_in,i3d.ny_in,i3d.x0_in,i3d.y0_in,i3d.dx_in,i3d.dy_in,i3d.nInterface);
	
	
	for(n=0;n<i3d.nInterface;n++)
		for(j=0;j<i3d.ny_in;j++)
			for(i=0;i<i3d.nx_in;i++)
				fscanf(fp, Rformat, &i3d.Zpos[i][j][n]);

	for(n=0;n<i3d.nInterface;n++)
		for(m=0;m<2;m++)
			for(j=0;j<i3d.ny_in;j++)
				for(i=0;i<i3d.nx_in;i++)
					fscanf(fp, Rformat, &i3d.Vp[m][i][j][n]);

	for(n=0;n<i3d.nInterface;n++)
		for(m=0;m<2;m++)
			for(j=0;j<i3d.ny_in;j++)
				for(i=0;i<i3d.nx_in;i++)
					fscanf(fp, Rformat, &i3d.Vs[m][i][j][n]);
	
	for(n=0;n<i3d.nInterface;n++)
		for(m=0;m<2;m++)
			for(j=0;j<i3d.ny_in;j++)
				for(i=0;i<i3d.nx_in;i++)
					fscanf(fp, Rformat, &i3d.Dc[m][i][j][n]);
	
	fclose(fp);
	
	//confirm Z for index 0 to Max is monotonously decrease
	//Z[0] = 8848m, Z[i3d.nInterface-1] = -630KM
	for(i=0;i<i3d.nx_in;i++)
		for(j=0;j<i3d.ny_in;j++)
			for(n=0;n<i3d.nInterface-1;n++)
				if(i3d.Zpos[i][j][n]<i3d.Zpos[i][j][n+1])
				{
					sprintf(errstr,"at location (%d,%d,%d) interface depth[%d]=%lf is lower than depth[%d]=%lf\n",
						i,j,n,n,i3d.Zpos[i][j][n],n+1,i3d.Zpos[i][j][n+1]);
					errprt(Fail2Check,errstr);
				}

	for(m=0;m<2;m++)
		for(i=0;i<i3d.nx_in;i++)
			for(j=0;j<i3d.ny_in;j++)
				for(n=0;n<i3d.nInterface;n++)
				{
					if(n==0 && m==0 && i3d.Vp[m][i][j][n]==0 && i3d.Vs[m][i][j][n]==0)
					{
						// beyond topo is air
					}
					else if( pow(i3d.Vp[m][i][j][n],2) <= 2.0*pow(i3d.Vs[m][i][j][n],2) )
					{
						sprintf(errstr,"at location (%d,%d,%d,%d) there is a velocity error i3d.Vp = %lf and i3d.Vs = %lf\n\n",
								m,i,j,n,i3d.Vp[m][i][j][n],i3d.Vs[m][i][j][n]);
						errprt(Fail2Check,errstr);
					}
				}

	//set other pars
	//mdpX and mdpY
	for(i=0;i<i3d.nx_in;i++)
		i3d.Xpos[i] = i3d.x0_in + i*i3d.dx_in;
	for(j=0;j<i3d.ny_in;j++)
		i3d.Ypos[j] = i3d.y0_in + j*i3d.dy_in;
	
	//i3d.interface_flat
	Real tempz;
	for(n=0;n<i3d.nInterface;n++)
	{
		i3d.interface_flat[n] = true;
		tempz = i3d.Zpos[0][0][n];
		for(i=0;i<i3d.nx_in;i++)
			for(j=0;j<i3d.ny_in;j++)
			{
				if( ABS(tempz - i3d.Zpos[i][j][n]) > SeisZero )
					i3d.interface_flat[n] = false;
			}
	}

	//i3d.interface_const
	Real vp,vs,dp;
	for(n=0;n<i3d.nInterface;n++)
		for(m=0;m<2;m++)
		{
			i3d.interface_const[n][m] = true;
			vp = i3d.Vp[m][0][0][n];
			vs = i3d.Vs[m][0][0][n];
			dp = i3d.Dc[m][0][0][n];

			for(i=0;i<i3d.nx_in;i++)
				for(j=0;j<i3d.ny_in;j++)
				{
					if( ABS(vp-i3d.Vp[m][i][j][n]) > SeisZero || ABS(vs-i3d.Vs[m][i][j][n]) > SeisZero || ABS(dp-i3d.Dc[m][i][j][n])>SeisZero )
						i3d.interface_const[n][m] = false;
				}
		}

	//i3d.layer_const
	for(n=0;n<i3d.nInterface;n++)
	{
		if( n==i3d.nInterface-1 && i3d.interface_const[n][1] )
			i3d.layer_const[n] = true; // the last layer, from last interface to geocentric
		else if( i3d.interface_const[n][1] && i3d.interface_const[n+1][0] &&
			 ABS(i3d.Vp[1][0][0][n]-i3d.Vp[0][0][0][n+1])<SeisZero    &&
			 ABS(i3d.Vs[1][0][0][n]-i3d.Vs[0][0][0][n+1])<SeisZero    &&
			 ABS(i3d.Dc[1][0][0][n]-i3d.Dc[0][0][0][n+1])<SeisZero )
			i3d.layer_const[n] = true; // the difference between this interface's under side value and the next interface's upper side value
		else
			i3d.layer_const[n] = false;
	}

	//i3d.layerZrange
	Real Zmax,Zmin;
	for(n=0;n<i3d.nInterface-1;n++)
	{
		Zmax = i3d.Zpos[0][0][n];
		Zmin = i3d.Zpos[0][0][n+1];
		for(i=0;i<i3d.nx_in;i++)
			for(j=0;j<i3d.ny_in;j++)
			{
				Zmax = MAX(Zmax, i3d.Zpos[i][j][n]);
				Zmin = MIN(Zmin, i3d.Zpos[i][j][n+1]);
			}
		i3d.layerZrange[n][0]=Zmax;
		i3d.layerZrange[n][1]=Zmin;
	}

	//the last layer(to geocentric)
	n = i3d.nInterface-1;
	Zmax = i3d.Zpos[0][0][n];
	Zmin = Zmax;
	for(i=0;i<i3d.nx_in;i++)
		for(j=0;j<i3d.ny_in;j++)
		{
			Zmax = MAX(Zmax, i3d.Zpos[i][j][n]);
			Zmin = MIN(Zmin, i3d.Zpos[i][j][n]);
		}
	i3d.layerZrange[n][0]=Zmax;
	i3d.layerZrange[n][1]=Zmin-6371E3;

	//i3d.interfaceZrange
	for(n=0;n<i3d.nInterface;n++)
	{
		Zmax = i3d.Zpos[0][0][n];
		Zmin = i3d.Zpos[0][0][n];
		for(i=0;i<i3d.nx_in;i++)
			for(j=0;j<i3d.ny_in;j++)
			{
				Zmax = MAX(Zmax, i3d.Zpos[i][j][n]);
				Zmin = MIN(Zmin, i3d.Zpos[i][j][n]);
			}
		i3d.interfaceZrange[n][0]=Zmax;
		i3d.interfaceZrange[n][1]=Zmin;
	}

	//interface coincide
	Real h;
	for(i=0;i<i3d.nx_in-1;i++)
		for(j=0;j<i3d.ny_in-1;j++)
		{
			n = i3d.nInterface-1;
			i3d.indexZ[n][i][j][1] = n+1;

			for(n=i3d.nInterface-2;n>=0;n--)
			{
				h = 0.0;
				h = MAX(h, ABS(i3d.Zpos[i][j][n]-i3d.Zpos[i][j][n+1]) );
				h = MAX(h, ABS(i3d.Zpos[i+1][j][n]-i3d.Zpos[i+1][j][n+1]) );
				h = MAX(h, ABS(i3d.Zpos[i][j+1][n]-i3d.Zpos[i][j+1][n+1]) );
				h = MAX(h, ABS(i3d.Zpos[i+1][j+1][n]-i3d.Zpos[i+1][j+1][n+1]) );

				//four point to confirm coincide;
				if( h<SeisZero )
					i3d.indexZ[n][i][j][1] = i3d.indexZ[n+1][i][j][1];
				else
					i3d.indexZ[n][i][j][1] = n+1;

			}

			n = 0;
			//upper side of the top interface
			if( i3d.interface_const[0][0] && i3d.Vp[0][0][0][0]==0 )
				i3d.indexZ[n][i][j][0] = -i3d.indexZ[n][i][j][1];
			else
				i3d.indexZ[n][i][j][0] = n+1;

			for(n=1;n<i3d.nInterface;n++)
			{
				h = 0.0;
				h = MAX(h, ABS(i3d.Zpos[i][j][n]-i3d.Zpos[i][j][n-1]) );
				h = MAX(h, ABS(i3d.Zpos[i+1][j][n]-i3d.Zpos[i+1][j][n-1]) );
				h = MAX(h, ABS(i3d.Zpos[i][j+1][n]-i3d.Zpos[i][j+1][n-1]) );
				h = MAX(h, ABS(i3d.Zpos[i+1][j+1][n]-i3d.Zpos[i+1][j+1][n-1]) );

				if(h<SeisZero)
					i3d.indexZ[n][i][j][0] = i3d.indexZ[n-1][i][j][0];
				else
					i3d.indexZ[n][i][j][0] = n+1;

			}

		}

	j = i3d.ny_in-1;
	for(i=0;i<i3d.nx_in;i++)
		for(n=0;n<i3d.nInterface;n++)
		{
			i3d.indexZ[n][i][j][0] = i3d.indexZ[n][i][j-1][0];
			i3d.indexZ[n][i][j][1] = i3d.indexZ[n][i][j-1][1];
		}

	i = i3d.nx_in-1;
	for(j=0;j<i3d.ny_in;j++)
		for(n=0;n<i3d.nInterface;n++)
		{
			i3d.indexZ[n][i][j][0] = i3d.indexZ[n][i-1][j][0];
			i3d.indexZ[n][i][j][1] = i3d.indexZ[n][i-1][j][1];
		}

	fprintf(stdout,"Input Z-direction Position information\n");
	for(n=0;n<i3d.nInterface;n++)
	{
		Zmax = i3d.Zpos[0][0][n];
		Zmin = Zmax;
		for(i=0;i<i3d.nx_in;i++)
			for(j=0;j<i3d.ny_in;j++)
			{
				Zmax = MAX(Zmax,i3d.Zpos[i][j][n]);
				Zmin = MIN(Zmin,i3d.Zpos[i][j][n]);
			}
		fprintf(stdout,"\tinterface[%d] i3d.Zpos[0][0][n]=%lf,i3d.Zpos[nx][ny][n]=%lf,Zmin=%lf,Zmax=%lf\n",n,i3d.Zpos[0][0][n],i3d.Zpos[i3d.nx_in-1][i3d.ny_in-1][n],Zmin,Zmax);
	}

	fprintf(stdout,"Input i3d.Vp information\n");
	for(n=0;n<i3d.nInterface;n++)
		for(m=0;m<2;m++)
		{
			Zmax = i3d.Vp[m][0][0][n];
			Zmin = Zmax;
			for(i=0;i<i3d.nx_in;i++)
				for(j=0;j<i3d.ny_in;j++)
				{
					Zmax = MAX(Zmax,i3d.Vp[m][i][j][n]);
					Zmin = MIN(Zmin,i3d.Vp[m][i][j][n]);
				}
			fprintf(stdout,"\tinterface[%d].Side[%d] i3d.Vp[m][0][0][n]=%lf,i3d.Vp[m][nx][ny][n]=%lf,i3d.Vpmin=%lf,i3d.Vpmax=%lf\n",
				n,m,i3d.Vp[m][0][0][n],i3d.Vp[m][i3d.nx_in-1][i3d.ny_in-1][n],Zmin,Zmax);
		}

	fprintf(stdout,"Input Vs information\n");
	for(n=0;n<i3d.nInterface;n++)
		for(m=0;m<2;m++)
		{
			Zmax = i3d.Vs[m][0][0][n];
			Zmin = Zmax;
			for(i=0;i<i3d.nx_in;i++)
				for(j=0;j<i3d.ny_in;j++)
				{
					Zmax = MAX(Zmax,i3d.Vs[m][i][j][n]);
					Zmin = MIN(Zmin,i3d.Vs[m][i][j][n]);
				}
			fprintf(stdout,"\tinterface[%d].Side[%d] Vs[m][0][0][n]=%lf,Vs[m][nx][ny][n]=%lf,Vsmin=%lf,Vsmax=%lf\n",
				n,m,i3d.Vs[m][0][0][n],i3d.Vs[m][i3d.nx_in-1][i3d.ny_in-1][n],Zmin,Zmax);
		}

	fprintf(stdout,"Input Density information\n");
	for(n=0;n<i3d.nInterface;n++)
		for(m=0;m<2;m++)
		{
			Zmax = i3d.Dc[m][0][0][n];
			Zmin = Zmax;
			for(i=0;i<i3d.nx_in;i++)
				for(j=0;j<i3d.ny_in;j++)
				{
					Zmax = MAX(Zmax,i3d.Dc[m][i][j][n]);
					Zmin = MIN(Zmin,i3d.Dc[m][i][j][n]);
				}
			fprintf(stdout,"\tinterface[%d].Side[%d] Dc[m][0][0][n]=%lf,Dc[m][nx][ny][n]=%lf,Dcmin=%lf,Dcmax=%lf\n",
				n,m,i3d.Dc[m][0][0][n],i3d.Dc[m][i3d.nx_in-1][i3d.ny_in-1][n],Zmin,Zmax);
		}

	fprintf(stdout,"Confirm interface flat information\n");
	for(n=0;n<i3d.nInterface;n++)
		fprintf(stdout,"\tinterface[%d] flat=%d\n",n,i3d.interface_flat[n]);

	fprintf(stdout,"Confirm interface constant information\n");
	for(n=0;n<i3d.nInterface;n++)
		for(m=0;m<2;m++)
			fprintf(stdout,"\tinterface[%d].Side[%d] constant=%d\n",n,m,i3d.interface_const[n][m]);
	
	fprintf(stdout,"Confirm interface range information\n");
	for(n=0;n<i3d.nInterface;n++)
		fprintf(stdout,"\tinterface[%d] range from %lf to %lf\n",n,i3d.interfaceZrange[n][0],i3d.interfaceZrange[n][1]);
	
	fprintf(stdout,"Confirm layer constant information\n");
	for(n=0;n<i3d.nInterface;n++)
		fprintf(stdout,"\tlayer[%d] flat=%d\n",n,i3d.layer_const[n]);
	
	fprintf(stdout,"Confirm layer range information\n");
	for(n=0;n<i3d.nInterface;n++)
		fprintf(stdout,"\tlayer[%d] range from %lf to %lf\n",n,i3d.layerZrange[n][0],i3d.layerZrange[n][1]);
	
	fprintf(stdout,"Confirm interface coincide information(fixed point (0,0) )\n");
	for(n=0;n<i3d.nInterface;n++)
		fprintf(stdout,"\tat interface[%d] point(0,0) the upper side i3d.indexZ=%3d, the under side i3d.indexZ=%3d\n",n,i3d.indexZ[n][0][0][0],i3d.indexZ[n][0][0][1]);
	

	fprintf(stdout,"---accomplished reading the 3Dinterface parameters in media porgram\n");
}

void mediapar::volume_discrete(Real x0, Real y0, Real z0, Real *Vp2, Real *Vs2, Real *Dc2)	
{
	int i,j,k;
	int i1,i2,j1,j2,k1,k2;
	Real x1,y1,z1;
	
	mathf.LocValue1d(x0, Vnc.Xpos, Vnc.ni, &i1, &i2, &x1);
	mathf.LocValue1d(y0, Vnc.Ypos, Vnc.nj, &j1, &j2, &y1);
	mathf.LocValue1d(z0, Vnc.Zpos, Vnc.nk, &k1, &k2, &z1);

	//printf("input value position is %f, %f, %f\n",x0,y0,z0);
	//printf("get location i = %d, %d, %f\n",i1,i2,x1);
	//printf("get location j = %d, %d, %f\n",j1,j2,y1);
	//printf("get location k = %d, %d, %f\n",k1,k2,z1);

	Real *XP,*YP,*ZP,***C;
	int nx,ny,nz;
	Real value;
	
	nx = i2-i1+1; ny = j2-j1+1; nz = k2-k1+1;
	XP = new Real[nx]; YP = new Real[ny]; ZP = new Real[nz];

	C = new Real **[nx];
	for(i=0;i<nx;i++)
	{
		C[i] = new Real *[ny];
		for(j=0;j<ny;j++)
			C[i][j] = new Real[nz];
	}

	for(i=0;i<nx;i++)
		XP[i] = Vnc.Xpos[i1+i];
	for(j=0;j<ny;j++)
		YP[j] = Vnc.Ypos[j1+j];
	for(k=0;k<nz;k++)
		ZP[k] = Vnc.Zpos[k1+k];
	
	//printf("start to interp3d\n");

	//interpolate VP
	for(i=0;i<nx;i++)
		for(j=0;j<ny;j++)
			for(k=0;k<nz;k++)
				C[i][j][k] = Vnc.vp[i1+i][j1+j][k1+k];
	value = mathf.interp3d(XP,YP,ZP,C,2,2,2,x1,y1,z1);
	Vp2[0] = value; Vp2[1] = value;
	
	//interpolate VS
	for(i=0;i<nx;i++)
		for(j=0;j<ny;j++)
			for(k=0;k<nz;k++)
				C[i][j][k] = Vnc.vs[i1+i][j1+j][k1+k];
	value = mathf.interp3d(XP,YP,ZP,C,2,2,2,x1,y1,z1);
	Vs2[0] = value; Vs2[1] = value;

	//interpolate DC
	for(i=0;i<nx;i++)
		for(j=0;j<ny;j++)
			for(k=0;k<nz;k++)
				C[i][j][k] = Vnc.den[i1+i][j1+j][k1+k];
	value = mathf.interp3d(XP,YP,ZP,C,2,2,2,x1,y1,z1);
	Dc2[0] = value; Dc2[1] = value;


}

void mediapar::interface3d_discrete(Real x0, Real y0, Real z0, Real *Vp2, Real *Vs2, Real *Dc2)	
{
	int i,j,k,m,n;
	int i1,i2,j1,j2;
	int n1,n2,side;
	Real z1,z2,L,L1,L2,x1,y1;
	bool found;
	
	//calculate the effective media parameters, map into coords
	Real **ziI;
	ziI = new Real *[2];
	for(i=0;i<2;i++)
		ziI[i] = new Real[2];
	
	mathf.LocValue1d(x0, i3d.Xpos, i3d.nx_in, &i1, &i2, &x1);
	mathf.LocValue1d(y0, i3d.Ypos, i3d.ny_in, &j1, &j2, &y1);

	found = false;
	z1 = -7000E3,z2=-7000e3;
	n1 = i3d.nInterface+1;
	n2 = -1;

	//confirm layer location
	for(n=0;n<i3d.nInterface;n++)
	{
		if( z0 >= i3d.layerZrange[n][1] && z0 <= i3d.layerZrange[n][0] )
		{
			n1 = MIN(n1,n);
			n2 = MAX(n2,n+1);
		}
	}
	if(n2>=i3d.nInterface)
		n2=i3d.nInterface-1;
	
	//above the free surface
	if(n1>n2)
	{
		k = i3d.indexZ[0][i1][j1][0];
		side = 0;
		if(k<0)
		{
			side = 1;
			k = -k;
		}
		k = k-1;//adjust the indexZ from display-use to index-use
		
		if(i3d.interface_const[k][side])
		{
			Vp2[0] = i3d.Vp[side][i1][j1][k];
			Vs2[0] = i3d.Vs[side][i1][j1][k];
			Dc2[0] = i3d.Dc[side][i1][j1][k];
		}
		else
		{
			for(i=0;i<2;i++)
				for(j=0;j<2;j++)
					ziI[i][j] = i3d.Vp[side][i+i1][j+j1][k];
			Vp2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

			for(i=0;i<2;i++)
				for(j=0;j<2;j++)
					ziI[i][j] = i3d.Vs[side][i+i1][j+j1][k];
			Vs2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

			for(i=0;i<2;i++)
				for(j=0;j<2;j++)
					ziI[i][j] = i3d.Dc[side][i+i1][j+j1][k];
			Dc2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);
		}

		Vp2[1] = Vp2[0];
		Vs2[1] = Vs2[0];
		Dc2[1] = Dc2[0];

		found = true;
	}
	else
	{
		for(n=n1; n<=n2; n++)
		{
			// zero-thickness layer(point type)
			if( i3d.indexZ[n][i1][j1][1] != n+1 )
				continue;

			// upper interface
			if(i3d.interface_flat[n])
			{
				z1 = i3d.Zpos[i1][j1][n];
			}
			else
			{
				for(i=0;i<2;i++)
					for(j=0;j<2;j++)
						ziI[i][j] = i3d.Zpos[i+i1][j+j1][n];
				z1 = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);
			}

			// near the upper interface
			if( ABS(z0-z1)<SeisZero )
			{
				//upper side
				k = i3d.indexZ[n][i1][j1][0];
				side = 0;
				if(k<0)
				{
					side = 1;
					k = -k;
				}
				k = k-1;//adjust the indexZ from display-use to index-use
				
				if(i3d.interface_const[k][side])
				{
					Vp2[0] = i3d.Vp[side][i1][j1][k];
					Vs2[0] = i3d.Vs[side][i1][j1][k];
					Dc2[0] = i3d.Dc[side][i1][j1][k];
				}
				else
				{
					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vp[side][i+i1][j+j1][k];
					Vp2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vs[side][i+i1][j+j1][k];
					Vs2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Dc[side][i+i1][j+j1][k];
					Dc2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);
				}

				//lower side
				k = i3d.indexZ[n][i1][j1][1];
				k = k-1;//adjust the indexZ from display-use to index-use
				
				if(i3d.interface_const[k][1])
				{
					Vp2[1] = i3d.Vp[1][i1][j1][k];
					Vs2[1] = i3d.Vs[1][i1][j1][k];
					Dc2[1] = i3d.Dc[1][i1][j1][k];
				}
				else
				{
					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vp[1][i+i1][j+j1][k];
					Vp2[1] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vs[1][i+i1][j+j1][k];
					Vs2[1] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Dc[1][i+i1][j+j1][k];
					Dc2[1] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);
				}

				found = true;
				break;
			}
			else if(z0 > z1)//above the upper interface
			{
				k = i3d.indexZ[n][i1][j1][0];
				side = 0;
				if(k<0)
				{
					side = 1;
					k = -k;
				}
				k = k-1;//adjust the indexZ from display-use to index-use

				if(i3d.interface_const[k][side])
				{
					Vp2[0] = i3d.Vp[side][i1][j1][k];
					Vs2[0] = i3d.Vs[side][i1][j1][k];
					Dc2[0] = i3d.Dc[side][i1][j1][k];
				}
				else
				{
					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vp[side][i+i1][j+j1][k];
					Vp2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vs[side][i+i1][j+j1][k];
					Vs2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Dc[side][i+i1][j+j1][k];
					Dc2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);
				}

				Vp2[1] = Vp2[0];
				Vs2[1] = Vs2[0];
				Dc2[1] = Dc2[0];

				found = true;
				break;
			}

			//lower interface
			//Last layer to geocentric
			if(n == i3d.nInterface-1)
			{
				k = i3d.indexZ[n][i1][j1][1];
				side = 1;
				k = k-1;//adjust the indexZ from display-use to index-use

				if(i3d.interface_const[k][side])
				{
					Vp2[0] = i3d.Vp[side][i1][j1][k];
					Vs2[0] = i3d.Vs[side][i1][j1][k];
					Dc2[0] = i3d.Dc[side][i1][j1][k];
				}
				else
				{
					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vp[side][i+i1][j+j1][k];
					Vp2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vs[side][i+i1][j+j1][k];
					Vs2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Dc[side][i+i1][j+j1][k];
					Dc2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);
				}

				Vp2[1] = Vp2[0];
				Vs2[1] = Vs2[0];
				Dc2[1] = Dc2[0];

				found = true;
				break;
			}
			
			//lower interface( n1<n<n2 )
			if(i3d.interface_flat[n+1])
				z2 = i3d.Zpos[i1][j1][n+1];
			else
			{
				for(i=0;i<2;i++)
					for(j=0;j<2;j++)
						ziI[i][j] = i3d.Zpos[i+i1][j+j1][n+1];
				z2 = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);
			}

			//near the lower interface
			if( ABS(z0-z2) < SeisZero )
			{
				k = i3d.indexZ[n+1][i1][j1][0];
				side = 0;
				if(k<0)
				{
					side = 1;
					k = -k;
				}
				k = k-1;//adjust the indexZ from display-use to index-use

				if(i3d.interface_const[k][side])
				{
					Vp2[0] = i3d.Vp[side][i1][j1][k];
					Vs2[0] = i3d.Vs[side][i1][j1][k];
					Dc2[0] = i3d.Dc[side][i1][j1][k];
				}
				else
				{
					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vp[side][i+i1][j+j1][k];
					Vp2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vs[side][i+i1][j+j1][k];
					Vs2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Dc[side][i+i1][j+j1][k];
					Dc2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);
				}

				k = i3d.indexZ[n+1][i1][j1][1];
				k = k-1;//adjust the indexZ from display-use to index-use

				if(i3d.interface_const[k][1])
				{
					Vp2[1] = i3d.Vp[1][i1][j1][k];
					Vs2[1] = i3d.Vs[1][i1][j1][k];
					Dc2[1] = i3d.Dc[1][i1][j1][k];
				}
				else
				{
					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vp[1][i+i1][j+j1][k];
					Vp2[1] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vs[1][i+i1][j+j1][k];
					Vs2[1] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Dc[1][i+i1][j+j1][k];
					Dc2[1] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);
				}

				found = true;
				break;
			}
			else if(z0>z2) //above the lower interface
			{
				if(i3d.layer_const[n])
				{
					Vp2[0] = i3d.Vp[1][i1][j1][n];
					Vs2[0] = i3d.Vs[1][i1][j1][n];
					Dc2[0] = i3d.Dc[1][i1][j1][n];
					Vp2[1] = Vp2[0];
					Vs2[1] = Vs2[0];
					Dc2[1] = Dc2[0];
				}
				else
				{
					L = z1-z2;
					L1 = (z0-z2)/L;
					L2 = 1-L1;
					
					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vp[1][i+i1][j+j1][n];
					Vp2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vs[1][i+i1][j+j1][n];
					Vs2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Dc[1][i+i1][j+j1][n];
					Dc2[0] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);
					
					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vp[0][i+i1][j+j1][n+1];
					Vp2[1] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Vs[0][i+i1][j+j1][n+1];
					Vs2[1] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					for(i=0;i<2;i++)
						for(j=0;j<2;j++)
							ziI[i][j] = i3d.Dc[0][i+i1][j+j1][n+1];
					Dc2[1] = mathf.interp2d(&i3d.Xpos[i1],&i3d.Ypos[j1],ziI,2,2,x1,y1);

					Vp2[0] = Vp2[0]*L1+Vp2[1]*L2;
					Vs2[0] = Vs2[0]*L1+Vs2[1]*L2;
					Dc2[0] = Dc2[0]*L1+Dc2[1]*L2;
					Vp2[1] = Vp2[0];
					Vs2[1] = Vs2[0];
					Dc2[1] = Dc2[0];
				}

				found = true;
				break;
			}

		}//end for loop
	}//end free surface

	if(!found)
		fprintf(stdout,"at point position (%g,%g,%g) find no proper media pars\n",x0,y0,z0);
	
	for(i=0;i<2;i++)
		delete [] ziI[i];
	delete [] ziI;
	
}

void mediapar::TwoinOne(Real *Vp, Real *Vs, Real *Dc, Real *lamval, Real *muval, Real *rhoval, int *nsamp)
{
	int i;
	Real lam[2],miu[2];
	/*
	   printf("2in1----->vp2el=%f,%f\n",Vp[0],Vp[1]);
	   printf("2in1----->Vs2el=%f,%f\n",Vs[0],Vs[1]);
	   printf("2in1----->Dc2el=%f,%f\n",Dc[0],Dc[1]);
	*/
	for(i=0;i<2;i++)
	{
		miu[i] = Dc[i]*Vs[i]*Vs[i];
		lam[i] = Vp[i]*Vp[i]*Dc[i]-2.0*miu[i]; 
		//printf("Vp = %g , Vs = %g , Dc = %g , miu = %g , lam = %g \n",Vp[i],Vs[i],Dc[i],miu[i],lam[i]);
	}

	*rhoval = *rhoval + 0.5*(Dc[0]+Dc[1]);

	if( (nsamp>0 && muval==0) || (miu[0]<SeisZero || miu[1]<SeisZero) )
		*muval = 0;
	else
		*muval = *muval + 0.5/miu[0] + 0.5/miu[1];
	
	*lamval = *lamval + 0.5/lam[0] + 0.5/lam[1];

	/*
	   printf("rho=%e ",*rhoval);
	   printf("mu=%e ",*muval);
	   printf("lam=%e\n",*lamval);
	*/
	*nsamp = *nsamp+1;

}

void mediapar::get2in1(int pi, int pj, int pk, Real *Vp, Real *Vs, Real *Dc)
{
	int i;
	Real lam[2],mu[2];
	for(i=0;i<2;i++)
	{
		mu[i] = Dc[i]*Vs[i]*Vs[i];
		lam[i] = Vp[i]*Vp[i]*Dc[i]-2.0*mu[i];
	}

	density[pi][pj][pk] = (Dc[0]+Dc[1])/2.0;
	miu[pi][pj][pk] = (mu[0]+mu[1])/2.0;
	lambda[pi][pj][pk] = (lam[0]+lam[1])/2.0;

}

void mediapar::average(int i, int j, int k, Real lamval, Real muval, Real rhoval, int nsamp)
{
	double rhoD, muD, lamD;

	rhoD = (double)rhoval;
	muD = (double)muval;
	lamD = (double)lamval;

	density[i][j][k] = (Real) rhoD/nsamp;
	muval>0 ? miu[i][j][k] = (Real) nsamp*1.0/muD : miu[i][j][k] = 0;
	lambda[i][j][k] = (Real) nsamp*1.0/lamD;
	
	//density[i][j][k] = rhoval/nsamp;
	//muval>0 ? miu[i][j][k] = nsamp/muval : miu[i][j][k] = 0;
	//lambda[i][j][k] = nsamp/lamval;
	
	//printf("lamval=%e, muval=%e, rhoval=%e, nsamp=%d\n",lamval,muval,rhoval,nsamp);

	//nancheck
	if( isnan(density[i][j][k]) )
		printf("at average step--->density(%d,%d,%d) is %f\n",i,j,k,density[i][j][k]);
	if( isnan(miu[i][j][k]) )
		printf("at average step--->miu(%d,%d,%d) is %f\n",i,j,k,miu[i][j][k]);
	if( isnan(lambda[i][j][k]) )
		printf("at average step--->lambda(%d,%d,%d) is %f\n",i,j,k,lambda[i][j][k]);

}

void mediapar::set_vel3d(int i, int j, int k, Real *Vp)
{
	if( ABS( Vp[0]-Vp[1] ) < crit_value && ABS( Vp[0]-Vp[1] )/MIN( Vp[0],Vp[1] ) < crit_perc )
		vel3d[i][j][k] = (Vp[0] + Vp[1])/2.0;
	else
		vel3d[i][j][k] = (Vp[0] + Vp[1])/2.0 + 9000E3;

}

void mediapar::read_interface(cindx cdx, coord crd)
{
	int i,j,k;
	char errstr[SeisStrLen];

	fprintf(stdout,"***Start to read the Interface-type media file %s in the media program\n",mediafile);

	FILE *fp;
	fp = fopen(mediafile,"r");
	if(!fp)
	{
		sprintf(errstr,"fail to open media interface file %s",mediafile);
		errprt(Fail2Open,errstr);
	}

	int xsn,ysn,zsn; //which-direction sampling numbers
	Real diszoom,velzoom,denzoom;

	com.get_conf(fp, "distance2meter", 3, &diszoom);
	com.get_conf(fp, "velocity2m/s", 3, &velzoom);
	com.get_conf(fp, "density2kg/m^3", 3, &denzoom);
	com.get_conf(fp, "number_of_interface", 3,&zsn);
	com.get_conf(fp, "horizontal_sampling", 3, &xsn);
	com.get_conf(fp, "horizontal_sampling", 4, &ysn);

	Real tempx,tempy;
	Real px[xsn],py[ysn],pz[xsn][ysn][zsn],ph[xsn][ysn][zsn];
	Real vp[zsn][2],vs[zsn][2],den[zsn][2];
	
	com.setchunk(fp, "<anchor_media>");
	for(k=0;k<zsn;k++)
	{
		fscanf(fp, Rformat Rformat Rformat Rformat Rformat Rformat,
		           &vp[k][0],&vp[k][1],&vs[k][0],&vs[k][1],&den[k][0],&den[k][1]);
		vp[k][0] *= velzoom; vp[k][1] *= velzoom;
		vs[k][0] *= velzoom; vs[k][1] *= velzoom;
		den[k][0] *= denzoom; den[k][1] *= denzoom;
	}
	
	com.setchunk(fp, "<anchor_interface>");
	for(j=0;j<ysn;j++)
	{
		for(i=0;i<xsn;i++)
		{
			fscanf(fp, Rformat, &tempx);
			fscanf(fp, Rformat, &tempy);
			for(k=0;k<zsn;k++)
			{
				fscanf(fp, Rformat, &pz[i][j][k]);
				pz[i][j][k] *= diszoom;
			}
			for(k=0;k<zsn-1;k++)
				ph[i][j][k] = pz[i][j][k] - pz[i][j][k+1];
			if(!j)
				px[i] = tempx*diszoom;
		}
		py[j] = tempy*diszoom;
	}
	fclose(fp);

	cout<<" finish reading media parfile, den[0][1]="<<den[0][1]<<endl;

	//check media parameters Vp,Vs,Density
	for(k=0;k<zsn;k++)
		for(i=0;i<2;i++)
			if( !k && !i )
				if( vp[k][i]<0 || vs[k][i]<0 || den[k][i]<0 )
					errprt(Fail2Check, " the media parameters on the upper side of first interface must be Non-negetive!\n");
			else if( vp[k][i]<=0 || vs[k][i] <=0 || den[k][i]<=0 )
				errprt(Fail2Check, " the media parameters in the inner layers must be positive!\n");
			else if( vp[k][i]*vp[k][i] <= 2.0*vs[k][i]*vs[k][i] )
				errprt(Fail2Check, " the P wave velocity must greater than S wave!\n");
	if(zsn>1)
		for(i=0;i<xsn;i++)
			for(j=0;j<ysn;j++)
				for(k=0;k<zsn-1;k++)
					if(ph[i][j][k]<0.0)
					{
						fprintf(stdout,"the thickness on location Interface(%d,%d,%d) is negetive!\n",i,j,k);
						errprt(Fail2Check," the layer thickness must be positive\n");
					}

	//calculate the effective media parameters, map into coords
	Real smoZdir;
	Real lamda[2],mu[2],velp[2],vels[2],density[2];
	Real lam0,mu0,rho0;//confirm directly from media par
	Real lam1,mu1,rho1;//confirm after smooth
	bool waterflag;
	int nsamp;
	int is,js,ks;//loop for sampling index
	int ic,jc,kc;//loop for 3-value interped coord index
	int ki,kkk;//loop for media interface
	Real xI[3],yI[3],zI[3],***cI;//cI[3][3][3];
	Real locx,locy,locz;// interp3D, physical point location
	Real x0,y0,z0;//interp3D, from physical coord
	Real xnear,ynear;//nearest value from px and py
	int xleft,xright,yleft,yright;//index from behind
	Real z1,h1=0.0;//interp2D, from media's coord
	//Real ziI[2][2],hiI[2][2];//wrong for convert Real[][][] to Real***
	Real **ziI,**hiI;
	int up1,up2,down1,down2;//index for vp[up2][up1],vs,den; 2 for zsn loop, 1 for above&below loop; up for above, down for below
	Real L1,L2;//Z direction interpolation weights
	
	xI[0]=-1.0; xI[1]=0.0; xI[2]=1.0;
	yI[0]=-1.0; yI[1]=0.0; yI[2]=1.0;
	zI[0]=-1.0; zI[1]=0.0; zI[2]=1.0;
	nsamp = MAX(2*sampx,1) * MAX(2*sampy,1) * MAX(2*sampz,1);

	cI = new Real **[3];
	for(i=0;i<3;i++)
	{
		cI[i] = new Real *[3];
		for(j=0;j<3;j++)
			cI[i][j] = new Real[3];
	}
	ziI = new Real *[2];
	hiI = new Real *[2];
	for(i=0;i<2;i++)
	{
		ziI[i] = new Real[2];
		hiI[i] = new Real[2];
	}
	
	for(i=cdx.ni1;i<cdx.ni2;i++)
	{
		for(j=cdx.nj1;j<cdx.nj2;j++)
		{
			for(k=cdx.nk1;k<cdx.nk2;k++)
			{
				smoZdir = (crd.z[i][j][k+1] - crd.z[i][j][k])/2.0/(sampz+1);
				lam0 = 0.0; mu0 = 0.0; rho0 = 0.0;
				waterflag = false;

				for(is=-sampx;is<=sampx;is++)
				{
					if( sampx!=0 && is==0)
						continue;
					for(js=-sampy;js<=sampy;js++)
					{
						if( sampy!=0 && js==0)
							continue;
						for(ks=-sampz;ks<=sampz;ks++)
						{
							if( sampz!=0 && ks==0 )
								continue;
							
							locx = is/(2.0*(sampx+1.0));
							locy = js/(2.0*(sampy+1.0));
							locz = ks/(2.0*(sampz+1.0));
							//interp x0
							for(ic=0;ic<3;ic++)
								for(jc=0;jc<3;jc++)
									for(kc=0;kc<3;kc++)
										cI[ic][jc][kc] = crd.x[i-1+ic][j-1+jc][k-1+kc];
							x0 = mathf.interp3d(xI,yI,zI,cI,3,3,3,locx,locy,locz);
							
							//interp y0
							for(ic=0;ic<3;ic++)
								for(jc=0;jc<3;jc++)
									for(kc=0;kc<3;kc++)
										cI[ic][jc][kc] = crd.y[i-1+ic][j-1+jc][k-1+kc];
							y0 = mathf.interp3d(xI,yI,zI,cI,3,3,3,locx,locy,locz);
							
							//interp z0
							for(ic=0;ic<3;ic++)
								for(jc=0;jc<3;jc++)
									for(kc=0;kc<3;kc++)
										cI[ic][jc][kc] = crd.z[i-1+ic][j-1+jc][k-1+kc];
							z0 = mathf.interp3d(xI,yI,zI,cI,3,3,3,locx,locy,locz);

							//calculate media par in Z dir
							mathf.LocValue1d(x0, px, xsn, &xleft, &xright, &xnear);
							mathf.LocValue1d(y0, py, ysn, &yleft, &yright, &ynear);

							for(ki = zsn-1;ki>=0;ki--)
							{
								for(ic=0;ic<2;ic++)
									for(jc=0;jc<2;jc++)
										ziI[ic][jc] = pz[xleft+ic][yleft+jc][ki];
								z1 = mathf.interp2d(&px[xleft],&py[yleft],ziI,2,2,xnear,ynear);

								if(ki == zsn-1)// confirm the current layer thickness
									h1 = 1.0e3;
								else
								{
									for(ic=0;ic<2;ic++)
										for(jc=0;jc<2;jc++)
											hiI[ic][jc] = ph[xleft+ic][yleft+jc][ki];
									h1 = mathf.interp2d(&px[xleft],&py[yleft],hiI,2,2,xnear,ynear);
								}

								if(h1<=SeisZero)// if the layer is too thin
									if( ki > 0 )
										continue;
									else if(ki==0)//the top layer is thin
									{
										if(vp[ki][0]>SeisZero)//there is velocity pars above this interface
										{
											up1=0; up2=0; down1=0; down2=0;
										}
										velp[0] = vp[up2][up1]; velp[1] = vp[down2][down1];
										vels[0] = vs[up2][up1]; vels[1] = vs[down2][down1];
										density[0] = den[up2][up1]; density[1] = den[down2][down1];
										break;
									}
								
								if( ABS(z1-z0)<= smoZdir/5.0)//depth location is close for media and coord
								{
									down2 = ki; down1 = 1;
									
									if(ki==0)//for the top surface
									{
										up2 = ki; up1 = 0; 
									}
									else
									{
										up2 = ki; up1 = 1;//means ignore the mediapar above the this interface
									}
									
									for(kkk=ki-1;kkk>=0;kkk--)
									{
										for(ic=0;ic<2;ic++)
											for(jc=0;jc<2;jc++)
												hiI[ic][jc] = ph[xleft+ic][yleft+jc][kkk];
										h1 = mathf.interp2d(&px[xleft],&py[yleft],hiI,2,2,xnear,ynear);
										if(h1>SeisZero)//thick layer
										{
											up2 = kkk+1; up1 = 0;
											break;
										}
										else if(kkk=0)
											if(vp[kkk][0]>SeisZero)
											{
												up2 = kkk; up1 = 0;
											}
											else
											{
												up2 = ki; up1 = 1;
											}
									}

									velp[0] = vp[up2][up1]; velp[1] = vp[down2][down1];
									vels[0] = vs[up2][up1]; vels[1] = vs[down2][down1];
									density[0] = den[up2][up1]; density[1] = den[down2][down1];
									break;
								}

								if(z0<z1)//coord point below the media point, physical point is inside the media layer
								{
									if(ki==zsn-1)//for bottom interface
									{
										velp[0] = vp[ki][1]; velp[1] = vp[ki][1];
										vels[0] = vs[ki][1]; vels[1] = vs[ki][1];
										density[0] = den[ki][1]; density[1] = den[ki][1];
										//due to this point is already inside the media layer
										//so media pars for this point(up and down) will all
										//equal to this layer interface's down value
									}
									else
									{
										L2 = (z1-z0)/h1;
										L1 = 1.0-L2;
										velp[0] = vp[ki][1]*L1 + vp[ki+1][0]*L2; velp[1] = velp[0];
										vels[0] = vs[ki][1]*L1 + vs[ki+1][0]*L2; vels[1] = vels[0];
										density[0] = den[ki][1]*L1 + den[ki+1][0]*L2; density[1] = density[0];
										//get the up&down value for given piont 
										//via interpolation of this layer's top and bottom media pars;
										//the top interface's down value and bottom interface's up value
									}
									break;
								}

								up2 = ki; up1 = 1;
								down2 = ki; down1 = 1; 
								//for default point in normal layers
								
								if(ki==0)// for top surface
								{
									if(vp[ki][0]>SeisZero)// have valid value
									{
										up2 = 0; up1 = 0; down2 = 0; down1 = 0;
									}
									// means the physical point is beyond the media point
									velp[0] = vp[up2][up1]; velp[1] = vp[down2][down1];
									vels[0] = vs[up2][up1]; vels[1] = vs[down2][down1];
									density[0] = den[up2][up1]; density[1] = den[down2][down1];
									break;
								}
							
							}//z interface loop

							mu[0] = density[0]*vels[0]*vels[0];
							mu[1] = density[0]*vels[0]*vels[0];
							lamda[0] = density[0]*velp[0]*velp[0] - 2*mu[0];
							lamda[1] = density[0]*velp[0]*velp[0] - 2*mu[0];
							
							rho0= rho0+0.5*(density[0]+density[1]);
							if(mu[0]<=SeisZero || mu[1]<=SeisZero)
								waterflag = true;//default mu0=0.0 before
							else
								mu0 = mu0 + 0.5*(1.0/mu[0]+1.0/mu[1]);
							lam0 = lam0 + 0.5*(1.0/lamda[0]+1.0/lamda[1]);
							

						}//sampz loop;
					}//sampy loop;
				}//sampx loop;
				
				rho1 = rho0/nsamp;
				if(waterflag)
					mu1 = 0.0;
				else
					mu1 = nsamp/mu0;
				lam1 = nsamp/lam0;

				mpa.alpha[i][j][k] = sqrt( (lam1 + 2.0*mu1)/rho1 );
				mpa.beta[i][j][k] = sqrt( mu1/rho1 );
				mpa.rho[i][j][k] = rho1;
				
				//lambda = rho*Vp*Vp-2*miu;
				//miu = rho*Vs*Vs;
				//Vp=alpha;
				//Vs=beta;
				
			}// cdx.nk loop
		}//cdx.nj loop
	}//cdx.ni loop
	
	for(i=0;i<2;i++)
	{
		delete [] hiI[i];
		delete [] ziI[i];
	}
	delete [] hiI;
	delete [] ziI;

	for(i=0;i<3;i++)
	{
		for(j=0;j<3;j++)
			delete [] cI[i][j];
		delete [] cI[i];
	}
	delete [] cI;

	fprintf(stdout,"---accomplished generating the media parameters in media porgram\n");

}

void mediapar::read_volume_new(cindx cdx, coord crd)
{
	//do read only
	int i,j,k;
	char errstr[SeisStrLen];
	Real *mvrx,*mvry,*mvrz;//media volume read XYZ
	int mvrnx,mvrny,mvrnz;//media volume read dimension length, later modified to valid scale
	Real *temp;//valid media pars

	
	fprintf(stdout,"***Start to read the Volume-type media file %s in the media program\n",mediafile);
	
	//get the dimension size
	Vnc.ni = snc.dimsize(mediafile,"x");
	Vnc.nj = snc.dimsize(mediafile,"y");
#ifndef ZWmedia	
	Vnc.nk = snc.dimsize(mediafile,"z");
#else
	Vnc.nk = snc.dimsize(mediafile,"depth");
#endif

	//get the media coordinates
	mvrx = new Real[Vnc.ni];
	mvry = new Real[Vnc.nj];
	mvrz = new Real[Vnc.nk];
	snc.varget(mediafile,"x",mvrx);
	snc.varget(mediafile,"y",mvry);
#ifndef ZWmedia	
	snc.varget(mediafile,"z",mvrz);
#else
	//snc.varget(mediafile,"depth2sealevel",mvrz);//negetive of Z
	snc.varget(mediafile,"depth",mvrz);//negetive of Z
#endif
	
	//calculate media range location
	int xminL,xminR,xmaxL,xmaxR;
	int yminL,yminR,ymaxL,ymaxR;
	int zminL,zminR,zmaxL,zmaxR;
	Real *tempcoord;
	Real xyzs;//scale and value of XYZ
	Real xyzmin,xyzmax;
	
	xminL=0;	xminR=0;	xmaxL=0;	xmaxR=0;
	yminL=0;	yminR=0;	ymaxL=0;	ymaxR=0;
	zminL=0;	zminR=0;	zmaxL=0;	zmaxR=0;
	mvrnx=Vnc.ni;	mvrny=Vnc.nj;	mvrnz=Vnc.nk;

	
	tempcoord = new Real[cdx.ni*cdx.nj*cdx.nk]();
	
	com.flatten21D(crd.x, tempcoord, cdx, 1);
	xyzmin = mathf.min(tempcoord, cdx.ni*cdx.nj*cdx.nk);
	xyzmax = mathf.max(tempcoord, cdx.ni*cdx.nj*cdx.nk);
	mathf.LocValue1d(xyzmin, mvrx, mvrnx, &xminL, &xminR, &xyzs);
	mathf.LocValue1d(xyzmax, mvrx, mvrnx, &xmaxL, &xmaxR, &xyzs);
	mvrnx = xmaxR - xminL +1;//broad scale, choose the out side
	
	com.flatten21D(crd.y, tempcoord, cdx, 1);
	xyzmin = mathf.min(tempcoord, cdx.ni*cdx.nj*cdx.nk);
	xyzmax = mathf.max(tempcoord, cdx.ni*cdx.nj*cdx.nk);
	mathf.LocValue1d(xyzmin, mvry, mvrny, &yminL, &yminR, &xyzs);
	mathf.LocValue1d(xyzmax, mvry, mvrny, &ymaxL, &ymaxR, &xyzs);
	mvrny = ymaxR - yminL +1;
	
	com.flatten21D(crd.z, tempcoord, cdx, 1);
	xyzmin = mathf.min(tempcoord, cdx.ni*cdx.nj*cdx.nk);
	xyzmax = mathf.max(tempcoord, cdx.ni*cdx.nj*cdx.nk);
	mathf.LocValue1d(xyzmin, mvrz, mvrnz, &zminL, &zminR, &xyzs);
	mathf.LocValue1d(xyzmax, mvrz, mvrnz, &zmaxL, &zmaxR, &xyzs);
	mvrnz = zmaxR - zminL +1;

	Vnc.ni = mvrnx;
	Vnc.nj = mvrny;
	Vnc.nk = mvrnz;
	
	tempcoord = NULL;
	delete [] tempcoord;

	//read valid datas
	temp = new Real[Vnc.ni*Vnc.nj*Vnc.nk];
	Vnc.Xpos = new Real[Vnc.ni];
	Vnc.Ypos = new Real[Vnc.nj];
	Vnc.Zpos = new Real[Vnc.nk];
	Vnc.vp = new Real **[Vnc.ni];
	Vnc.vs = new Real **[Vnc.ni];
	Vnc.den = new Real **[Vnc.ni];
	for(i=0;i<Vnc.ni;i++)
	{
		Vnc.vp[i] = new Real *[Vnc.nj];
		Vnc.vs[i] = new Real *[Vnc.nj];
		Vnc.den[i] = new Real *[Vnc.nj];
		for(j=0;j<Vnc.nj;j++)
		{
			Vnc.vp[i][j] = new Real [Vnc.nk];
			Vnc.vs[i][j] = new Real [Vnc.nk];
			Vnc.den[i][j] = new Real [Vnc.nk];
		}
	}

	size_t subs[SeisGeo],subn[SeisGeo];
	ptrdiff_t subi[SeisGeo];
	subs[0] = xminL; subs[1] = yminL; subs[2] = zminL;
	subn[0] = Vnc.ni; subn[1] = Vnc.nj; subn[2] = Vnc.nk;
	subi[0] = 1;     subi[1] = 1;     subi[2] = 1;

	memcpy(Vnc.Xpos, mvrx+xminL, Vnc.ni*sizeof(Real));
	memcpy(Vnc.Ypos, mvry+yminL, Vnc.nj*sizeof(Real));
	memcpy(Vnc.Zpos, mvrz+zminL, Vnc.nk*sizeof(Real));

	snc.varget(mediafile, "Vp", temp, subs, subn, subi);
	for(i=0;i<Vnc.ni;i++)
		for(j=0;j<Vnc.nj;j++)
			for(k=0;k<Vnc.nk;k++)
				Vnc.vp[i][j][k] = temp[i*Vnc.nj*Vnc.nk + j*Vnc.nk +k];
	snc.varget(mediafile, "Vs", temp, subs, subn, subi);
	for(i=0;i<Vnc.ni;i++)
		for(j=0;j<Vnc.nj;j++)
			for(k=0;k<Vnc.nk;k++)
				Vnc.vs[i][j][k] = temp[i*Vnc.nj*Vnc.nk + j*Vnc.nk +k];
	snc.varget(mediafile, "rho", temp, subs, subn, subi);
	for(i=0;i<Vnc.ni;i++)
		for(j=0;j<Vnc.nj;j++)
			for(k=0;k<Vnc.nk;k++)
				Vnc.den[i][j][k] = temp[i*Vnc.nj*Vnc.nk + j*Vnc.nk +k];
	

	for(i=0;i<Vnc.ni;i++)
		for(j=0;j<Vnc.nj;j++)
			for(k=0;k<Vnc.nk;k++)
			{
				if( isnan(Vnc.den[i][j][k]) )
					printf("density at[%d][%d][%d] is %f\n",i,j,k,Vnc.den[i][j][k]);
				if( isnan(Vnc.vp[i][j][k]) )
					printf("vp at[%d][%d][%d] is %f\n",i,j,k,Vnc.vp[i][j][k]);
				if( isnan(Vnc.vs[i][j][k]) )
					printf("vs at[%d][%d][%d] is %f\n",i,j,k,Vnc.vs[i][j][k]);
			}
	printf("pass nan read check\n");




	temp = NULL;
	delete [] temp;

	delete [] mvrz;
	delete [] mvry;
	delete [] mvrx;
	
	fprintf(stdout,"---accomplished generating the media parameters in media porgram, by Volume type\n");

}

void mediapar::read_volume_old(cindx cdx, coord crd)
{
	//in this part, do read and interpolation
	int i,j,k;
	char errstr[SeisStrLen];
	Real *mvrx,*mvry,*mvrz;//media volume read XYZ
	int mvrnx,mvrny,mvrnz;//media volume read dimension length, later modified to valid scale
	Real *xval,*yval,*zval,***vp,***vs,***den,*temp;//valid media pars
	
	fprintf(stdout,"***Start to read the Volume-type media file %s in the media program\n",mediafile);
	
	//get the dimension size
	mvrnx = snc.dimsize(mediafile,"x");
	mvrny = snc.dimsize(mediafile,"y");
	mvrnz = snc.dimsize(mediafile,"z");
	//get the media coordinates
	mvrx = new Real[mvrnx];
	mvry = new Real[mvrny];
	mvrz = new Real[mvrnz];
	snc.varget(mediafile,"x",mvrx);
	snc.varget(mediafile,"y",mvry);
	snc.varget(mediafile,"z",mvrz);

	int xminL,xminR,xmaxL,xmaxR;
	int yminL,yminR,ymaxL,ymaxR;
	int zminL,zminR,zmaxL,zmaxR;
	Real *tempcoord;
	Real xyzs;//scale and value of XYZ
	Real xyzmin,xyzmax;

	tempcoord = new Real[cdx.ni*cdx.nj*cdx.nk]();
	
	com.flatten21D(crd.x, tempcoord, cdx, 1);
	xyzmin = mathf.min(tempcoord, cdx.ni*cdx.nj*cdx.nk);
	xyzmax = mathf.max(tempcoord, cdx.ni*cdx.nj*cdx.nk);
	mathf.LocValue1d(xyzmin, mvrx, mvrnx, &xminL, &xminR, &xyzs);
	mathf.LocValue1d(xyzmax, mvrx, mvrnx, &xmaxL, &xmaxR, &xyzs);
	mvrnx = xmaxR - xminL +1;//broad scale, choose the out side
	
	com.flatten21D(crd.y, tempcoord, cdx, 1);
	xyzmin = mathf.min(tempcoord, cdx.ni*cdx.nj*cdx.nk);
	xyzmax = mathf.max(tempcoord, cdx.ni*cdx.nj*cdx.nk);
	mathf.LocValue1d(xyzmin, mvry, mvrny, &yminL, &yminR, &xyzs);
	mathf.LocValue1d(xyzmax, mvry, mvrny, &ymaxL, &ymaxR, &xyzs);
	mvrny = ymaxR - yminL +1;
	
	com.flatten21D(crd.z, tempcoord, cdx, 1);
	xyzmin = mathf.min(tempcoord, cdx.ni*cdx.nj*cdx.nk);
	xyzmax = mathf.max(tempcoord, cdx.ni*cdx.nj*cdx.nk);
	mathf.LocValue1d(xyzmin, mvrz, mvrnz, &zminL, &zminR, &xyzs);
	mathf.LocValue1d(xyzmax, mvrz, mvrnz, &zmaxL, &zmaxR, &xyzs);
	mvrnz = zmaxR - zminL +1;
	
	tempcoord = NULL;
	delete [] tempcoord;

	//read valid datas
	xval = new Real[mvrnx];
	yval = new Real[mvrny];
	zval = new Real[mvrnz];
	temp = new Real[mvrnx*mvrny*mvrnz];
	vp = new Real **[mvrnx];
	vs = new Real **[mvrnx];
	den = new Real **[mvrnx];
	for(i=0;i<mvrnx;i++)
	{
		vp[i] = new Real *[mvrny];
		vs[i] = new Real *[mvrny];
		den[i] = new Real *[mvrny];
		for(j=0;j<mvrny;j++)
		{
			vp[i][j] = new Real [mvrnz];
			vs[i][j] = new Real [mvrnz];
			den[i][j] = new Real [mvrnz];
		}
	}

	size_t subs[SeisGeo],subn[SeisGeo];
	ptrdiff_t subi[SeisGeo];
	subs[0] = xminL; subs[1] = yminL; subs[2] = zminL;
	subn[0] = mvrnx; subn[1] = mvrny; subn[2] = mvrnz;
	subi[0] = 1;     subi[1] = 1;     subi[2] = 1;

	memcpy(xval, mvrx+xminL, mvrnx*sizeof(Real));
	memcpy(yval, mvry+yminL, mvrny*sizeof(Real));
	memcpy(zval, mvrz+zminL, mvrnz*sizeof(Real));

	snc.varget(mediafile, "Vp", temp, subs, subn, subi);
	for(i=0;i<mvrnx;i++)
		for(j=0;j<mvrny;j++)
			for(k=0;k<mvrnz;k++)
				vp[i][j][k] = temp[i*mvrny*mvrnz + j*mvrnz +k];
	snc.varget(mediafile, "Vs", temp, subs, subn, subi);
	for(i=0;i<mvrnx;i++)
		for(j=0;j<mvrny;j++)
			for(k=0;k<mvrnz;k++)
				vs[i][j][k] = temp[i*mvrny*mvrnz + j*mvrnz +k];
	snc.varget(mediafile, "rho", temp, subs, subn, subi);
	for(i=0;i<mvrnx;i++)
		for(j=0;j<mvrny;j++)
			for(k=0;k<mvrnz;k++)
				den[i][j][k] = temp[i*mvrny*mvrnz + j*mvrnz +k];
	
	temp = NULL;
	delete [] temp;

	//calculate effective media pars
	Real lam0,mu0,rho0;//confirm directly from media par
	Real lam1,mu1,rho1;//confirm after smooth
	bool waterflag;
	int nsamp;
	int is,js,ks;//loop for sampling index
	int ic,jc,kc;//loop for 3-value interped coord index
	Real xI[3],yI[3],zI[3],***cI;//cI[3][3][3];
	Real locx,locy,locz;// interp3D, physical point location
	Real x0,y0,z0;//interp3D, from physical coord
	Real xV[2],yV[2],zV[2],***cV;//cV[2][2][2];//interp3D, from media coord
	Real xnear,ynear,znear;//nearest value from xval yval and zval
	int xleft,xright,yleft,yright,zleft,zright;//index from behind
	Real tempvp,tempvs,tempden,tempmu,templam;
	
	xI[0]=-1.0; xI[1]=0.0; xI[2]=1.0;
	yI[0]=-1.0; yI[1]=0.0; yI[2]=1.0;
	zI[0]=-1.0; zI[1]=0.0; zI[2]=1.0;
	nsamp = MAX(2*sampx,1) * MAX(2*sampy,1) * MAX(2*sampz,1);
	
	cI = new Real **[3];
	for(i=0;i<3;i++)
	{
		cI[i] = new Real *[3];
		for(j=0;j<3;j++)
			cI[i][j] = new Real[3];
	}

	cV = new Real **[2];
	for(i=0;i<2;i++)
	{
		cV[i] = new Real *[2];
		for(j=0;j<2;j++)
			cV[i][j] = new Real[2];
	}
	
	for(i=0;i<cdx.nk;i++)
	{
		for(j=0;j<cdx.nj;j++)
		{
			for(k=0;k<cdx.nk;k++)
			{
				lam0 = 0.0; mu0 = 0.0; rho0 = 0.0;
				waterflag = false;

				for(is=-sampx;is<=sampx;is++)
				{
					if( sampx!=0 && is==0)
						continue;
					for(js=-sampy;js<=sampy;js++)
					{
						if( sampy!=0 && js==0)
							continue;
						for(ks=-sampz;ks<=sampz;ks++)
						{
							if( sampz!=0 && ks==0 )
								continue;

							locx = is/(2.0*(sampx+1.0));
							locy = js/(2.0*(sampy+1.0));
							locz = ks/(2.0*(sampz+1.0));
							
							//interp x0
							for(ic=0;ic<3;ic++)
								for(jc=0;jc<3;jc++)
									for(kc=0;kc<3;kc++)
										cI[ic][jc][kc] = crd.x[i-1+ic][j-1+jc][k-1+kc];
							x0 = mathf.interp3d(xI,yI,zI,cI,3,3,3,locx,locy,locz);
							
							//interp y0
							for(ic=0;ic<3;ic++)
								for(jc=0;jc<3;jc++)
									for(kc=0;kc<3;kc++)
										cI[ic][jc][kc] = crd.y[i-1+ic][j-1+jc][k-1+kc];
							y0 = mathf.interp3d(xI,yI,zI,cI,3,3,3,locx,locy,locz);
							
							//interp z0
							for(ic=0;ic<3;ic++)
								for(jc=0;jc<3;jc++)
									for(kc=0;kc<3;kc++)
										cI[ic][jc][kc] = crd.z[i-1+ic][j-1+jc][k-1+kc];
							z0 = mathf.interp3d(xI,yI,zI,cI,3,3,3,locx,locy,locz);

							mathf.LocValue1d(x0, xval, mvrnx, &xleft, &xright, &xnear);
							mathf.LocValue1d(y0, yval, mvrny, &yleft, &yright, &ynear);
							mathf.LocValue1d(z0, zval, mvrnz, &zleft, &zright, &znear);
							
							xV[0] = xval[xleft]; xV[1] = xval[xright];
							yV[0] = yval[yleft]; yV[1] = yval[yright];
							zV[0] = zval[zleft]; zV[1] = zval[zright];

							for(ic=0;ic<2;ic++)
								for(jc=0;jc<2;jc++)
									for(kc=0;kc<2;kc++)
										cV[ic][jc][kc] = vp[xleft+ic][yleft+jc][zleft+kc];
							tempvp = mathf.interp3d(xV,yV,zV,cV,2,2,2,xnear,ynear,znear);

							for(ic=0;ic<2;ic++)
								for(jc=0;jc<2;jc++)
									for(kc=0;kc<2;kc++)
										cV[ic][jc][kc] = vs[xleft+ic][yleft+jc][zleft+kc];
							tempvs = mathf.interp3d(xV,yV,zV,cV,2,2,2,xnear,ynear,znear);

							for(ic=0;ic<2;ic++)
								for(jc=0;jc<2;jc++)
									for(kc=0;kc<2;kc++)
										cV[ic][jc][kc] = den[xleft+ic][yleft+jc][zleft+kc];
							tempden = mathf.interp3d(xV,yV,zV,cV,2,2,2,xnear,ynear,znear);

							tempmu = tempden*tempvs*tempvs;
							templam = tempvp*tempvp*tempden - 2.0*tempmu;

							rho0 += tempden;
							lam0 += 1.0/templam;
							if(tempmu<SeisZero)
								waterflag = true;
							else
								mu0 += 1.0/tempmu;

						}//sampz loop;
					}//sampy loop;
				}//sampx loop;

				lam1 = nsamp/lam0;
				rho1 = rho0/nsamp;
				if(waterflag)
					mu1 = 0.0;
				else
					mu1 = nsamp/mu0;

				mpa.alpha[i][j][k] = sqrt( (lam1 + 2.0*mu1)/rho1 );
				mpa.beta[i][j][k] = sqrt( mu1/rho1 );
				mpa.rho[i][j][k] = rho1;

			}//loop for cdx.nk
		}//loop for cdx.nj
	}//loop for cdx.ni
	
	for(i=0;i<2;i++)
	{
		for(j=0;j<2;j++)
			delete [] cV[i][j];
		delete [] cV[i];
	}
	delete [] cV;

	for(i=0;i<3;i++)
	{
		for(j=0;j<3;j++)
			delete [] cI[i][j];
		delete [] cI[i];
	}
	delete [] cI;
	
	for(i=0;i<mvrnx;i++)
	{
		for(j=0;j<mvrny;j++)
		{
			delete [] den[i][j];
			delete [] vs[i][j];
			delete [] vs[i][j];
		}
		delete [] den[i];
		delete [] vs[i];
		delete [] vp[i];
	}
	delete [] den;
	delete [] vs;
	delete [] vp;
	delete [] xval;
	delete [] yval;
	delete [] zval;
	delete [] mvrz;
	delete [] mvry;
	delete [] mvrx;
	
	fprintf(stdout,"---accomplished generating the media parameters in media porgram\n");

}

void mediapar::ApplySmooth(cindx cdx, coord crd)
{
	int i,j,k,mi,mj,mk;
	int imin,imax, jmin,jmax, kmin,kmax;
	int number_of_avg,nmax_sum;
	int nsamp,nsampx,nsampy,nsampz;
	int ic,jc,kc;
	
	Real ***cV;
	Real xsamp,ysamp,zsamp;
	Real lamval,muval,rhoval;
	Real Vp2el[2],Vs2el[2],Dc2el[2];
	Real x0,y0,z0,d0,x1,y1,z1,z2;
	Real ztopo;
	Real vec3[3];

	fprintf(stdout,"****Start to apply smoothing for media\n");
	
	cV = new Real **[3];
	for(i=0;i<3;i++)
	{
		cV[i] = new Real *[3];
		for(j=0;j<3;j++)
			cV[i][j] = new Real[3];
	}

	vec3[0] = 0; 	vec3[1] = 2.0;  vec3[2] =  4.0;
	imin = cdx.nx + 1;	imax = 0;
	jmin = cdx.ny + 1;	jmax = 0;
	kmin = cdx.nz + 1;	kmax = 0;
	number_of_avg = 0;	nmax_sum = 0;

	for(i=cdx.ni1;i<cdx.ni2;i++)
		for(j=cdx.nj1;j<cdx.nj2;j++)
			for(k=cdx.nk1;k<cdx.nk2;k++)
			{
				nsampx = 1;	nsampy = 1;	nsampz = 1;
				if( OverLimits(vel3d[i][j][k],vel3d[i-1][j][k]) || OverLimits(vel3d[i][j][k],vel3d[i+1][j][k]) )
					nsampx = this->sampx;
				if( OverLimits(vel3d[i][j][k],vel3d[i][j-1][k]) || OverLimits(vel3d[i][j][k],vel3d[i][j+1][k]) )
					nsampy = this->sampy;
				if( OverLimits(vel3d[i][j][k],vel3d[i][j][k-1]) || OverLimits(vel3d[i][j][k],vel3d[i][j][k+1]) )
					nsampz = this->sampz;


				if( nsampx > 1 || nsampy > 1 || nsampz > 1 )
				{
					imin = MIN(imin,i);	imax = MAX(imax,i);
					jmin = MIN(jmin,j);	jmax = MAX(jmax,j);
					kmin = MIN(kmin,k);	kmax = max(kmax,k);
					number_of_avg = number_of_avg + 1;
					nmax_sum = MAX( nmax_sum, nsampx*nsampy*nsampz );

					nsamp = 0;
					lamval = 0;	muval = 0; 	rhoval = 0;

					for(mi=1;mi<=nsampx;mi++)
						for(mj=1;mj<=nsampy;mj++)
							for(mk=1;mk<=nsampz;mk++)
							{
								xsamp = 1.0 + 2.0/nsampx*(mi-1 +0.5);
								ysamp = 1.0 + 2.0/nsampy*(mj-1 +0.5);
								zsamp = 1.0 + 2.0/nsampz*(mk-1 +0.5);
								//here surround center is 2, and while VEC corresponding center is 2
								//perfectly mathched
								//if use VEC=[-1,1,1], then surround center should be 1.
							
								for(ic=0;ic<3;ic++)
									for(jc=0;jc<3;jc++)
										for(kc=0;kc<3;kc++)
											cV[ic][jc][kc] = crd.x[i-1 +ic][j-1 +jc][k-1 +kc];
								x0 = mathf.interp3d(vec3, vec3, vec3, cV, 3,3,3, xsamp,ysamp,zsamp);
								
								for(ic=0;ic<3;ic++)
									for(jc=0;jc<3;jc++)
										for(kc=0;kc<3;kc++)
											cV[ic][jc][kc] = crd.y[i-1 +ic][j-1 +jc][k-1 +kc];
								y0 = mathf.interp3d(vec3, vec3, vec3, cV, 3,3,3, xsamp,ysamp,zsamp);
								
								for(ic=0;ic<3;ic++)
									for(jc=0;jc<3;jc++)
										for(kc=0;kc<3;kc++)
											cV[ic][jc][kc] = crd.z[i-1 +ic][j-1 +jc][k-1 +kc];
								z0 = mathf.interp3d(vec3, vec3, vec3, cV, 3,3,3, xsamp,ysamp,zsamp);


								if(ISEQSTR(mediatype,"interface3D"))
									interface3d_discrete(x0,y0,z0,Vp2el,Vs2el,Dc2el);
								if(ISEQSTR(mediatype,"volume"))
									volume_discrete(x0,y0,z0,Vp2el,Vs2el,Dc2el);
								
								TwoinOne(Vp2el,Vs2el,Dc2el,&lamval,&muval,&rhoval,&nsamp);
							}//loop of all samping area
					
					average(i,j,k,lamval,muval,rhoval,nsamp);

				}//judgement of nsampXYZ
			}//loop of all valid points

	fprintf(stdout,"index range for average: X[%d,%d],Y[%d,%d],Z[%d,%d]\n",imin,imax,jmin,jmax,kmin,kmax);
	fprintf(stdout,"totally average points is %d\n",number_of_avg);
	fprintf(stdout,"the maxinum sampling points is %d\n",nmax_sum);

	for(i=0;i<3;i++)
	{
		for(j=0;j<3;j++)
			delete [] cV[i][j];
		delete [] cV[i];
	}
	delete [] cV;

	fprintf(stdout,"----acompilished media smoothing\n");

}
bool mediapar::OverLimits(Real a, Real b)
{
	bool cross;
	cross = false;
	
	if( ABS(a-b) >= crit_value || ABS(a-b)/MIN(a,b) >= crit_perc )
		cross = true;
	
	return cross;
}

void mediapar::MediaStatistics(cindx cdx)
{
	int i,j,k;

	Real avg_mu,max_mu,min_mu;
	Real avg_la,max_la,min_la;
	Real avg_dp,max_dp,min_dp;
	Real avg_vp,max_vp,min_vp;
	Real avg_vs,max_vs,min_vs;
	Real VP, VS, DC;

	avg_mu=0;	max_mu=0;	min_mu=9E25;
	avg_la=0;	max_la=0;	min_la=9E25;
	avg_dp=0;	max_dp=0;	min_dp=9E25;
	avg_vp=0;	max_vp=0;	min_vp=9E25;
	avg_vs=0;	max_vs=0;	min_vs=9E25;

	for(i=cdx.ni1;i<cdx.ni2;i++)
		for(j=cdx.nj1;j<cdx.nj2;j++)
			for(k=cdx.nk1;k<cdx.nk2;k++)
			{
				DC = density[i][j][k];
				VS = sqrt( miu[i][j][k]/DC );
				VP = sqrt( (lambda[i][j][k] + 2.0*miu[i][j][k])/DC );
				
				//statistics
				avg_vp = avg_vp + VP;
				min_vp = MIN(min_vp,VP);
				max_vp = MAX(max_vp,VP);

				avg_vs = avg_vs + VS;
				min_vs = MIN(min_vs,VS);
				max_vs = MAX(max_vs,VS);

				avg_dp = avg_dp + DC;
				min_dp = MIN(min_dp,DC);
				max_dp = MAX(max_dp,DC);

				avg_mu = avg_mu + miu[i][j][k];
				min_mu = MIN(min_mu,miu[i][j][k]);
				max_mu = MAX(max_mu,miu[i][j][k]);

				avg_la = avg_la + lambda[i][j][k];
				min_la = MIN(min_la,lambda[i][j][k]);
				max_la = MAX(max_la,lambda[i][j][k]);

				//transfer
				mpa.alpha[i][j][k] = VP;
				mpa.beta[i][j][k] = VS;
				mpa.rho[i][j][k] = DC;
				
				if(isnan(VP))
					printf("at(%d,%d,%d) vp=%f\n",i,j,k,VP);
				if(isnan(VS))
					printf("at(%d,%d,%d) vs=%f\n",i,j,k,VS);
				if(isnan(DC))
					printf("at(%d,%d,%d) dc=%f\n",i,j,k,DC);
			}
	fprintf(stdout,"   miu:average=%25.10lf, min=%25.10lf, max=%25.10lf\n",avg_mu/(cdx.ni*cdx.nj*cdx.nk),min_mu,max_mu);
	fprintf(stdout,"lambda:average=%25.10lf, min=%25.10lf, max=%25.10lf\n",avg_la/(cdx.ni*cdx.nj*cdx.nk),min_la,max_la);
	fprintf(stdout,"    Vp:average=%25.16lf, min=%25.16lf, max=%25.16lf\n",avg_vp/(cdx.ni*cdx.nj*cdx.nk),min_vp,max_vp);
	fprintf(stdout,"    Vs:average=%25.16lf, min=%25.16lf, max=%25.16lf\n",avg_vs/(cdx.ni*cdx.nj*cdx.nk),min_vs,max_vs);
	fprintf(stdout,"   rho:average=%25.16lf, min=%25.16lf, max=%25.16lf\n",avg_dp/(cdx.ni*cdx.nj*cdx.nk),min_dp,max_dp);

}


//------------------public-----------------------------------
mediapar::mediapar(const char* filename, cindx cdx, const int restart, const int Myid)
{
	myid = Myid;
	if(myid)
	{
		Mflag = false;//child procs
		printf("child procs %d doesn't paticipate into computing parameters (media)\n",myid);
		return;
	}
	else
		Mflag = true;//master procs
	
	getConf(filename);
	
	if(restart==1)
		Rwork = true;
	else
		Rwork = false;
	
	int i,j,m,n;

	Dnx = cdx.nx;
	Dny = cdx.ny;
	Dnz = cdx.nz;
	mpa.alpha = new Real **[Dnx];
	mpa.beta = new Real **[Dnx];
	mpa.rho = new Real **[Dnx];
	lambda = new Real **[Dnx];
	miu = new Real **[Dnx];
	density = new Real **[Dnx];
	vel3d = new Real **[Dnx];
	for(i=0;i<Dnx;i++)
	{
		mpa.alpha[i] = new Real *[Dny];
		mpa.beta[i] = new Real *[Dny];
		mpa.rho[i] = new Real *[Dny];
		lambda[i] = new Real *[Dny];
		miu[i] = new Real *[Dny];
		density[i] = new Real *[Dny];
		vel3d[i] = new Real *[Dny];
		for(j=0;j<Dny;j++)
		{
			mpa.alpha[i][j] = new Real [Dnz]();
			mpa.beta[i][j] = new Real [Dnz]();
			mpa.rho[i][j] = new Real [Dnz]();
			lambda[i][j] = new Real [Dnz]();
			miu[i][j] = new Real [Dnz]();
			density[i][j] = new Real [Dnz]();
			vel3d[i][j] = new Real [Dnz]();
		}
	}
	
	//interface3D pars
	if(ISEQSTR(mediatype,"interface3D"))
	{
		i3d.interface_flat = new bool[i3d.nInterface];
		i3d.layer_const = new bool[i3d.nInterface];

		i3d.interface_const = new bool*[i3d.nInterface];
		i3d.layerZrange = new Real*[i3d.nInterface];
		i3d.interfaceZrange = new Real*[i3d.nInterface];
		for(n=0;n<i3d.nInterface;n++)
		{
			i3d.interface_const[n] = new bool[2];
			i3d.layerZrange[n] = new Real[2];
			i3d.interfaceZrange[n] = new Real[2];
		}

		i3d.indexZ = new int***[i3d.nInterface];
		for(n=0;n<i3d.nInterface;n++)
		{
			i3d.indexZ[n] = new int**[i3d.nx_in];
			for(i=0;i<i3d.nx_in;i++)
			{
				i3d.indexZ[n][i] = new int*[i3d.ny_in];
				for(j=0;j<i3d.ny_in;j++)
					i3d.indexZ[n][i][j] = new int[2];
			}
		}

		i3d.Xpos = new Real[i3d.nx_in];
		i3d.Ypos = new Real[i3d.ny_in];

		i3d.Zpos = new Real**[i3d.nx_in];
		for(i=0;i<i3d.nx_in;i++)
		{
			i3d.Zpos[i] = new Real*[i3d.ny_in];
			for(j=0;j<i3d.ny_in;j++)
				i3d.Zpos[i][j] = new Real[i3d.nInterface];
		}

		i3d.Vp = new Real ***[2];
		i3d.Vs = new Real ***[2];
		i3d.Dc = new Real ***[2];
		for(m=0;m<2;m++)
		{
			i3d.Vp[m] = new Real**[i3d.nx_in];
			i3d.Vs[m] = new Real**[i3d.nx_in];
			i3d.Dc[m] = new Real**[i3d.nx_in];
			for(i=0;i<i3d.nx_in;i++)
			{
				i3d.Vp[m][i] = new Real*[i3d.ny_in];
				i3d.Vs[m][i] = new Real*[i3d.ny_in];
				i3d.Dc[m][i] = new Real*[i3d.ny_in];
				for(j=0;j<i3d.ny_in;j++)
				{
					i3d.Vp[m][i][j] = new Real[i3d.nInterface];
					i3d.Vs[m][i][j] = new Real[i3d.nInterface];
					i3d.Dc[m][i][j] = new Real[i3d.nInterface];
				}
			}
		}
	}

}
mediapar::~mediapar()
{
	fprintf(stdout,"into data free at Procs[%d],in mediapar.cpp\n",myid);
	if(this->Mflag)
	{
		int i,j,m,n;

		if(ISEQSTR(mediatype,"interface3D"))
		{
			for(m=0;m<2;m++)
			{
				for(i=0;i<i3d.nx_in;i++)
				{
					for(j=0;j<i3d.ny_in;j++)
					{
						delete [] i3d.Dc[m][i][j];
						delete [] i3d.Vs[m][i][j];
						delete [] i3d.Vp[m][i][j];
					}
					delete [] i3d.Dc[m][i];
					delete [] i3d.Vs[m][i];
					delete [] i3d.Vp[m][i];
				}
				delete [] i3d.Dc[m];
				delete [] i3d.Vs[m];
				delete [] i3d.Vp[m];
			}
			delete [] i3d.Dc;
			delete [] i3d.Vs;
			delete [] i3d.Vp;

			for(i=0;i<i3d.nx_in;i++)
			{
				for(j=0;j<i3d.ny_in;j++)
					delete [] i3d.Zpos[i][j];
				delete [] i3d.Zpos[i];
			}
			delete [] i3d.Zpos;
			delete [] i3d.Ypos;
			delete [] i3d.Xpos;

			for(n=0;n<i3d.nInterface;n++)
			{
				for(i=0;i<i3d.nx_in;i++)
				{
					for(j=0;j<i3d.ny_in;j++)
						delete [] i3d.indexZ[n][i][j];
					delete [] i3d.indexZ[n][i];
				}
				delete [] i3d.indexZ[n];
			}
			delete [] i3d.indexZ;

			for(n=0;n<i3d.nInterface;n++)
			{
				delete [] i3d.interfaceZrange[n];
				delete [] i3d.layerZrange[n];
				delete [] i3d.interface_const[n];
			}
			delete [] i3d.interfaceZrange;
			delete [] i3d.layerZrange;
			delete [] i3d.interface_const;

			delete [] i3d.layer_const;
			delete [] i3d.interface_flat;
		}

		if(ISEQSTR(mediatype,"volume"))
		{
			for(i=0;i<Vnc.ni;i++)
			{
				for(j=0;j<Vnc.nj;j++)
				{
					delete [] Vnc.den[i][j];
					delete [] Vnc.vs[i][j];
					delete [] Vnc.vp[i][j];
				}
				delete [] Vnc.den[i];
				delete [] Vnc.vs[i];
				delete [] Vnc.vp[i];
			}
			delete [] Vnc.den;
			delete [] Vnc.vs;
			delete [] Vnc.vp;
			delete [] Vnc.Xpos;
			delete [] Vnc.Ypos;
			delete [] Vnc.Zpos;
		}

		for(i=0;i<Dnx;i++)
		{
			for(j=0;j<Dny;j++)
			{
				delete [] vel3d[i][j];
				delete [] density[i][j];
				delete [] miu[i][j];
				delete [] lambda[i][j];
				delete [] mpa.rho[i][j];
				delete [] mpa.beta[i][j];
				delete [] mpa.alpha[i][j];
			}
			delete [] vel3d[i];
			delete [] density[i];
			delete [] miu[i];
			delete [] lambda[i];
			delete [] mpa.rho[i];
			delete [] mpa.beta[i];
			delete [] mpa.alpha[i];
		}
		delete [] vel3d;
		delete [] density;
		delete [] miu;
		delete [] lambda;
		delete [] mpa.rho;
		delete [] mpa.beta;
		delete [] mpa.alpha;
	}
	fprintf(stdout,"data free at Procs[%d],in mediapar.cpp\n",myid);
}

void mediapar::readdata(cindx cdx, coord crd)
{
	if(this->Rwork)
	{
		fprintf(stdout,"---Reading the exists media's data, there's no needs to read from ordinary one ,due to the restart work\n");
		return;
	}

	
	if(ISEQSTR(mediatype,"interface"))
		read_interface(cdx, crd);
	else if(ISEQSTR(mediatype,"interface3D"))
		read_interface3D(cdx, crd);
	else if(ISEQSTR(mediatype,"volume"))
		read_volume_new(cdx, crd);
		//read_volume_old(cdx, crd);
	else
	{
		char errstr[SeisStrLen];
		sprintf(errstr,"mediatype configure %s is wrong when reading media configure in main par file",mediatype);
		errprt(Fail2Check,errstr);
	}

	int nsamp;
	Real Vp2el[2], Vs2el[2], Dc2el[2];
	Real lamval, muval, rhoval;
	Real x0, y0, z0;
	int i,j,k;

	Vp2el[0] = 0; 	Vs2el[0] = 0;	Dc2el[0] = 0;
	Vp2el[1] = 0; 	Vs2el[1] = 0;	Dc2el[1] = 0;

	//discreate
	//i=49;	j==106;	k=5;
	for(i=cdx.nx1;i<cdx.nx2;i++)
		for(j=cdx.ny1;j<cdx.ny2;j++)
			for(k=cdx.nz1;k<cdx.nz2;k++)
			{
				x0 = crd.x[i][j][k]; 	y0 = crd.y[i][j][k]; 	z0 = crd.z[i][j][k];
				lamval = 0; 	muval = 0; 	rhoval = 0; 	nsamp=0;

				//printf("at (%d,%d,%d) start discreate positon is(%f, %f, %f)\n",i,j,k,x0,y0,z0);
				
				if(ISEQSTR(mediatype,"interface3D"))
					interface3d_discrete(x0,y0,z0,Vp2el,Vs2el,Dc2el);
				if(ISEQSTR(mediatype,"volume"))
					volume_discrete(x0,y0,z0,Vp2el,Vs2el,Dc2el);
				
				if(isnan(Vp2el[0]) || isnan(Vp2el[1]) )
					printf("----->at (%d,%d,%d) vp2el=%f,%f\n",i,j,k,Vp2el[0],Vp2el[1]);
				if(isnan(Vs2el[0]) || isnan(Vs2el[1]) )
					printf("----->at (%d,%d,%d) Vs2el=%f,%f\n",i,j,k,Vs2el[0],Vs2el[1]);
				if(isnan(Dc2el[0]) || isnan(Dc2el[1]) )
					printf("----->at (%d,%d,%d) Dc2el=%f,%f\n",i,j,k,Dc2el[0],Dc2el[1]);
				
				if(ISEQSTR(mediatype,"interface3D"))
				{
					TwoinOne(Vp2el,Vs2el,Dc2el,&lamval,&muval,&rhoval,&nsamp);
					average(i,j,k,lamval,muval,rhoval,nsamp);
				}
				else
					get2in1(i,j,k,Vp2el,Vs2el,Dc2el);
				
				set_vel3d(i,j,k,Vp2el);
			}
	
	//smoothing effective media
	if(sampx>1 || sampy>1 || sampz>1)
	{
		fprintf(stdout,"Apply smoothing work by samping factor %d %d and %d for effective media\n",sampx,sampy,sampz);
		ApplySmooth(cdx, crd);
	}

	//Statistics and transfer
	MediaStatistics(cdx);

	com.equivalence_extend(mpa.alpha, cdx);
	com.equivalence_extend(mpa.beta, cdx);
	com.equivalence_extend(mpa.rho, cdx);
}

void mediapar::timecheck(Real stept, cindx cdx, coord crd)
{
	if(this->Rwork)
	{
		fprintf(stdout,"---Reading the exists media's data, there's no needs to check for the max spread time, due to the restart work\n");
		return;
	}
	
	int i,j,k;
	int ii,jj,kk;
	Real vp,dtmin,distmin,dtmax;
	Real L;
	Real vec1[3]{},vec2[3]{},vec3[3]{};
	int index[3];
	Real velmax,distmax;
	Real largeL;
	char errstr[SeisStrLen];

	dtmax = SeisInf;
	distmin = SeisInf;
	largeL = -1.0*SeisInf;

	for(i=cdx.ni1;i<cdx.ni2;i++)
		for(j=cdx.nj1;j<cdx.nj2;j++)
			for(k=cdx.nk1;k<cdx.nk2;k++)
			{
				vp = mpa.alpha[i][j][k];
				
				for(ii=-1;ii<=1;ii++)
					for(jj=-1;jj<=1;jj++)
						for(kk=-1;kk<=1;kk++)
							if(ii!=0 && jj!=0 && kk!=0)
							{
								vec1[0] = crd.x[i-ii][j][k]; vec1[1] = crd.y[i-ii][j][k]; vec1[2] = crd.z[i-ii][j][k];
								vec2[0] = crd.x[i][j-jj][k]; vec2[1] = crd.y[i][j-jj][k]; vec2[2] = crd.z[i][j-jj][k];
								vec3[0] = crd.x[i][j][k-kk]; vec3[1] = crd.y[i][j][k-kk]; vec3[2] = crd.z[i][j][k-kk];

								L = mathf.distanceP2S(crd.x[i][j][k], crd.y[i][j][k], crd.z[i][j][k], vec1, vec2, vec3);
								distmin = MIN(distmin, L);
								largeL = MAX(largeL, L);
							}

				dtmin = 1.3/vp*distmin; 

				if(dtmin < dtmax)
				{
					dtmax = dtmin;
					velmax = vp;
					distmax = distmin;
					index[0] = i;
					index[1] = j;
					index[2] = k;
				}

			}
		
		fprintf(stdout,"in this media, the max spread time is %f (actually used %f)",dtmin, stept);
		fprintf(stdout," on the point(%d,%d,%d) which maximum velocity is ",index[0],index[1],index[2]);
		fprintf(stdout,"%f and the minimal distace is %f the maximum distance is %f (used for grid computation)\n",velmax,distmax,largeL);

		if(dtmin < stept)
		{
			sprintf(errstr,"The computational stept (%f seconds) exceeds limitation (%f seconds), Please reduce the stept or adjust the model\n",
				stept,dtmin);
			errprt(Fail2Check,errstr);
		}

}

void mediapar::export_data(const char *path, cindx cdx, coord crd)
{
	char mfile[SeisStrLen];
	sprintf(mfile, "%s/media.nc", path);

	if(this->Rwork)
	{
		fprintf(stdout,"---There's no needs to store data, due to the restart work\n");
		snc.media_import(mpa, cdx, crd, mfile);
	}
	else
		snc.media_export(mpa, cdx, crd, mfile);
}
			
















