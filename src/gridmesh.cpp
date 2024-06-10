#include "typenew.h"
#include<math.h>

using namespace std;
using namespace defstruct;
using namespace constant;

#define errprt(...) com.errorprint(__FILE__, __LINE__, __VA_ARGS__)

//-------------------private--------------------------------
void gridmesh::getConf(const char *filename)
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
	com.get_conf(fp, "grid_type", 3, gridtype);
	com.get_conf(fp, "grid_filename", 3, name);
	com.get_conf(fp, "hybrid_grid", 3, &HyGrid);
	com.get_conf(fp, "conversion_interface_depth", 3, &ConversDepth);
	fclose(fp);
	
	sprintf(gridfile, "%s/%s",parpath,name);

}

void gridmesh::read_point(cindx cdx)
{
	size_t subs[SeisGeo],subn[SeisGeo];
	ptrdiff_t subi[SeisGeo];
	Real *var;
	int i,j,k;
	var = new Real[cdx.ni*cdx.nj*cdx.nk]();
	//just read the valid data and put into related index.
	//the ordinary Point file doesn't have virtual points
 
 	fprintf(stdout,"***Start to read the Point-type grid file %s in the gridmesh program\n",gridfile);

	subs[0]=0;      subs[1]=0;      subs[2]=0;
	subn[0]=cdx.ni; subn[1]=cdx.nj; subn[2]=cdx.nk;
	subi[0]=1;      subi[1]=1;      subi[2]=1;
	
	snc.varget(gridfile,"x",var,subs,subn,subi);
	for(i=0;i<cdx.ni;i++)
		for(j=0;j<cdx.nj;j++)
			for(k=0;k<cdx.nk;k++)
				crd.x[i+cdx.ni1][j+cdx.nj1][k+cdx.nk1]=var[i*cdx.nj*cdx.nk + j*cdx.nk + k];

	snc.varget(gridfile,"y",var,subs,subn,subi);
	for(i=0;i<cdx.ni;i++)
		for(j=0;j<cdx.nj;j++)
			for(k=0;k<cdx.nk;k++)
				crd.x[i+cdx.ni1][j+cdx.nj1][k+cdx.nk1]=var[i*cdx.nj*cdx.nk + j*cdx.nk + k];

	snc.varget(gridfile,"z",var,subs,subn,subi);
	for(i=0;i<cdx.ni;i++)
		for(j=0;j<cdx.nj;j++)
			for(k=0;k<cdx.nk;k++)
				crd.x[i+cdx.ni1][j+cdx.nj1][k+cdx.nk1]=var[i*cdx.nj*cdx.nk + j*cdx.nk + k];
	delete [] var;
}

void gridmesh::read_vmapnew(cindx cdx, const char* filename, Real steph)
{
	int i,j,k;
	char errstr[SeisStrLen*3];
	FILE *fp;
	
	fp=fopen(gridfile,"r");
	if(!fp)
	{
		sprintf(errstr,"fail to open gridvmap data file %s",gridfile);
		errprt(Fail2Open,errstr);
	}

	fprintf(stdout,"***Start to read the Vmap-type grid file %s in the gridmesh program\n",gridfile);

	int dims[3], *gridpoints;
	bool *equalspacing;

	com.get_conf(fp, "vmap_dims", 3, &dims[0]);
	com.get_conf(fp, "vmap_dims", 4, &dims[1]);
	com.get_conf(fp, "vmap_dims", 5, &dims[2]);
	gridpoints = new int[dims[2]-1];//grid points between two surface;
	equalspacing = new bool[dims[2]-1];//whether the grid point distance is euqal;

	for(i=0;i<dims[2]-1;i++)//surface number = segment number +1
	{
		com.get_conf(fp, "vmap_gridpoints", 3+i, &gridpoints[i]);
		com.get_conf(fp, "vmap_equalspacing", 3+i, &equalspacing[i]);
	}

	com.setchunk(fp,"<vmap_anchor>");
	//assuming that X and Y is fixed space interval
	Real **posx,**posy,***posz,tempx,tempy;
	posx = new Real*[dims[0]];
	posy = new Real*[dims[0]];
	posz = new Real**[dims[0]];
	for(i=0;i<dims[0];i++)
	{
		posx[i] = new Real[dims[1]];
		posy[i] = new Real[dims[1]];
		posz[i] = new Real*[dims[1]];
		for(j=0;j<dims[1];j++)
		{
			posz[i][j] = new Real[dims[2]];
		}
	}

	for(i=0;i<dims[0];i++)//xcircle, low
	{
		for(j=0;j<dims[1];j++)//ycircle, fast
		{
			fscanf(fp, Rformat, &tempx);
			fscanf(fp, Rformat, &tempy);
			for(k=0;k<dims[2];k++)// coopprate with x
				fscanf(fp,Rformat,&posz[i][j][k]);//write Z every step
			posx[i][j] = tempx;
			posy[i][j] = tempy;
		}
	}
	fclose(fp);


	//check data input
	printf("Gridfile %d %d %d, SeisDefine %d %d %d\n",dims[0],dims[1],dims[2],cdx.ni,cdx.nj,cdx.nk);
	if(dims[0] != cdx.ni)
		errprt(Fail2Check, "The X value number(dims[0]) obtained from gridfile is different with configure number(cdx.ni)");
	if(dims[1] != cdx.nj)
		errprt(Fail2Check, "The Y value number(dims[0]) obtained from gridfile is different with configure number(cdx.nj)");
	if( (!equalspacing[0]) || (!equalspacing[dims[2]-2]))
		errprt(Fail2Check, "The first layer and last layer should be equal spacing");
	for(i=1;i<dims[2];i++)
		if( (!equalspacing[i-1]) && (!equalspacing[i]) )
		errprt(Fail2Check, "There is only one layer(in two adjacent layers) need to adjust the grid spacing");
	if( mathf.sum(gridpoints,dims[2]-1)+1 != cdx.nk)
		errprt(Fail2Check, "The Z value number[ sum(gridpoints)+1 ] obtained from gridfile is different with configure number(cdx.nk)");
	for(i=0;i<dims[0];i++)
		for(j=0;j<dims[1];j++)
			for(k=1;k<dims[2];k++)
				if( posz[i][j][k]>posz[i][j][k-1] )
				{
					sprintf(errstr,"The Z direction depth should be monotonically decreasing, while position[%d][%d][%d]"
						" should deeper than [%d][%d][%d] at location (%fm,%fm) and the depth is %fm  %fm, respectively \n",
						     i,j,k, i,j,k-1, posx[i][j],posy[i][j], posz[i][j][k], posz[i][j][k-1]);
					errprt(Fail2Check, errstr);
				}
	
	//PPW calculating
	FILE *fps;
	char parpath[SeisStrLen];
	char srcfile[SeisStrLen];
	char name[SeisStrLen2];
	int nfrc, nmnt, nwin;
	Real tempfreq, maxfreq;
	Real RefVel, MaxInterval;

	fps = fopen(filename,"r");
	com.get_conf(fps, "seispath", 3, parpath);
	com.get_conf(fps, "source_filename", 3, name);
	com.get_conf(fps, "reference_velocity",3, &RefVel);
	fclose(fps);

	sprintf(srcfile,"%s/%s",parpath,name);

	fps = fopen(srcfile,"r");
	if(!fps)
	{
		sprintf(errstr,"fail to open source par file %s in gridmesh.cpp",srcfile);
		errprt(Fail2Open,errstr);
	}

	com.get_conf(fps,"number_of_force_source",3,&nfrc);
	com.get_conf(fps,"number_of_moment_source",3,&nmnt);
	maxfreq = -1.0;

	if(nfrc)
	{
		com.get_conf(fps,"force_stf_window",3,&nwin);

		for(i=0;i<nfrc;i++)
			for(j=0;j<nwin;j++)
			{
				com.get_conf(fps,"force_stf_freqfactor",3+j,&tempfreq);
				maxfreq = MAX(maxfreq, tempfreq);
			}
	}
	if(nmnt)
	{
		com.get_conf(fps,"moment_stf_window",3,&nwin);
		for(i=0;i<nmnt;i++)
			for(j=0;j<nwin;j++)
			{
				com.get_conf(fps,"moment_stf_freqfactor",3+j,&tempfreq);
				maxfreq = MAX(maxfreq,tempfreq);
			}
	}
	fclose(fps);

	//PPW conputes max intervals
	MaxInterval = RefVel/(2.0*maxfreq*8);
	fprintf(stdout,"The min Vs = %g m/s and the max frequency (2*f0) = %g Hz,"
			"for 8 grid point per wavelegth the maximum interval is %g m"
			" and the RECOMMANDED interval(steph) is %g m\n",RefVel,2.0*maxfreq,MaxInterval,steph);
	
	//Z direction value generating
	fprintf(stdout,"***Start to generate the curvilinear coordinate in gridmesh porgram\n");

	Real GinterfaceZrange[ dims[2] ][2];
	Real GlayerZrange[ dims[2]-1 ][2];
	Real Zmin,Zmax;
	//interface
	for(k=0;k<dims[2];k++)
	{
		Zmax = posz[0][0][k];
		Zmin = Zmax;
		for(i=0;i<dims[0];i++)
			for(j=0;j<dims[1];j++)
			{
				Zmax = MAX(posz[i][j][k],Zmax);
				Zmin = MIN(posz[i][j][k],Zmin);
			}
		GinterfaceZrange[k][0] = Zmin;
		GinterfaceZrange[k][1] = Zmax;
	}
	//layer
	for(k=0;k<dims[2]-1;k++)
	{
		Zmax = posz[0][0][k];
		Zmin = posz[0][0][k+1];
		for(i=0;i<dims[0];i++)
			for(j=0;j<dims[1];j++)
			{
				Zmax = MAX(posz[i][j][k],Zmax);
				Zmin = MIN(posz[i][j][k+1],Zmin);
			}
		GlayerZrange[k][0] = Zmin;
		GlayerZrange[k][1] = Zmax;
	}

	fprintf(stdout,"Confirm Gird-Z-dir interface range information\n");
	for(k=0;k<dims[2];k++)
		fprintf(stdout,"\tinterface[%d] range from %lf to %lf\n",k,GinterfaceZrange[k][0],GinterfaceZrange[k][1]);

	fprintf(stdout,"Confirm Grid-Zdir layer range information\n");
	for(k=0;k<dims[2]-1;k++)
		fprintf(stdout,"\tlayer[%d] range from %lf to %lf\n",k,GlayerZrange[k][0],GlayerZrange[k][1]);
	
	if(this->HyGrid)
	{
		fprintf(stdout,"---->>>Will applying hybrid grid techniques, the conversion depth is %g,"
			"above this interface using curvilinear grid, below this using rectagular grid\n",ConversDepth);

		//confirm ConverDepth layer location and interface location
		int ConInterface;
		Real TopInterface,BottomInterface;//last layer depth
		for(k=0;k<dims[2]-1;k++)
		{
			if(ConversDepth >= GlayerZrange[k][0] && ConversDepth <= GlayerZrange[k][1])
			{
				fprintf(stdout,"\tConversDepth(%gm) locates in layer[%d] between %gm and %gm\n",
						ConversDepth,k,GlayerZrange[k][0],GlayerZrange[k][1]);
				ConInterface = k+1;//ConLayer(contain as bottom)=ConInterface-1
				TopInterface = ConversDepth;
				BottomInterface = GinterfaceZrange[ dims[2]-1 ][0];
				fprintf(stdout,"\tset ConversInterface at interface[%d](start from 0) and adjust last layer ranges from %gm to %gm\n",
						ConInterface,TopInterface,BottomInterface);
				break;
				//locates in Layer K
				//total K+1 layer
				//Layer[K]-lower interface = constant ConversDepth
				//Layer[K+1]-upper interface = constant ConversDepth, lower interface = minimal value of orginal last interface
				//total K+2 interface
				//Interface[K]-upper interface of LayerK, keep unchanged
				//Interface[K+1]-lower of LayerK and upper of LayerK+1, set to constant ConversDepth
				//Interface[K+2]-lower of layerK+1, set to the minimal value of last interface
			}
		}

		Real NewInt[ConInterface+1+1][2],NewLayer[ConInterface+1][2];//total ConInterface+2 interface(0---CI+1), ConInterface+1 layer(0----Ci+1)
		Real ***Zdep;//new Z position(above convers interface)
		int SpaceEQ[ConInterface+1];
		Zdep = new Real**[dims[0]];
		for(i=0;i<dims[0];i++)
		{
			Zdep[i] = new Real*[dims[1]];
			for(j=0;j<dims[1];j++)
				Zdep[i][j] = new Real[ConInterface+2];//all XY range and all interface
		}

		for(k=0;k<ConInterface;k++)
			SpaceEQ[k] = equalspacing[k];
		SpaceEQ[ConInterface] = 1;

		for(i=0;i<dims[0];i++)
			for(j=0;j<dims[1];j++)
				for(k=0;k<ConInterface;k++)
					Zdep[i][j][k] = posz[i][j][k];

		for(k=0;k<ConInterface;k++)
		{
			NewInt[k][0] = GinterfaceZrange[k][0];
			NewInt[k][1] = GinterfaceZrange[k][1];
		}
		NewInt[ConInterface][0] = TopInterface;
		NewInt[ConInterface][1] = NewInt[ConInterface][0];
		NewInt[ConInterface+1][0] = BottomInterface;
		NewInt[ConInterface+1][1] = NewInt[ConInterface+1][0];
		
		for(i=0;i<dims[0];i++)
			for(j=0;j<dims[1];j++)
			{
				k = ConInterface;//upper interface of last layer
				Zdep[i][j][k] = NewInt[ConInterface][0];
				k = ConInterface+1;//bottom interface of last layer
				Zdep[i][j][k] = NewInt[ConInterface+1][0];
			}

		for(k=0;k<ConInterface;k++)
		{
			NewLayer[k][0] = GlayerZrange[k][0];
			NewLayer[k][1] = GlayerZrange[k][1];
		}
		NewLayer[ConInterface-1][0] = TopInterface;
		NewLayer[ConInterface][0] = BottomInterface;
		NewLayer[ConInterface][1] = TopInterface;

		fprintf(stdout,"New-Gird-Z-dir interface range information\n");
		for(k=0;k<ConInterface+2;k++)
			fprintf(stdout,"\tinterface[%d] range from %lf to %lf\n",k,NewInt[k][0],NewInt[k][1]);

		fprintf(stdout,"New-Grid-Zdir layer range information\n");
		for(k=0;k<ConInterface+1;k++)
			fprintf(stdout,"\tlayer[%d] range from %lf to %lf, and space-equal parameter is %d \n",k,NewLayer[k][0],NewLayer[k][1],SpaceEQ[k]);

		int GPs[ConInterface+1];
		Real deltaZ[ConInterface+1];
		int indexZ[ConInterface+2];
		Real totalL,partL;
		
		totalL=0;
		for(k=0;k<ConInterface+1;k++)
			totalL += NewLayer[k][1]-NewLayer[k][0];

		GPs[ConInterface] = cdx.nk-1;
		for(k=0;k<ConInterface+1;k++)
		{
			partL = NewLayer[k][1]-NewLayer[k][0];
			if(k<ConInterface)
			{
				GPs[k] = cdx.nk*partL/totalL;
				GPs[ConInterface] = GPs[ConInterface]-GPs[k];//last layer
			}
			deltaZ[k] = partL/GPs[k];
			printf("partL=%g,totalL=%g,GP=%d,deltaZ=%g\n",partL,totalL,GPs[k],deltaZ[k]);
		}

		indexZ[ConInterface+1] = cdx.nk1;
		for(k=ConInterface;k>=0;k--)
			indexZ[k] = indexZ[k+1] + GPs[k];
		for(k=0;k<ConInterface+2;k++)
			printf("indexZ[%d]=%d\n",k,indexZ[k]);

		this->ConIndex = indexZ[ConInterface+1]+GPs[ConInterface];
		fprintf(stdout,"ConversDepth's related Z-dir location is %d(to Max is curved, to Min is straight)"
			       ", interface index is %d, layer index (as bottom) is %d\n",ConIndex,ConInterface,ConInterface-1);
		
		Real zvec[cdx.nz];
		Real zstep0,zstep1;
		int zidx1,zidx2;//store the layer interface index of Z
		int m;
		Real diff,maxdiff,mindiff;
		maxdiff=0;	mindiff = MaxInterval;
		
		for(i=cdx.ni1;i<cdx.ni2;i++)
		{
			for(j=cdx.nj1;j<cdx.nj2;j++)
			{
				//store X and Y
				for(k=cdx.nk1;k<cdx.nk2;k++)
				{
					crd.x[i][j][k] = posx[i-cdx.ni1][j-cdx.nj1];//valid position
					crd.y[i][j][k] = posy[i-cdx.ni1][j-cdx.nj1];
				}
				
				//interpoalte Z
				for(k=0;k<ConInterface+1;k++)
				{
					zidx2 = indexZ[k];//big
					zidx1 = indexZ[k+1];//small
					
					if( SpaceEQ[k] )
					{
						zstep0 = ( Zdep[i-cdx.ni1][j-cdx.nj1][k] - Zdep[i-cdx.ni1][j-cdx.nj1][k+1] )/ (1.0*GPs[k]);
						for( m=zidx1; m <= zidx2; m++)
							zvec[m] = Zdep[i-cdx.ni1][j-cdx.nj1][k] + (m - zidx2)*zstep0;
					}
					else
					{
						zstep0 = ( Zdep[i-cdx.ni1][j-cdx.nj1][k-1] - Zdep[i-cdx.ni1][j-cdx.nj1][k] ) / (1.0*GPs[k-1]);
						zstep1 = ( Zdep[i-cdx.ni1][j-cdx.nj1][k+1] - Zdep[i-cdx.ni1][j-cdx.nj1][k+2] ) / (1.0*GPs[k+1]);
						smooth_zdepth(&zvec[indexZ[k+1]],Zdep[i-cdx.ni1][j-cdx.nj1][k],Zdep[i-cdx.ni1][j-cdx.nj1][k+1],
								zstep0,zstep1,GPs[k]);
					}
					zvec[zidx2] = Zdep[i-cdx.ni1][j-cdx.nj1][k];
					zvec[zidx1] = Zdep[i-cdx.ni1][j-cdx.nj1][k+1];
					
				}
				for(k=cdx.nk1;k<cdx.nk2;k++)
					crd.z[i][j][k] = zvec[k];
				for(k=cdx.nk1;k<cdx.nk2-1;k++)
				{
					diff=ABS(zvec[k]-zvec[k+1]);
					mindiff = MIN(mindiff,diff);
					maxdiff = MAX(maxdiff,diff);
				}
			}
		}
		printf("maxdiff=%g, mindiff=%g\n",maxdiff,mindiff);
		
		for(i=0;i<dims[0];i++)
		{
			for(j=0;j<dims[1];j++)
				delete [] Zdep[i][j];
			delete [] Zdep[i];
		}
		delete [] Zdep;
	}
	else
	{
		fprintf(stdout,"---->>>Will applying curvilinear grid for all range\n");
		this->ConIndex = cdx.nk1;
		
		int *zindex;
		zindex = new int[dims[2]];
		zindex[dims[2]-1] = cdx.nk1; //confirm the Z index is reverse, the bottom is index:0, the top is index:nk-1 
		for(i=dims[2]-2;i>=0;i--)
			zindex[i] = zindex[i+1] + gridpoints[i];
		// 4 surface ; 3 layer ; 3 gridpoints ; gridpoints index 0-2;
		// gridpoints 2,1,0;  2= 4 - 2;

		int zidx1,zidx2;//store the layer interface index of Z
		Real zstep0,zstep1;
		Real zvec[cdx.nz];
		int m;
		
		Real diff,maxdiff,mindiff;
		maxdiff=0;	mindiff = MaxInterval;

		for(i=cdx.ni1;i<cdx.ni2;i++)
		{
			for(j=cdx.nj1;j<cdx.nj2;j++)
			{
				//store X and Y
				for(k=cdx.nk1;k<cdx.nk2;k++)
				{
					crd.x[i][j][k] = posx[i-cdx.ni1][j-cdx.nj1];//valid position
					crd.y[i][j][k] = posy[i-cdx.ni1][j-cdx.nj1];
				}

				//interpoalte Z
				for(k=0;k<dims[2]-1;k++)
				{
					zidx2 = zindex[k];//big
					zidx1 = zindex[k+1];//small
					if( equalspacing[k] )
					{
						zstep0 = ( posz[i-cdx.ni1][j-cdx.nj1][k] - posz[i-cdx.ni1][j-cdx.nj1][k+1] )/ (1.0*gridpoints[k]);
						// posz[k]-posz[k+1] = positive
						// m-zidx2 = negetive
						// zvec[small to big] = value[big to small]=[0 to negetive] 
						// zindex[0] big, zindex[dims[2]-1] small=cdx.nk1
						// posz[0] big=0, posz[dims[2]-1] small=-10000
						for( m=zidx1; m <= zidx2; m++)
							zvec[m] = posz[i-cdx.ni1][j-cdx.nj1][k] + (m - zidx2)*zstep0;
					}
					else
					{
						zstep0 = ( posz[i-cdx.ni1][j-cdx.nj1][k-1] - posz[i-cdx.ni1][j-cdx.nj1][k] ) / (1.0*gridpoints[k-1]);
						zstep1 = ( posz[i-cdx.ni1][j-cdx.nj1][k+1] - posz[i-cdx.ni1][j-cdx.nj1][k+2] ) / (1.0*gridpoints[k+1]);
						//there is a transtion zone between two fixed spacing layer. In this zone, the spacing should
						//change from zstep0 to zstep1.
						smooth_zdepth(&zvec[zindex[k+1]],posz[i-cdx.ni1][j-cdx.nj1][k],posz[i-cdx.ni1][j-cdx.nj1][k+1],
								zstep0,zstep1,gridpoints[k]);
					}
					zvec[zidx2] = posz[i-cdx.ni1][j-cdx.nj1][k];
					zvec[zidx1] = posz[i-cdx.ni1][j-cdx.nj1][k+1];
				}
				for(k=cdx.nk1;k<cdx.nk2;k++)
					crd.z[i][j][k] = zvec[k];
				for(k=cdx.nk1;k<cdx.nk2-1;k++)
				{
					diff=ABS(zvec[k]-zvec[k+1]);
					mindiff = MIN(mindiff,diff);
					maxdiff = MAX(maxdiff,diff);
				}
			}
		}
		printf("maxdiff=%g, mindiff=%g\n",maxdiff,mindiff);

		delete [] zindex;
	}
	
	
	//varying X Y intervals
	for(i=0;i<dims[0];i++)
	{
		for(j=0;j<dims[1];j++)
		{
			delete [] posz[i][j];
		}
		delete [] posz[i];
		delete [] posy[i];
		delete [] posx[i];
	}
	delete [] posz;
	delete [] posy;
	delete [] posx;
	delete [] equalspacing;
	delete [] gridpoints;
	
	fprintf(stdout,"---accomplished generating the curvilinear coordinate in gridmesh porgram\n");
}

void gridmesh::read_vmap(cindx cdx, const char* filename, Real steph)
{
	int i,j,k;
	char errstr[SeisStrLen];
	FILE *fp;
	
	fp=fopen(gridfile,"r");
	if(!fp)
	{
		sprintf(errstr,"fail to open gridvmap data file %s",gridfile);
		errprt(Fail2Open,errstr);
	}

	fprintf(stdout,"***Start to read the Vmap-type grid file %s in the gridmesh program\n",gridfile);

	int dims[3], *gridpoints;
	bool *equalspacing;

	com.get_conf(fp, "vmap_dims", 3, &dims[0]);
	com.get_conf(fp, "vmap_dims", 4, &dims[1]);
	com.get_conf(fp, "vmap_dims", 5, &dims[2]);
	gridpoints = new int[dims[2]-1];//grid points between two surface;
	equalspacing = new bool[dims[2]-1];//whether the grid point distance is euqal;

	for(i=0;i<dims[2]-1;i++)//surface number = segment number +1
	{
		com.get_conf(fp, "vmap_gridpoints", 3+i, &gridpoints[i]);
		com.get_conf(fp, "vmap_equalspacing", 3+i, &equalspacing[i]);
	}

	com.setchunk(fp,"<vmap_anchor>");
	//assuming that X and Y is fixed space interval
	/*
	Real *posx,*posy,***posz,tempx,tempy;
	posx = new Real[dims[0]];
	posy = new Real[dims[1]];
	posz = new Real**[dims[0]];
	for(i=0;i<dims[0];i++)
	{
		posz[i] = new Real*[dims[1]];
		for(j=0;j<dims[1];j++)
			posz[i][j] = new Real[dims[2]];
	}

	for(j=0;j<dims[1];j++)//ycircle, 2nd change
	{
		for(i=0;i<dims[0];i++)//xcircle, first change
		{
			fscanf(fp, Rformat, &tempx);
			fscanf(fp, Rformat, &tempy);
			for(k=0;k<dims[2];k++)// coopprate with x
				fscanf(fp,Rformat,&posz[i][j][k]);//write Z every step
			if(!j)
				posx[i]=tempx;//obtain the X value in first Y loop
		}
		posy[j]=tempy;//write the Y value at every new Y step
	}
	*/
	//assuming that X and Y is varying space interval
	Real **posx,**posy,***posz,tempx,tempy;
	posx = new Real*[dims[0]];
	posy = new Real*[dims[0]];
	posz = new Real**[dims[0]];
	for(i=0;i<dims[0];i++)
	{
		posx[i] = new Real[dims[1]];
		posy[i] = new Real[dims[1]];
		posz[i] = new Real*[dims[1]];
		for(j=0;j<dims[1];j++)
		{
			posz[i][j] = new Real[dims[2]];
		}
	}

	for(j=0;j<dims[1];j++)//ycircle, 2nd change
	{
		for(i=0;i<dims[0];i++)//xcircle, first change
		{
			fscanf(fp, Rformat, &tempx);
			fscanf(fp, Rformat, &tempy);
			for(k=0;k<dims[2];k++)// coopprate with x
				fscanf(fp,Rformat,&posz[i][j][k]);//write Z every step
			posx[i][j] = tempx;
			posy[i][j] = tempy;
		}
	}
	fclose(fp);

	//check data input
	printf("Gridfile %d %d %d, SeisDefine %d %d %d\n",dims[0],dims[1],dims[2],cdx.ni,cdx.nj,cdx.nk);
	if(dims[0] != cdx.ni)
		errprt(Fail2Check, "The X value number(dims[0]) obtained from gridfile is different with configure number(cdx.ni)");
	if(dims[1] != cdx.nj)
		errprt(Fail2Check, "The Y value number(dims[0]) obtained from gridfile is different with configure number(cdx.nj)");
	if( (!equalspacing[0]) || (!equalspacing[dims[2]-2]))
		errprt(Fail2Check, "The first layer and last layer should be equal spacing");
	for(i=1;i<dims[2];i++)
		if( (!equalspacing[i-1]) && (!equalspacing[i]) )
		errprt(Fail2Check, "There is only one layer(in two adjacent layers) need to adjust the grid spacing");
	if( mathf.sum(gridpoints,dims[2]-1)+1 != cdx.nk)
		errprt(Fail2Check, "The Z value number[ sum(gridpoints)+1 ] obtained from gridfile is different with configure number(cdx.nk)");
	for(i=0;i<dims[0];i++)
		for(j=0;j<dims[1];j++)
			for(k=1;k<dims[2];k++)
				if( posz[i][j][k]>posz[i][j][k-1] )
				{
					sprintf(errstr,"The Z direction depth should be monotonically decreasing, while position[%d][%d][%d]>[%d][%d][%d]",
					             i,j,k,i,j,k-1," the value is" Rformat Rformat Rformat Rformat ".\n",posx[i],posy[j],posz[i][j][k],
						     posz[i][j][k-1]);
					errprt(Fail2Check, errstr);
				}
	
	//PPW calculating
	FILE *fps;
	char parpath[SeisStrLen];
	char srcfile[SeisStrLen];
	char name[SeisStrLen2];
	int nfrc, nmnt, nwin;
	Real tempfreq, maxfreq;
	Real RefVel, MaxInterval;

	fps = fopen(filename,"r");
	com.get_conf(fps, "seispath", 3, parpath);
	com.get_conf(fps, "source_filename", 3, name);
	com.get_conf(fps, "reference_velocity",3, &RefVel);
	fclose(fps);

	sprintf(srcfile,"%s/%s",parpath,name);

	fps = fopen(srcfile,"r");
	if(!fps)
	{
		sprintf(errstr,"fail to open source par file %s in gridmesh.cpp",srcfile);
		errprt(Fail2Open,errstr);
	}

	com.get_conf(fps,"number_of_force_source",3,&nfrc);
	com.get_conf(fps,"number_of_moment_source",3,&nmnt);
	maxfreq = -1.0;

	if(nfrc)
	{
		com.get_conf(fps,"force_stf_window",3,&nwin);

		for(i=0;i<nfrc;i++)
			for(j=0;j<nwin;j++)
			{
				com.get_conf(fps,"force_stf_freqfactor",3+j,&tempfreq);
				maxfreq = MAX(maxfreq, tempfreq);
			}
	}
	if(nmnt)
	{
		com.get_conf(fps,"moment_stf_window",3,&nwin);
		for(i=0;i<nmnt;i++)
			for(j=0;j<nwin;j++)
			{
				com.get_conf(fps,"moment_stf_freqfactor",3+j,&tempfreq);
				maxfreq = MAX(maxfreq,tempfreq);
			}
	}
	fclose(fps);

	//PPW conputes max intervals
	MaxInterval = RefVel/(2.0*maxfreq*8);
	fprintf(stdout,"The min Vs = %g m/s and the max frequency (2*f0) = %g Hz,"
			"for 8 grid point per wavelegth the maximum interval is %g m"
			" and the RECOMMANDED interval(steph) is %g m\n",RefVel,2.0*maxfreq,MaxInterval,steph);
	
	//Z direction value generating
	fprintf(stdout,"***Start to generate the curvilinear coordinate in gridmesh porgram\n");

	Real GinterfaceZrange[ dims[2] ][2];
	Real GlayerZrange[ dims[2]-1 ][2];
	Real Zmin,Zmax;
	//interface
	for(k=0;k<dims[2];k++)
	{
		Zmax = posz[0][0][k];
		Zmin = Zmax;
		for(i=0;i<dims[0];i++)
			for(j=0;j<dims[1];j++)
			{
				Zmax = MAX(posz[i][j][k],Zmax);
				Zmin = MIN(posz[i][j][k],Zmin);
			}
		GinterfaceZrange[k][0] = Zmin;
		GinterfaceZrange[k][1] = Zmax;
	}
	//layer
	for(k=0;k<dims[2]-1;k++)
	{
		Zmax = posz[0][0][k];
		Zmin = posz[0][0][k+1];
		for(i=0;i<dims[0];i++)
			for(j=0;j<dims[1];j++)
			{
				Zmax = MAX(posz[i][j][k],Zmax);
				Zmin = MIN(posz[i][j][k+1],Zmin);
			}
		GlayerZrange[k][0] = Zmin;
		GlayerZrange[k][1] = Zmax;
	}

	fprintf(stdout,"Confirm Gird-Z-dir interface range information\n");
	for(k=0;k<dims[2];k++)
		fprintf(stdout,"\tinterface[%d] range from %lf to %lf\n",k,GinterfaceZrange[k][0],GinterfaceZrange[k][1]);

	fprintf(stdout,"Confirm Grid-Zdir layer range information\n");
	for(k=0;k<dims[2]-1;k++)
		fprintf(stdout,"\tlayer[%d] range from %lf to %lf\n",k,GlayerZrange[k][0],GlayerZrange[k][1]);
	
	if(this->HyGrid)
	{
		fprintf(stdout,"---->>>Will applying hybrid grid techniques, the conversion depth is %g, "
			       "above this interface using curvilinear grid, below this using rectagular grid\n",ConversDepth);

		//confirm ConverDepth layer location and interface location
		int ConInterface;
		Real TopInterface,BottomInterface;//last layer depth
		for(k=0;k<dims[2]-1;k++)
		{
			if(ConversDepth >= GlayerZrange[k][0] && ConversDepth <= GlayerZrange[k][1])
			{
				fprintf(stdout,"\tConversDepth(%gm) locates in layer[%d] between %gm and %gm\n",
						ConversDepth,k,GlayerZrange[k][0],GlayerZrange[k][1]);
				ConInterface = k+1;//ConLayer(contain as bottom)=ConInterface-1
				TopInterface = ConversDepth;
				BottomInterface = GinterfaceZrange[ dims[2]-1 ][0];
				fprintf(stdout,"\tset ConversInterface at interface[%d](start from 0) and adjust last layer ranges from %gm to %gm\n",
						ConInterface,TopInterface,BottomInterface);
				break;
				//locates in Layer K
				//total K+1 layer
				//Layer[K]-lower interface = constant ConversDepth
				//Layer[K+1]-upper interface = constant ConversDepth, lower interface = minimal value of orginal last interface
				//total K+2 interface
				//Interface[K]-upper interface of LayerK, keep unchanged
				//Interface[K+1]-lower of LayerK and upper of LayerK+1, set to constant ConversDepth
				//Interface[K+2]-lower of layerK+1, set to the minimal value of last interface
			}
		}

		Real NewInt[ConInterface+1+1][2],NewLayer[ConInterface+1][2];//total ConInterface+2 interface(0---CI+1), ConInterface+1 layer(0----Ci+1)
		Real ***Zdep;//new Z position(above convers interface)
		Zdep = new Real**[dims[0]];
		for(i=0;i<dims[0];i++)
		{
			Zdep[i] = new Real*[dims[1]];
			for(j=0;j<dims[1];j++)
				Zdep[i][j] = new Real[ConInterface];//all XY range
		}

		for(i=0;i<dims[0];i++)
			for(j=0;j<dims[1];j++)
				for(k=0;k<ConInterface;k++)
					Zdep[i][j][k] = posz[i][j][k];

		for(k=0;k<ConInterface;k++)
		{
			NewInt[k][0] = GinterfaceZrange[k][0];
			NewInt[k][1] = GinterfaceZrange[k][1];
		}
		NewInt[ConInterface][0] = TopInterface;
		NewInt[ConInterface][1] = NewInt[ConInterface][0];
		NewInt[ConInterface+1][0] = BottomInterface;
		NewInt[ConInterface+1][1] = NewInt[ConInterface+1][0];

		for(k=0;k<ConInterface;k++)
		{
			NewLayer[k][0] = GlayerZrange[k][0];
			NewLayer[k][1] = GlayerZrange[k][1];
		}
		NewLayer[ConInterface-1][0] = TopInterface;
		NewLayer[ConInterface][0] = BottomInterface;
		NewLayer[ConInterface][1] = TopInterface;

		fprintf(stdout,"New-Gird-Z-dir interface range information\n");
		for(k=0;k<ConInterface+2;k++)
			fprintf(stdout,"\tinterface[%d] range from %lf to %lf\n",k,NewInt[k][0],NewInt[k][1]);

		fprintf(stdout,"New-Grid-Zdir layer range information\n");
		for(k=0;k<ConInterface+1;k++)
			fprintf(stdout,"\tlayer[%d] range from %lf to %lf\n",k,NewLayer[k][0],NewLayer[k][1]);
		
		int GPs[ConInterface+1];
		Real deltaZ[ConInterface+1];
		int indexZ[ConInterface+2];
		Real totalL,partL;
		
		totalL=0;
		for(k=0;k<ConInterface+1;k++)
			totalL += NewLayer[k][1]-NewLayer[k][0];
		GPs[ConInterface] = cdx.nk;
		for(k=0;k<ConInterface+1;k++)
		{
			partL = NewLayer[k][1]-NewLayer[k][0];
			if(k<ConInterface)
			{
				GPs[k] = cdx.nk*partL/totalL;
				GPs[ConInterface] = GPs[ConInterface]-GPs[k];//last layer
			}
			deltaZ[k] = partL/GPs[k];
			printf("partL=%g,totalL=%g,GP=%d,deltaZ=%g\n",partL,totalL,GPs[k],deltaZ[k]);
		}

		indexZ[ConInterface+1] = cdx.nk1;
		for(k=ConInterface;k>=0;k--)
			indexZ[k] = indexZ[k+1] + GPs[k];
		for(k=0;k<ConInterface+2;k++)
			printf("indexZ[%d]=%d\n",k,indexZ[k]);

		this->ConIndex = indexZ[ConInterface+1]+GPs[ConInterface];
		fprintf(stdout,"ConversDepth's related Z-dir location is %d(to Max is curved, to Min is straight)"
			       ", interface index is %d, layer index (as bottom) is %d\n",ConIndex,ConInterface,ConInterface-1);

		
		Real zvec[cdx.nz];
		Real zstep0,zstep1;
		int zidx1,zidx2;//store the layer interface index of Z
		int m;
		Real diff,maxdiff,mindiff;
		maxdiff=0;	mindiff = MaxInterval;

		
		if(ConInterface+1 == 2)//two layer case
		{
			for(i=cdx.ni1;i<cdx.ni2;i++)
				for(j=cdx.nj1;j<cdx.nj2;j++)
				{
					//store X and Y
					for(k=cdx.nk1;k<cdx.nk2;k++)
					{
						crd.x[i][j][k] = posx[i-cdx.ni1][j-cdx.nj1];//valid position
						crd.y[i][j][k] = posy[i-cdx.ni1][j-cdx.nj1];
					}
					//interpolate Z
					for(k=0;k<ConInterface+1;k++)
					{
						zidx2 = indexZ[k]-1;//big (outter points)
						zidx1 = indexZ[k+1];//small
						
						if(k)//bottom layer
						{
							zstep0 = (NewLayer[k][1] - NewLayer[k][0])/(1.0*GPs[k]);
							for(m=0;m<GPs[k];m++)
								zvec[m+zidx1] = NewLayer[k][0] + m*zstep0;
						}
						else//top layer
						{
							zstep0 = ( Zdep[i-cdx.ni1][j-cdx.nj1][k] - NewLayer[k][0] ) / (1.0*GPs[k]);
							zstep1 = ( NewLayer[k+1][1] - NewLayer[k+1][0] ) / (1.0*GPs[k+1]);
							//there is a transtion zone between two fixed spacing layer. In this zone, the spacing should
							//change from zstep0(up) to zstep1(below).
							smooth_zdepth(&zvec[zidx1],Zdep[i-cdx.ni1][j-cdx.nj1][k],NewLayer[k][0],
									zstep0,zstep1,GPs[k]-1);//GP contains two side
						}
					}//end loop of layer

					for(k=cdx.nk1;k<cdx.nk2;k++)
						crd.z[i][j][k] = zvec[k];
					//difference check
					for(k=cdx.nk1;k<cdx.nk2-1;k++)
					{
						diff=ABS(zvec[k]-zvec[k+1]);
						mindiff = MIN(mindiff,diff);
						maxdiff = MAX(maxdiff,diff);
					}
				}
			printf("maxdiff=%g, mindiff=%g\n",maxdiff,mindiff);
		}//end loop of two layer case
		else
		{
			for(i=cdx.ni1;i<cdx.ni2;i++)
				for(j=cdx.nj1;j<cdx.nj2;j++)
				{
					//store X and Y
					for(k=cdx.nk1;k<cdx.nk2;k++)
					{
						crd.x[i][j][k] = posx[i-cdx.ni1][j-cdx.nj1];//valid position
						crd.y[i][j][k] = posy[i-cdx.ni1][j-cdx.nj1];
					}
					//interpolate Z
					for(k=0;k<ConInterface+1;k++)//layer loop
					{
						zidx2 = indexZ[k]-1;//big (outter points)
						zidx1 = indexZ[k+1];//small
						
						if(k<ConInterface-1)//above ConversLayer
						{
							if(!k)//Layer 0
								zstep0 = (Zdep[i-cdx.ni1][j-cdx.nj1][k] - Zdep[i-cdx.ni1][j-cdx.nj1][k+1])/(1.0*GPs[k]);
							else
								zstep0 = (Zdep[i-cdx.ni1][j-cdx.nj1][k-1] - Zdep[i-cdx.ni1][j-cdx.nj1][k])/(1.0*GPs[k-1]);
							
							if(k==ConInterface-2)//ConversLayer
								zstep1 = (Zdep[i-cdx.ni1][j-cdx.nj1][k+1] - NewLayer[k+1][0])/(1.0*GPs[k+1]);
							else
								zstep1 = (Zdep[i-cdx.ni1][j-cdx.nj1][k+1] - Zdep[i-cdx.ni1][j-cdx.nj1][k+2])/(1.0*GPs[k+1]);
							
							smooth_zdepth(&zvec[zidx1],Zdep[i-cdx.ni1][j-cdx.nj1][k],Zdep[i-cdx.ni1][j-cdx.nj1][k+1]+zstep1,
									zstep0,zstep1,GPs[k]-1);
							zvec[zidx2] = Zdep[i-cdx.ni1][j-cdx.nj1][k];
						}
						else
						{
							if(k==ConInterface)//bottom layer
							{
								zstep0 = (NewLayer[k][1] - NewLayer[k][0])/(1.0*GPs[k]);
								for(m=0;m<GPs[k];m++)
									zvec[m+zidx1] = NewLayer[k][0] + m*zstep0;
								zvec[zidx1] = NewLayer[k][0];
							}
							else//top layer
							{
								zstep0 = ( Zdep[i-cdx.ni1][j-cdx.nj1][k] - NewLayer[k][0] ) / (1.0*GPs[k]);
								zstep1 = ( NewLayer[k+1][1] - NewLayer[k+1][0] ) / (1.0*GPs[k+1]);
								smooth_zdepth(&zvec[zidx1],Zdep[i-cdx.ni1][j-cdx.nj1][k],NewLayer[k][0],
										zstep0,zstep1,GPs[k]-1);//GP contains two side
								zvec[zidx2] = Zdep[i-cdx.ni1][j-cdx.nj1][k];
							}
						}
					}//end loop of layer

					for(k=cdx.nk1;k<cdx.nk2;k++)
						crd.z[i][j][k] = zvec[k];
					//difference check
					for(k=cdx.nk1;k<cdx.nk2-1;k++)
					{
						diff=ABS(zvec[k]-zvec[k+1]);
						if(diff>steph && diff>maxdiff)
							maxdiff=diff;
						if(diff<mindiff)
							mindiff = diff;
					}
				}
			cout<<endl;
			printf("maxdiff=%g, mindiff=%g\n",maxdiff,mindiff);
		}//end loop of multiple layer case

		if( maxdiff > MaxInterval)
		{
			sprintf(errstr,"The actual maxmium grid distance (%f m) exceed the PPW limitation (%f m) for HyGrid,"
				       " Please increase the dominant frequency or adjust the model\n",
				       maxdiff,MaxInterval);
			errprt(Fail2Check,errstr);
		}

		
		for(i=0;i<dims[0];i++)
		{
			for(j=0;j<dims[1];j++)
				delete [] Zdep[i][j];
			delete [] Zdep[i];
		}
		delete [] Zdep;
	}
	else
	{
		fprintf(stdout,"---->>>Will applying curvilinear grid for all range\n");
		this->ConIndex = cdx.nk1;
		
		int *zindex;
		zindex = new int[dims[2]];
		zindex[dims[2]-1] = cdx.nk1; //confirm the Z index is reverse, the bottom is index:0, the top is index:nk-1 
		for(i=dims[2]-2;i>=0;i--)
			zindex[i] = zindex[i+1] + gridpoints[i];
		// 4 surface ; 3 layer ; 3 gridpoints ; gridpoints index 0-2;
		// gridpoints 2,1,0;  2= 4 - 2;

		int zidx1,zidx2;//store the layer interface index of Z
		Real zstep0,zstep1;
		Real zvec[cdx.nz];
		int m;

		for(i=cdx.ni1;i<cdx.ni2;i++)
		{
			for(j=cdx.nj1;j<cdx.nj2;j++)
			{
				//store X and Y
				for(k=cdx.nk1;k<cdx.nk2;k++)
				{
					crd.x[i][j][k] = posx[i-cdx.ni1][j-cdx.nj1];//valid position
					crd.y[i][j][k] = posy[i-cdx.ni1][j-cdx.nj1];
				}

				//interpoalte Z
				for(k=0;k<dims[2]-1;k++)
				{
					zidx2 = zindex[k];//big
					zidx1 = zindex[k+1];//small
					if( equalspacing[k] )
					{
						zstep0 = ( posz[i-cdx.ni1][j-cdx.nj1][k] - posz[i-cdx.ni1][j-cdx.nj1][k+1] )/ (1.0*gridpoints[k]);
						// posz[k]-posz[k+1] = positive
						// m-zidx2 = negetive
						// zvec[small to big] = value[big to small]=[0 to negetive] 
						// zindex[0] big, zindex[dims[2]-1] small=cdx.nk1
						// posz[0] big=0, posz[dims[2]-1] small=-10000
						for( m=zidx1; m <= zidx2; m++)
							zvec[m] = posz[i-cdx.ni1][j-cdx.nj1][k] + (m - zidx2)*zstep0;
					}
					else
					{
						zstep0 = ( posz[i-cdx.ni1][j-cdx.nj1][k-1] - posz[i-cdx.ni1][j-cdx.nj1][k] ) / (1.0*gridpoints[k-1]);
						zstep1 = ( posz[i-cdx.ni1][j-cdx.nj1][k+1] - posz[i-cdx.ni1][j-cdx.nj1][k+2] ) / (1.0*gridpoints[k+1]);
						//there is a transtion zone between two fixed spacing layer. In this zone, the spacing should
						//change from zstep0 to zstep1.
						smooth_zdepth(&zvec[zindex[k+1]],posz[i-cdx.ni1][j-cdx.nj1][k],posz[i-cdx.ni1][j-cdx.nj1][k+1],
								zstep0,zstep1,gridpoints[k]);
					}
					zvec[zidx2] = posz[i-cdx.ni1][j-cdx.nj1][k];
					zvec[zidx1] = posz[i-cdx.ni1][j-cdx.nj1][k+1];

				}
				for(k=cdx.nk1;k<cdx.nk2;k++)
				{
					crd.z[i][j][k] = zvec[k];
				}
			}
		}

	delete [] zindex;
	}
	
	
	/*
	//fixed X Y intervals
	for(i=0;i<dims[0];i++)
	{
		for(j=0;j<dims[1];j++)
			delete [] posz[i][j];
		delete [] posz[i];
	}
	delete [] posz;
	delete [] posy;
	delete [] posx;
	*/
	//varying X Y intervals
	for(i=0;i<dims[0];i++)
	{
		for(j=0;j<dims[1];j++)
		{
			delete [] posz[i][j];
		}
		delete [] posz[i];
		delete [] posy[i];
		delete [] posx[i];
	}
	delete [] posz;
	delete [] posy;
	delete [] posx;
	delete [] equalspacing;
	delete [] gridpoints;
	
	fprintf(stdout,"---accomplished generating the curvilinear coordinate in gridmesh porgram\n");
}

void gridmesh::smooth_zdepth(Real *z, Real za, Real zb, Real ha, Real hb, int n)
{
  Real l, h0, h[n + 2];
  l = za - zb;
  h0 = l/n;
  h[0] = hb; h[n + 1] = ha;

  int na, nb, n0;
  Real l0, f;
  Real ha0, hb0;
  Real ha1, ha2, hb1, hb2;
  Real la1, la2, lb1, lb2;

  
  int k;
  if((h0 >= ha && h0 <= hb) || (h0 <= ha && h0 >= hb))
  {
    nb = 1; na = n; n0 = n;
    while(1)
    {
      l0 = 0.0;
      f = pow((ha/hb), (double)1.0/(n0 + 1)) - 1.0;
      for(k = 1; k <= nb - 1; k++)
        h[k] = hb;
      for(k = na + 1; k <= n + 1; k++)
        h[k] = ha;
      for(k = nb; k <= na; k++)
      {
        h[k] = h[k - 1]*(1 + f);
        l0 += h[k];
      }
      ha1 = ha*MIN(1.0, 1.0 - f);
      ha2 = ha*MAX(1.0, 1.0 - f);
      hb1 = hb*MIN(1.0, 1.0 + f);
      hb2 = hb*MAX(1.0, 1.0 + f);
      la1 = (n - na)*ha1;
      la2 = (n - na)*ha2;
      lb1 = (nb - 1)*hb1;
      lb2 = (nb - 1)*hb2;
      if((la1 + lb1 <= l - l0) && (l - l0 <= la2 + lb2))
      {
        set_twoside_space(l - l0, n - na, nb - 1, ha1, ha2, hb1, hb2,
          &ha0, &hb0);
        for(k = na + 1; k <= n; k++)
          h[k] = ha0;
        for(k = 1; k <= nb - 1; k++)
          h[k] = hb0;
        break;
      }
      else if(l - l0 < la1 + lb1)
        if(ha < hb)
          {na--; n0--;}
        else
          {nb++; n0--;}
      else
        if(ha < hb)
          {nb++; n0--;}
        else
          {na--; n0--;}
    }
  }
  else
  {
    na = n/4;
    nb = n/4;
    n0 = n - na - nb;
    la1 = (na + 1)*na/2.0;
    lb1 = (nb + 1)*nb/2.0;
    la2 = la1 + na*lb1/nb + na*n0;
    l0 = l - na*ha - nb*hb - n0*ha - la1*(ha - hb)/nb;
    ha0 = l0/la2;
    hb0 = (na*ha0 + ha - hb)/nb;
    for(k = 1; k <= nb; k++)
      h[k] = hb + k*hb0;
    for(k = nb + 1; k <= nb + n0; k++)
      h[k] = h[k - 1];
    for(k = nb + n0 + 1; k <= n; k++)
      h[k] = ha + (n - k + 1)*ha0;
  }
  z[0] = 0.0;
  for(k = 0; k < n; k++)
  {
    z[k + 1] = z[k] + h[k + 1];
  }
  for(k=0;k<n+1;k++)
  {
	  z[k]=z[k]+zb;
  }
  

}

void gridmesh::set_twoside_space(Real l, int na, int nb, Real ha1, Real ha2, Real hb1, Real hb2, Real *ha0, Real *hb0)
{
  Real l0, la, lb;
  l0 = l - na*ha1 - nb*hb1;
  la = na*(ha2 - ha1);
  lb = nb*(hb2 - hb1);
  if(ABS(l0) < SeisEqual)
  {
    *ha0 = ha1;
    *hb0 = hb1;
  }
  else if(ABS(la) < SeisEqual)
  {
    *ha0 = ha1;
    *hb0 = hb1 + l0/nb;
  }
  else if(ABS(lb) < SeisEqual)
  {
    *ha0 = ha1 + l0/na;
    *hb0 = hb1;
  }
  else
  {
    *ha0 = ha1 + l0*la/(la + lb)/na;
    *hb0 = hb1 + l0*lb/(la + lb)/nb;
  }
}

void gridmesh::GridCheck(cindx cdx, Real steph)
{
	fprintf(stdout,"***start to check grid model's plausibility in gridmesh porgram\n");
	
	int i,j,k;
	
	Real der21,der22;
	Real der11,der12;
	int conflag,Pcount;

	//z_xi
	Pcount = 0;
	for(i=cdx.ni1;i<cdx.ni2-1;i++)
	{
		for(j=cdx.nj1;j<cdx.nj2;j++)
		{
			for(k=cdx.nk1;k<cdx.nk2;k++)
			{
				der11 = (crd.z[i+1][j][k]-crd.z[i-1][j][k])/2.0/steph;
				der12 = (crd.z[i+2][j][k]-crd.z[i][j][k])/2.0/steph;
				der21 = (crd.z[i+1][j][k]-2*crd.z[i][j][k]+crd.z[i-1][j][k])/2.0/steph;
				der22 = (crd.z[i+2][j][k]-2*crd.z[i+1][j][k]+crd.z[i][j][k])/2.0/steph;
				if(der21*der22<0 && der12/der11>=1.5 && der11 !=0)
				{
					printf("at(%d,%d,%d),crd[%d]=%10.6f,crd[%d]=%10.6f,crd[%d]=%10.6f,crd[%d]=%10.6f\n\t\t\tfirst-order=%g,%g,second-order=%g,%g\n",
					i,j,k, i-1,crd.z[i-1][j][k],i,crd.z[i][j][k],i+1,crd.z[i+1][j][k],i+2,crd.z[i+2][j][k],der11,der12,der21,der22);
					conflag=1;
					Pcount++;
					break;
				}
			}
			if(conflag)
			{
				conflag=0;
				break;
			}
		}
	}
	if(Pcount) fprintf(stdout,"Check z_xi's plausibility %d lines need check again\n",Pcount);
	
	//z_eta
	Pcount = 0;
	for(i=cdx.ni1;i<cdx.ni2;i++)
	{
		for(j=cdx.nj1;j<cdx.nj2-1;j++)
		{
			for(k=cdx.nk1;k<cdx.nk2;k++)
			{
				der11 = (crd.z[i][j+1][k]-crd.z[i][j-1][k])/2.0/steph;
				der12 = (crd.z[i][j+2][k]-crd.z[i][j][k])/2.0/steph;
				der21 = (crd.z[i][j+1][k]-2*crd.z[i][j][k]+crd.z[i][j-1][k])/2.0/steph;
				der22 = (crd.z[i][j+2][k]-2*crd.z[i][j+1][k]+crd.z[i][j][k])/2.0/steph;
				if(der21*der22<0 && der12/der11>=1.5 && der11 !=0)
				{
					printf("at(%d,%d,%d),crd[%d]=%10.6f,crd[%d]=%10.6f,crd[%d]=%10.6f,crd[%d]=%10.6f\n\t\t\tfirst-order=%g,%g,second-order=%g,%g\n",
					i,j,k, j-1,crd.z[i][j-1][k],j,crd.z[i][j][k],j+1,crd.z[i][j+1][k],j+2,crd.z[i][j+2][k],der11,der12,der21,der22);
					conflag=1;
					Pcount++;
					break;
				}
			}
			if(conflag)
			{
				conflag=0;
				break;
			}
		}
	}
	if(Pcount) fprintf(stdout,"Check z_eta's plausibility %d lines need check again\n",Pcount);
	
	//z_zeta
	Pcount = 0;
	for(i=cdx.ni1;i<cdx.ni2;i++)
	{
		for(j=cdx.nj1;j<cdx.nj2;j++)
		{
			for(k=cdx.nk1;k<cdx.nk2-1;k++)
			{
				der11 = (crd.z[i][j][k+1]-crd.z[i][j][k-1])/2.0/steph;
				der12 = (crd.z[i][j][k+2]-crd.z[i][j][k])/2.0/steph;
				der21 = (crd.z[i][j][k+1]-2*crd.z[i][j][k]+crd.z[i][j][k-1])/2.0/steph;
				der22 = (crd.z[i][j][k+2]-2*crd.z[i][j][k+1]+crd.z[i][j][k])/2.0/steph;
				if(der21*der22<0 && der12/der11>=1.5 && der11 !=0)
				{
					printf("at(%d,%d,%d),crd[%d]=%10.6f,crd[%d]=%10.6f,crd[%d]=%10.6f,crd[%d]=%10.6f\n\t\t\tfirst-order=%g,%g,second-order=%g,%g\n",
					i,j,k, k-1,crd.z[i][j][k-1],k,crd.z[i][j][k],k+1,crd.z[i][j][k+1],k+2,crd.z[i][j][k+2],der11,der12,der21,der22);
					conflag=1;
					Pcount++;
					break;
				}
			}
			if(conflag)
			{
				conflag=0;
				break;
			}
		}
	}
	if(Pcount) fprintf(stdout,"Check z_zeta's plausibility %d lines need check again\n",Pcount);
	
	fprintf(stdout,"---accomplished to check grid model's plausibility in gridmesh porgram\n");
}


//-------------------------------------public------------------------
gridmesh::gridmesh(const char *filename, cindx cdx, const int restart, const int Myid)
{
	myid = Myid;
	if(myid)
	{
		Mflag = false;//child procs
		printf("child procs %d doesn't paticipate into computing parameters (grid)\n",myid);
		return;
	}
	else
		Mflag = true;//master procs

	getConf(filename);
	if(restart == 1)
		Rwork = true; //restart work, reading the exists.
	else
		Rwork = false;

	//declared before in the struct;
	//cdx.nx cdx.ny cdx.nz
	//   X	   Y      Z
	//   low   mid   fast
	//   [x][y][z]
	int i,j;
	Dnx = cdx.nx;
	Dny = cdx.ny;
	Dnz = cdx.nz;
	
	crd.x = new Real**[Dnx];
	crd.y = new Real**[Dnx];
	crd.z = new Real**[Dnx];
	drv.xi_x = new Real **[Dnx];
	drv.xi_y = new Real **[Dnx];
	drv.xi_z = new Real **[Dnx];
	drv.eta_x = new Real **[Dnx];
	drv.eta_y = new Real **[Dnx];
	drv.eta_z = new Real **[Dnx];
	drv.zeta_x = new Real **[Dnx];
	drv.zeta_y = new Real **[Dnx];
	drv.zeta_z = new Real **[Dnx];
	drv.jac = new Real **[Dnx];
	for(i=0;i<Dnx;i++)
	{
		crd.x[i] = new Real*[Dny];
		crd.y[i] = new Real*[Dny];
		crd.z[i] = new Real*[Dny];
		drv.xi_x[i] = new Real *[Dny];
		drv.xi_y[i] = new Real *[Dny];
		drv.xi_z[i] = new Real *[Dny];
		drv.eta_x[i] = new Real *[Dny];
		drv.eta_y[i] = new Real *[Dny];
		drv.eta_z[i] = new Real *[Dny];
		drv.zeta_x[i] = new Real *[Dny];
		drv.zeta_y[i] = new Real *[Dny];
		drv.zeta_z[i] = new Real *[Dny];
		drv.jac[i] = new Real *[Dny];
		for(j=0;j<Dny;j++)
		{
			crd.x[i][j] = new Real[Dnz]();
			crd.y[i][j] = new Real[Dnz]();
			crd.z[i][j] = new Real[Dnz]();
			drv.xi_x[i][j] = new Real [Dnz]();
			drv.xi_y[i][j] = new Real [Dnz]();
			drv.xi_z[i][j] = new Real [Dnz]();
			drv.eta_x[i][j] = new Real [Dnz]();
			drv.eta_y[i][j] = new Real [Dnz]();
			drv.eta_z[i][j] = new Real [Dnz]();
			drv.zeta_x[i][j] = new Real [Dnz]();
			drv.zeta_y[i][j] = new Real [Dnz]();
			drv.zeta_z[i][j] = new Real [Dnz]();
			drv.jac[i][j] = new Real [Dnz]();
		}
	}
}

gridmesh::~gridmesh()
{
	fprintf(stdout,"into data free at Procs[%d],in gridmesh.cpp\n",myid);
	if(this->Mflag)
	{
		for(int i=0;i<Dnx;i++)
		{
			for(int j=0;j<Dny;j++)
			{
				delete [] drv.jac[i][j];
				delete [] drv.zeta_z[i][j];
				delete [] drv.zeta_y[i][j];
				delete [] drv.zeta_x[i][j];
				delete [] drv.eta_z[i][j];
				delete [] drv.eta_y[i][j];
				delete [] drv.eta_x[i][j];
				delete [] drv.xi_z[i][j];
				delete [] drv.xi_y[i][j];
				delete [] drv.xi_x[i][j];
				delete [] crd.z[i][j];
				delete [] crd.y[i][j];
				delete [] crd.x[i][j];
			}
			delete [] drv.jac[i];
			delete [] drv.zeta_z[i];
			delete [] drv.zeta_y[i];
			delete [] drv.zeta_x[i];
			delete [] drv.eta_z[i];
			delete [] drv.eta_y[i];
			delete [] drv.eta_x[i];
			delete [] drv.xi_z[i];
			delete [] drv.xi_y[i];
			delete [] drv.xi_x[i];
			delete [] crd.z[i];
			delete [] crd.y[i];
			delete [] crd.x[i];
		}
		delete [] drv.jac;
		delete [] drv.zeta_z;
		delete [] drv.zeta_y;
		delete [] drv.zeta_x;
		delete [] drv.eta_z;
		delete [] drv.eta_y;
		delete [] drv.eta_x;
		delete [] drv.xi_z;
		delete [] drv.xi_y;
		delete [] drv.xi_x;
		delete [] crd.z;
		delete [] crd.y;
		delete [] crd.x;
	}
	fprintf(stdout,"data free at Procs[%d],in gridmesh.cpp\n",myid);
}
	
void gridmesh::readdata(cindx cdx, const char *filename, Real steph)
{
	if(this->Rwork)
	{
		fprintf(stdout,"---Reading the exists coord's data, there's no needs to read from ordinary one ,due to the restart work\n");
		return;
	}

	if(ISEQSTR(gridtype,"vmap"))
		read_vmapnew(cdx, filename, steph);
	else if(ISEQSTR(gridtype,"point"))
		read_point(cdx);//nc file
	else
	{
		char errstr[SeisStrLen];
		sprintf(errstr,"gridtype configure %s is wrong when reading grid data",gridtype);
		errprt(Fail2Check,errstr);
	}

	//printf("readdata before inter -crd xyz=%f,%f,%f\n",crd.x[0][0][0],crd.y[0][0][0],crd.z[0][0][0]);
	
	com.interpolated_extend(crd.x,cdx);
	com.interpolated_extend(crd.y,cdx);
	com.interpolated_extend(crd.z,cdx);
	
	//open if need
	//GridCheck(cdx, steph);
	
}

void gridmesh::calmetric(Real steph, cindx cdx)
{
	if(this->Rwork)
	{
		fprintf(stdout,"---Reading the exists deriv's data, there's no needs to read from ordinary one ,due to the restart work\n");
		return;
	}
	
	int i,j,k,idx;
	Real *x_xi,*y_xi,*z_xi,*x_eta,*y_eta,*z_eta,*x_zeta,*y_zeta,*z_zeta;
	
	fprintf(stdout,"***Start to calculate the partial derivative in gridmesh porgram\n");

	//5 point first order derivative 1/8 1/4 -1/4 -1/8
	
	x_xi = new Real[Dnx*Dny*Dnz]();
	y_xi = new Real[Dnx*Dny*Dnz]();
	z_xi = new Real[Dnx*Dny*Dnz]();
	x_eta = new Real[Dnx*Dny*Dnz]();
	y_eta = new Real[Dnx*Dny*Dnz]();
	z_eta = new Real[Dnx*Dny*Dnz]();
	x_zeta = new Real[Dnx*Dny*Dnz]();
	y_zeta = new Real[Dnx*Dny*Dnz]();
	z_zeta = new Real[Dnx*Dny*Dnz]();
	Real vec1[3],vec2[3],vec3[3],vecg[3];
	for(i=cdx.ni1;i<cdx.ni2;i++)
		for(j=cdx.nj1;j<cdx.nj2;j++)
			for(k=cdx.nk1;k<cdx.nk2;k++)
			{
				idx = i*Dny*Dnz + j*Dnz + k;

				//calculate deritative of X to Xi, equation 2.2 in Thesis
				
				//3 point
				x_xi[idx] = ( crd.x[i+1][j][k] - crd.x[i-1][j][k] )/2.0/steph;
				y_xi[idx] = ( crd.y[i+1][j][k] - crd.y[i-1][j][k] )/2.0/steph;
				z_xi[idx] = ( crd.z[i+1][j][k] - crd.z[i-1][j][k] )/2.0/steph;
				
				x_eta[idx] = ( crd.x[i][j+1][k] - crd.x[i][j-1][k] )/2.0/steph;
				y_eta[idx] = ( crd.y[i][j+1][k] - crd.y[i][j-1][k] )/2.0/steph;
				z_eta[idx] = ( crd.z[i][j+1][k] - crd.z[i][j-1][k] )/2.0/steph;
				
				x_zeta[idx] = ( crd.x[i][j][k+1] - crd.x[i][j][k-1] )/2.0/steph;
				y_zeta[idx] = ( crd.y[i][j][k+1] - crd.y[i][j][k-1] )/2.0/steph;
				z_zeta[idx] = ( crd.z[i][j][k+1] - crd.z[i][j][k-1] )/2.0/steph;
				
				/*
				//default
				x_xi[idx] =  crd.x[i+1][j][k] - crd.x[i-1][j][k] ;
				y_xi[idx] =  crd.y[i+1][j][k] - crd.y[i-1][j][k] ;
				z_xi[idx] =  crd.z[i+1][j][k] - crd.z[i-1][j][k] ;
				
				x_eta[idx] =  crd.x[i][j+1][k] - crd.x[i][j-1][k] ;
				y_eta[idx] =  crd.y[i][j+1][k] - crd.y[i][j-1][k] ;
				z_eta[idx] =  crd.z[i][j+1][k] - crd.z[i][j-1][k] ;
				
				x_zeta[idx] =  crd.x[i][j][k+1] - crd.x[i][j][k-1] ;
				y_zeta[idx] =  crd.y[i][j][k+1] - crd.y[i][j][k-1] ;
				z_zeta[idx] =  crd.z[i][j][k+1] - crd.z[i][j][k-1] ;
				*/
				/*
				//5 point
				x_xi[idx] = ( crd.x[i+2][j][k] + 2*crd.x[i+1][j][k] - 2*crd.x[i-1][j][k] - crd.x[i-2][j][k] )/8.0/steph;
				y_xi[idx] = ( crd.y[i+2][j][k] + 2*crd.y[i+1][j][k] - 2*crd.y[i-1][j][k] - crd.y[i-2][j][k] )/8.0/steph;
				z_xi[idx] = ( crd.z[i+2][j][k] + 2*crd.z[i+1][j][k] - 2*crd.z[i-1][j][k] - crd.z[i-2][j][k] )/8.0/steph;
				
				x_eta[idx] = ( crd.x[i][j+2][k] + 2*crd.x[i][j+1][k] - 2*crd.x[i][j-1][k] -crd.x[i][j-2][k] )/8.0/steph;
				y_eta[idx] = ( crd.y[i][j+2][k] + 2*crd.y[i][j+1][k] - 2*crd.y[i][j-1][k] -crd.y[i][j-2][k] )/8.0/steph;
				z_eta[idx] = ( crd.z[i][j+2][k] + 2*crd.z[i][j+1][k] - 2*crd.z[i][j-1][k] -crd.z[i][j-2][k] )/8.0/steph;
				
				x_zeta[idx] = ( crd.x[i][j][k+2] + 2*crd.x[i][j][k+1] - 2*crd.x[i][j][k-1] -crd.x[i][j][k-2] )/8.0/steph;
				y_zeta[idx] = ( crd.y[i][j][k+2] + 2*crd.y[i][j][k+1] - 2*crd.y[i][j][k-1] -crd.y[i][j][k-2] )/8.0/steph;
				z_zeta[idx] = ( crd.z[i][j][k+2] + 2*crd.z[i][j][k+1] - 2*crd.z[i][j][k-1] -crd.z[i][j][k-2] )/8.0/steph;
				*/

				//calculate the Jacobian matrix, equation 2.5 in Thesis
				//accoridng to the determinant calculation rule.
				vec1[0] = x_xi[idx];   vec1[1] = y_xi[idx];   vec1[2] = z_xi[idx];
				vec2[0] = x_eta[idx];  vec2[1] = y_eta[idx];  vec2[2] = z_eta[idx];
				vec3[0] = x_zeta[idx]; vec3[1] = y_zeta[idx]; vec3[2] = z_zeta[idx];
				mathf.crossproduct(vec1,vec2,vecg);
				drv.jac[i][j][k] = mathf.dotproduct(vecg,vec3,3);
				//calculate the determinate of the Jacobi, means volume of matrix
				//first calculate two vector's crossproduct, to get its areas,but in vector 3's direction
				//then mutiply with vector 3, get the volume of 3 vectors.

				//calculate the partial derivative of Xi to X, equation 2.4 in Thesis
				//should do cross product first, then mutiply the invert of J, equation 2.2
				//mathf.crossproduct(vec1,vec2,vecg); //same with last step;
				drv.zeta_x[i][j][k] = vecg[0]/drv.jac[i][j][k];
				drv.zeta_y[i][j][k] = vecg[1]/drv.jac[i][j][k];
				drv.zeta_z[i][j][k] = vecg[2]/drv.jac[i][j][k];

				mathf.crossproduct(vec3,vec1,vecg); //same with last step;
				drv.eta_x[i][j][k] = vecg[0]/drv.jac[i][j][k];
				drv.eta_y[i][j][k] = vecg[1]/drv.jac[i][j][k];
				drv.eta_z[i][j][k] = vecg[2]/drv.jac[i][j][k];
				
				mathf.crossproduct(vec2,vec3,vecg);
				drv.xi_x[i][j][k] = vecg[0]/drv.jac[i][j][k];
				drv.xi_y[i][j][k] = vecg[1]/drv.jac[i][j][k];
				drv.xi_z[i][j][k] = vecg[2]/drv.jac[i][j][k];
				//to get the inverse matrix of the Jacobi
				//is equivalent to calculate the partial derivative of Xi to X, etc.
				//using cross product to get the adjoint matrix, then get the inverse matrix

#ifdef DisBug				
if(i==90 && j==201 && k==229)
{
	printf("vec1=%e, %e, %e\n",vec1[0], vec1[1], vec1[2]);
	printf("vec2=%e, %e, %e\n",vec2[0], vec2[1], vec2[2]);
	printf("vec3=%e, %e, %e\n",vec3[0], vec3[1], vec3[2]);
	
	printf("xi_XYZ=%e, %e, %e\n",drv.xi_x[i][j][k], drv.xi_y[i][j][k], drv.xi_z[i][j][k]);
	printf("eta_XYZ=%e, %e, %e\n",drv.eta_x[i][j][k], drv.eta_y[i][j][k], drv.eta_z[i][j][k]);
	printf("zeta_XYZ=%e, %e, %e\n",drv.zeta_x[i][j][k], drv.zeta_y[i][j][k], drv.zeta_z[i][j][k]);
	
	for(int iii=0;iii<11;iii++)
	{
		printf("%f,",crd.z[i-5+iii][j][k]);
	}
	//printf("CDR z XI: %f, %f, %f, %f, %f\n",crd.z[i+2][j][k],crd.z[i+1][j][k],crd.z[i][j][k],crd.z[i-1][j][k],crd.z[i-2][j][k]);
	//printf("CDR z ET: %f, %f, %f, %f, %f\n",crd.z[i][j+2][k],crd.z[i][j+1][k],crd.z[i][j][k],crd.z[i][j-1][k],crd.z[i][j+2][k]);
	//printf("CDR z ZT: %f, %f, %f, %f, %f\n",crd.z[i][j][k+2],crd.z[i][j][k+1],crd.z[i][j][k],crd.z[i][j][k-1],crd.z[i][j][k-2]);
	
	printf("jac=%e\n",drv.jac[i][j][k]); 

}
#endif				

				if(k<=ConIndex && (drv.zeta_x[i][j][k]!=0 || drv.zeta_y[i][j][k]!=0))
					printf("zeta_x[%d][%d][%d] = %g, zeta_y[%d][%d][%d]=%g, x_zeta=%g, y_zeta=%g\n",
						i,j,k,drv.zeta_x[i][j][k],i,j,k,drv.zeta_y[i][j][k],x_zeta[idx],y_zeta[idx]);

			}
	com.mirror_extend(drv.xi_x,  cdx);
	com.mirror_extend(drv.xi_y,  cdx);
	com.mirror_extend(drv.xi_z,  cdx);
	com.mirror_extend(drv.eta_x, cdx);
	com.mirror_extend(drv.eta_y, cdx);
	com.mirror_extend(drv.eta_z, cdx);
	com.mirror_extend(drv.zeta_x,cdx);
	com.mirror_extend(drv.zeta_y,cdx);
	com.mirror_extend(drv.zeta_z,cdx);
	com.mirror_extend(drv.jac,   cdx);
	
	delete [] z_zeta; delete [] y_zeta; delete [] x_zeta;
	delete [] z_eta; delete [] y_eta; delete [] x_eta;
	delete [] z_xi; delete [] y_xi; delete [] x_xi;
	
	fprintf(stdout,"---accomplished calculating the partial derivative in gridmesh porgram\n");
}

void gridmesh::calderivative(Real steph, cindx cdx)
{
	if(this->Rwork)
	{
		fprintf(stdout,"---Reading the exists deriv's data, there's no needs to read from ordinary one ,due to the restart work\n");
		return;
	}
	
	int i,j,k,m,idx;
	Real *x_xi,*y_xi,*z_xi,*x_eta,*y_eta,*z_eta,*x_zeta,*y_zeta,*z_zeta;
	
	fprintf(stdout,"***Start to calculate the partial derivative in gridmesh porgram\n");
	
	x_xi = new Real[Dnx*Dny*Dnz]();
	y_xi = new Real[Dnx*Dny*Dnz]();
	z_xi = new Real[Dnx*Dny*Dnz]();
	x_eta = new Real[Dnx*Dny*Dnz]();
	y_eta = new Real[Dnx*Dny*Dnz]();
	z_eta = new Real[Dnx*Dny*Dnz]();
	x_zeta = new Real[Dnx*Dny*Dnz]();
	y_zeta = new Real[Dnx*Dny*Dnz]();
	z_zeta = new Real[Dnx*Dny*Dnz]();
        Real coffF[5] = {-0.30874,-0.6326,1.2330,-0.3334,0.04168};
	Real coffB[5] = {-0.04168,0.3334,-1.2330,0.6326,0.30874};
	int incF[5] = {-1,0,1,2,3};
	int incB[5] = {-3,-2,-1,0,1};
	Real Vforward[5],Vbackward[5];
	Real vec1[3],vec2[3],vec3[3],vecg[3];
	for(i=cdx.ni1;i<cdx.ni2;i++)
		for(j=cdx.nj1;j<cdx.nj2;j++)
			for(k=cdx.nk1;k<cdx.nk2;k++)
			{
				idx = i*Dny*Dnz + j*Dnz + k;

				//calculate deritative of X to Xi, equation 2.2 in Thesis
				for(m=0;m<5;m++)
				{
					Vforward[m] = crd.x[i+incF[m]][j][k];
					Vbackward[m] = crd.x[i+incB[m]][j][k];
				}
				x_xi[idx] = ( mathf.dotproduct(Vforward,coffF,5) + mathf.dotproduct(Vbackward,coffB,5) )/2.0/steph;
				for(m=0;m<5;m++)
				{
					Vforward[m] = crd.y[i+incF[m]][j][k];
					Vbackward[m] = crd.y[i+incB[m]][j][k];
				}
				y_xi[idx] = ( mathf.dotproduct(Vforward,coffF,5) + mathf.dotproduct(Vbackward,coffB,5) )/2.0/steph;
				for(m=0;m<5;m++)
				{
					Vforward[m] = crd.z[i+incF[m]][j][k];
					Vbackward[m] = crd.z[i+incB[m]][j][k];
				}
				z_xi[idx] = ( mathf.dotproduct(Vforward,coffF,5) + mathf.dotproduct(Vbackward,coffB,5) )/2.0/steph;
				for(m=0;m<5;m++)
				{
					Vforward[m] = crd.x[i][j+incF[m]][k];
					Vbackward[m] = crd.x[i][j+incB[m]][k];
				}
				x_eta[idx] = ( mathf.dotproduct(Vforward,coffF,5) + mathf.dotproduct(Vbackward,coffB,5) )/2.0/steph;
				for(m=0;m<5;m++)
				{
					Vforward[m] = crd.y[i][j+incF[m]][k];
					Vbackward[m] = crd.y[i][j+incB[m]][k];
				}
				y_eta[idx] = ( mathf.dotproduct(Vforward,coffF,5) + mathf.dotproduct(Vbackward,coffB,5) )/2.0/steph;
				for(m=0;m<5;m++)
				{
					Vforward[m] = crd.z[i][j+incF[m]][k];
					Vbackward[m] = crd.z[i][j+incB[m]][k];
				}
				z_eta[idx] = ( mathf.dotproduct(Vforward,coffF,5) + mathf.dotproduct(Vbackward,coffB,5) )/2.0/steph;
				for(m=0;m<5;m++)
				{
					Vforward[m] = crd.x[i][j][k+incF[m]];
					Vbackward[m] = crd.x[i][j][k+incB[m]];
				}
				x_zeta[idx] = ( mathf.dotproduct(Vforward,coffF,5) + mathf.dotproduct(Vbackward,coffB,5) )/2.0/steph;
				for(m=0;m<5;m++)
				{
					Vforward[m] = crd.y[i][j][k+incF[m]];
					Vbackward[m] = crd.y[i][j][k+incB[m]];
				}
				y_zeta[idx] = ( mathf.dotproduct(Vforward,coffF,5) + mathf.dotproduct(Vbackward,coffB,5) )/2.0/steph;
				for(m=0;m<5;m++)
				{
					Vforward[m] = crd.z[i][j][k+incF[m]];
					Vbackward[m] = crd.z[i][j][k+incB[m]];
				}
				z_zeta[idx] = ( mathf.dotproduct(Vforward,coffF,5) + mathf.dotproduct(Vbackward,coffB,5) )/2.0/steph;

				//calculate the Jacobian matrix, equation 2.5 in Thesis
				//accoridng to the determinant calculation rule.
				vec1[0] = x_xi[idx];   vec1[1] = y_xi[idx];   vec1[2] = z_xi[idx];
				vec2[0] = x_eta[idx];  vec2[1] = y_eta[idx];  vec2[2] = z_eta[idx];
				vec3[0] = x_zeta[idx]; vec3[1] = y_zeta[idx]; vec3[2] = z_zeta[idx];
				mathf.crossproduct(vec1,vec2,vecg);
				drv.jac[i][j][k] = mathf.dotproduct(vecg,vec3,3);

				//calculate the partial derivative of Xi to X, equation 2.4 in Thesis
				//should do cross product first, then mutiply the invert of J, equation 2.2
				//mathf.crossproduct(vec1,vec2,vecg); //same with last step;
				drv.zeta_x[i][j][k] = vecg[0]/drv.jac[i][j][k];
				drv.zeta_y[i][j][k] = vecg[1]/drv.jac[i][j][k];
				drv.zeta_z[i][j][k] = vecg[2]/drv.jac[i][j][k];

				mathf.crossproduct(vec3,vec1,vecg); //same with last step;
				drv.eta_x[i][j][k] = vecg[0]/drv.jac[i][j][k];
				drv.eta_y[i][j][k] = vecg[1]/drv.jac[i][j][k];
				drv.eta_z[i][j][k] = vecg[2]/drv.jac[i][j][k];
				
				mathf.crossproduct(vec2,vec3,vecg);
				drv.xi_x[i][j][k] = vecg[0]/drv.jac[i][j][k];
				drv.xi_y[i][j][k] = vecg[1]/drv.jac[i][j][k];
				drv.xi_z[i][j][k] = vecg[2]/drv.jac[i][j][k];

			}
	com.mirror_extend(drv.xi_x,  cdx);
	com.mirror_extend(drv.xi_y,  cdx);
	com.mirror_extend(drv.xi_z,  cdx);
	com.mirror_extend(drv.eta_x, cdx);
	com.mirror_extend(drv.eta_y, cdx);
	com.mirror_extend(drv.eta_z, cdx);
	com.mirror_extend(drv.zeta_x,cdx);
	com.mirror_extend(drv.zeta_y,cdx);
	com.mirror_extend(drv.zeta_z,cdx);
	com.mirror_extend(drv.jac,   cdx);
	
	delete [] z_zeta; delete [] y_zeta; delete [] x_zeta;
	delete [] z_eta; delete [] y_eta; delete [] x_eta;
	delete [] z_xi; delete [] y_xi; delete [] x_xi;
	
	fprintf(stdout,"---accomplished calculating the partial derivative in gridmesh porgram\n");
}

void gridmesh::export_data(const char *path, cindx cdx)
{
	char crdfile[SeisStrLen],drvfile[SeisStrLen];
	sprintf(crdfile, "%s/coord.nc", path);//path=seisInpath
	sprintf(drvfile, "%s/metric.nc", path);
	
	if(this->Rwork)
	{
		fprintf(stdout,"---There's no needs to store data, due to the restart work\n");
		snc.grid_import(crd, drv, cdx, crdfile, drvfile);
	}
	else
		snc.grid_export(crd, drv, cdx, crdfile, drvfile);
}

























