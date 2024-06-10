#include "typenew.h"
#include<string.h>
#include<math.h>

using namespace std;
using namespace defstruct;
using namespace constant;

#define errprt(...) com.errorprint(__FILE__, __LINE__, __VA_ARGS__)

//-------------------private--------------------------------
void seisplot::getConf(const char *filename)
{
	char parpath[SeisStrLen];
	char name[SeisStrLen2];
	char errstr[SeisStrLen];
	FILE *fp;

	fp=fopen(filename,"r");
	if(!fp)
	{
		sprintf(errstr,"Fail to open main par file %s in seisplot.cpp",filename);
		errprt(Fail2Open,errstr);
	}
	com.get_conf(fp, "seispath", 3, parpath);
	com.get_conf(fp, "station_filename", 3, name);
	com.get_conf(fp, "full_wave_field_storage_interval", 3, &FWSI);
	com.get_conf(fp, "output_peakvel", 3, &PVflag);
	fclose(fp);

	sprintf(sptfile,"%s/%s",parpath,name);

	fp=fopen(sptfile,"r");
	if(!fp)
	{
		sprintf(errstr,"Fail to open seisplot configure file %s in seisplot.cpp",sptfile);
		errprt(Fail2Open,errstr);
	}
	com.get_conf(fp,"number_of_snap",3,&nsnap);
	com.get_conf(fp,"number_of_inline",3,&nline);
	com.get_conf(fp,"number_of_recv",3,&nrecv);

	int i,n;
	char strline[SeisStrLen2];
	npnt=nrecv;
	n=0;
	for(i=0;i<nline;i++)
	{
		sprintf(strline,"line_%3.3d",i+1);
		com.get_conf(fp,strline,11,&n);
		npnt += n;
	}
	fclose(fp);

}


//-----------------------------public----------------------------------
seisplot::seisplot(const char *filename, cindx cdx, const int restart, const int Myid, int cpn, int i_nt)
{
	myid = Myid;
	getConf(filename);
	
	if(myid)
	{
		Mflag = false;//child procs
		printf("child procs %d doesn't paticipate into computing parameters (plot/station)\n",myid);
		return;
	}
	else
		Mflag = true;//master procs
	
	/*
	if(restart==1)
		Rwork = true;
	else
		Rwork = false;
	*/
	Rwork = false; //for real use


	CPN = cpn;
	nt = i_nt;
	int i;
	//malloc point memory
	pnt.SN = new int[npnt]();
	pnt.Pnum = new int[npnt]();
	pnt.Lnum = new int[npnt]();
	pnt.locx = new int[npnt]();
	pnt.locy = new int[npnt]();
	pnt.locz = new int[npnt]();
	pnt.posx = new Real[npnt]();
	pnt.posy = new Real[npnt]();
	pnt.posz = new Real[npnt]();
	memset(pnt.file,'\0',sizeof(char)*SeisStrLen);
	memset(pnt.vid,'0',sizeof(int)*SeisNcNum);
	pnt.ncid = 0;//initialization
	pnt.tinv = 0;
	for(i=0;i<npnt;i++)
		pnt.SN[i] = i;

	//malloc snap memory
	Snp = new int[cpn*nsnap]();
	snp.xs = new int[nsnap]();
	snp.ys = new int[nsnap]();
	snp.zs = new int[nsnap]();
	snp.xn = new int[nsnap]();
	snp.yn = new int[nsnap]();
	snp.zn = new int[nsnap]();
	snp.xi = new int[nsnap]();
	snp.yi = new int[nsnap]();
	snp.zi = new int[nsnap]();
	snp.tinv = new int[nsnap]();
	snp.cmp = new int[nsnap]();
	snp.file = new char*[nsnap];
	for(i=0;i<nsnap;i++)
	{
		snp.file[i] = new char[SeisStrLen];
		memset(snp.file[i],'\0',sizeof(char)*SeisStrLen);
	}
	snp.ncid = new int[nsnap]();
	snp.vid = new int*[nsnap];
	for(i=0;i<nsnap;i++)
	{
		snp.vid[i] = new int[SeisNcNum]();
		memset(snp.vid[i],'0',sizeof(int)*SeisNcNum);
	}


	//malloc wavebuffer memory
	memset(wbuffer.file,'\0',sizeof(char)*SeisStrLen);
	memset(wbuffer.vid,'0',sizeof(int)*SeisNcNum);
	wbuffer.ncid = 0;

	//malloc point wavefield memory
	wpoint.Vx = new Real[npnt]();
	wpoint.Vy = new Real[npnt]();
	wpoint.Vz = new Real[npnt]();
	wpoint.Txx = new Real[npnt]();
	wpoint.Tyy = new Real[npnt]();
	wpoint.Tzz = new Real[npnt]();
	wpoint.Txy = new Real[npnt]();
	wpoint.Txz = new Real[npnt]();
	wpoint.Tyz = new Real[npnt]();

	//malloc point wavefield memory
	MPW.Vx = new Real[nt*npnt]();
	MPW.Vy = new Real[nt*npnt]();
	MPW.Vz = new Real[nt*npnt]();
	MPW.Txx = new Real[nt*npnt]();
	MPW.Tyy = new Real[nt*npnt]();
	MPW.Tzz = new Real[nt*npnt]();
	MPW.Txy = new Real[nt*npnt]();
	MPW.Txz = new Real[nt*npnt]();
	MPW.Tyz = new Real[nt*npnt]();
	
	//malloc snap wavefield memory
	int size;
	size = cdx.nx*cdx.ny*cdx.nz;//store full frame, but record valid points
	wsnap.Vx = new Real[size]();
	wsnap.Vy = new Real[size]();
	wsnap.Vz = new Real[size]();
	wsnap.Txx = new Real[size]();
	wsnap.Tyy = new Real[size]();
	wsnap.Tzz = new Real[size]();
	wsnap.Txy = new Real[size]();
	wsnap.Txz = new Real[size]();
	wsnap.Tyz = new Real[size]();

	//malloc point index buffer
	Hpt.Rsn = new int*[cpn];
	Hpt.Gsn = new int*[cpn];
	Hpt.locx = new int*[cpn];
	Hpt.locy = new int*[cpn];
	Hpt.locz = new int*[cpn];

	//malloc snap buffer
	MSW = new wfield[nsnap]();
	//malloc snap index buffer
	HSpt = new SnapIndexBuffer[nsnap]();
	for(i=0;i<nsnap;i++)
	{
		HSpt[i].Rsn = new int*[cpn];
		HSpt[i].Gsn = new int*[cpn];
		HSpt[i].locx = new int*[cpn];
		HSpt[i].locy = new int*[cpn];
		HSpt[i].locz = new int*[cpn];
	}

	//malloc peak velocity buffer
	if(PVflag)
	{
		pv.Vx = new Real[cdx.nx*cdx.ny]();
		pv.Vy = new Real[cdx.nx*cdx.ny]();
		pv.Vz = new Real[cdx.nx*cdx.ny]();
	}

}

seisplot::~seisplot()
{
	fprintf(stdout,"into data free at Procs[%d],in seisplot.cpp\n",myid);
	if(this->Mflag)
	{
		int i,j;

		//for peak velocity
		if(PVflag)
		{
			delete [] pv.Vx;
			delete [] pv.Vy;
			delete [] pv.Vz;
		}
		
		//for snapshot
		for(i=0;i<nsnap;i++)
		{
#ifndef PointOnly			
			//free MSW
			if(HSpt[i].cmp==2 || HSpt[i].cmp==3)
			{
				delete [] MSW[i].Tyz;	delete [] MSW[i].Txz;	delete [] MSW[i].Txy;
				delete [] MSW[i].Tzz;	delete [] MSW[i].Tyy;	delete [] MSW[i].Txx;
			}
			if(HSpt[i].cmp==1 || HSpt[i].cmp==3)
			{
				delete [] MSW[i].Vz;	delete [] MSW[i].Vy;	delete [] MSW[i].Vx;
			}
#endif			
			
			//free HSpt
			for(j=0;j<CPN;j++)
			{
				delete [] HSpt[i].locz[j];
				delete [] HSpt[i].locy[j];
				delete [] HSpt[i].locx[j];
				delete [] HSpt[i].Gsn[j];
				delete [] HSpt[i].Rsn[j];
			}
			delete [] HSpt[i].locz;
			delete [] HSpt[i].locy;
			delete [] HSpt[i].locx;
			delete [] HSpt[i].Gsn;
			delete [] HSpt[i].Rsn;
		}
		delete [] HSpt;
		delete [] MSW;
		
		//for point
		for(i=0;i<CPN;i++)
		{
			delete [] Hpt.locz[i];
			delete [] Hpt.locy[i];
			delete [] Hpt.locx[i];
			delete [] Hpt.Gsn[i];
			delete [] Hpt.Rsn[i];
		}
		delete [] Hpt.locz;
		delete [] Hpt.locy;
		delete [] Hpt.locx;
		delete [] Hpt.Gsn;
		delete [] Hpt.Rsn;

		delete [] wsnap.Tyz;
		delete [] wsnap.Txz;
		delete [] wsnap.Txy;
		delete [] wsnap.Tzz;
		delete [] wsnap.Tyy;
		delete [] wsnap.Txx;
		delete [] wsnap.Vz;
		delete [] wsnap.Vy;
		delete [] wsnap.Vx;

		delete [] MPW.Tyz;
		delete [] MPW.Txz;
		delete [] MPW.Txy;
		delete [] MPW.Tzz;
		delete [] MPW.Tyy;
		delete [] MPW.Txx;
		delete [] MPW.Vz;
		delete [] MPW.Vy;
		delete [] MPW.Vx;
		
		delete [] wpoint.Tyz;
		delete [] wpoint.Txz;
		delete [] wpoint.Txy;
		delete [] wpoint.Tzz;
		delete [] wpoint.Tyy;
		delete [] wpoint.Txx;
		delete [] wpoint.Vz;
		delete [] wpoint.Vy;
		delete [] wpoint.Vx;
	
		for(i=0;i<nsnap;i++)
		{
			delete [] snp.SN[i];
			delete [] snp.vid[i];
			delete [] snp.file[i];
		}
		delete [] snp.SN;
		delete [] snp.vid;
		delete [] snp.file;
		delete [] snp.ncid;
		delete [] snp.cmp;
		delete [] snp.tinv;
		delete [] snp.zi;
		delete [] snp.yi;
		delete [] snp.xi;
		delete [] snp.zn;
		delete [] snp.yn;
		delete [] snp.xn;
		delete [] snp.zs;
		delete [] snp.ys;
		delete [] snp.xs;
		delete [] Snp;

		delete [] pnt.posz;
		delete [] pnt.posy;
		delete [] pnt.posx;
		delete [] pnt.locz;
		delete [] pnt.locy;
		delete [] pnt.locx;
		delete [] pnt.Lnum;
		delete [] pnt.Pnum;
		delete [] pnt.SN;
	}
	fprintf(stdout,"data free at Procs[%d],in seisplot.cpp\n",myid);
}

void seisplot::readdata(const char *path)
{
	char errstr[SeisStrLen],strline[SeisStrLen2];
	int i,j;
	int ipnt;//global index of point
		
	FILE *fp;
	fp=fopen(sptfile,"r");
	if(!fp)
	{
		sprintf(errstr,"Fail to open seisplot configure file %s in seisplot.cpp",sptfile);
		errprt(Fail2Open,errstr);
	}
	com.get_conf(fp,"topo_hyper_height",3,&hyp2g);
	com.get_conf(fp,"tinv_of_seismo",3,&pnt.tinv);

	//read snap pars
	snp.SN = new int*[nsnap];
	for(i=0;i<nsnap;i++)
	{
		sprintf(strline,"snap_%3.3d",i+1);
		com.get_conf(fp,strline,3,&snp.xs[i]);
		com.get_conf(fp,strline,4,&snp.ys[i]);
		com.get_conf(fp,strline,5,&snp.zs[i]);
		com.get_conf(fp,strline,6,&snp.xn[i]);
		com.get_conf(fp,strline,7,&snp.yn[i]);
		com.get_conf(fp,strline,8,&snp.zn[i]);
		com.get_conf(fp,strline,9,&snp.xi[i]);
		com.get_conf(fp,strline,10,&snp.yi[i]);
		com.get_conf(fp,strline,11,&snp.zi[i]);
		com.get_conf(fp,strline,12,&snp.tinv[i]);
		com.get_conf(fp,strline,13,&snp.cmp[i]);
		sprintf(snp.file[i],"%s/%s.nc",path,strline);
		
		snp.SN[i] = new int [ snp.xn[i]*snp.yn[i]*snp.zn[i] ]();
		for(j=0;j<snp.xn[i]*snp.yn[i]*snp.zn[i];j++)
			snp.SN[i][j] = j;
	}
	
	//read receiver
	ipnt=0;
	for(i=0;i<nrecv;i++)
	{
		sprintf(strline,"recv_%3.3d",i+1);
		com.get_conf(fp,strline,3,&pnt.posx[i]);
		com.get_conf(fp,strline,4,&pnt.posy[i]);
		com.get_conf(fp,strline,5,&pnt.posz[i]);
		pnt.Pnum[i] = i;
		pnt.Lnum[i] = 0;
	}
	ipnt += nrecv;
	
	//read line
	Real x0,y0,z0,dx,dy,dz,num;
	for(i=0;i<nline;i++)
	{
		sprintf(strline,"line_%3.3d",i+1);
		com.get_conf(fp,strline,3,&x0);
		com.get_conf(fp,strline,4,&y0);
		com.get_conf(fp,strline,5,&z0);
		com.get_conf(fp,strline,7,&dx);
		com.get_conf(fp,strline,8,&dy);
		com.get_conf(fp,strline,9,&dz);
		com.get_conf(fp,strline,11,&num);
		for(j=0;j<num;j++)
		{
			pnt.posx[ipnt+j] = x0+j*dx;
			pnt.posy[ipnt+j] = y0+j*dy;
			pnt.posz[ipnt+j] = z0+j*dz;
			pnt.Pnum[ipnt+j] = j;
			pnt.Lnum[ipnt+j] = i+1;
		}
		ipnt += num;
	}
	sprintf(pnt.file,"%s/seismo.nc",path);

	//set wave buffer file path
	sprintf(wbuffer.file,"%s/wavebuffer.nc",path);
	
	fclose(fp);
}

void seisplot::locpoint(cindx cdx, coord crd)
{
	char errstr[SeisStrLen];
	Real Px,Py,Pz;
	int Lx,Ly,Lz;
	int i;
	Real tempPz;

	for(i=0;i<npnt;i++)
	{
		Px = pnt.posx[i]; Py = pnt.posy[i]; Pz = pnt.posz[i];
		com.reposition(Px,Py,Pz,cdx,crd,hyp2g,&Lx,&Ly,&Lz,&tempPz);
		pnt.posz[i] = tempPz;
		if( Lx >= cdx.ni1 && Lx < cdx.ni2 && Ly >= cdx.nj1 && Ly < cdx.nj2 && Lz >= cdx.nk1 && Lz < cdx.nk2 )
		{
			pnt.locx[i] = Lx; pnt.locy[i] = Ly; pnt.locz[i] = Lz;
		}
		else
		{
			sprintf(errstr,"No.%d reciver is out of Computing Boundary (No Virtual)!",i);
			errprt(Fail2Check,errstr);
		}
		//printf("for Receiver %d (%g,%g,%g) locates at(%d,%d,%d) and corresponding position is (%gm, %gm, %gm)\n",
		//	i+1,	pnt.posx[i],pnt.posy[i],pnt.posz[i],	pnt.locx[i],pnt.locy[i],pnt.locz[i],
		//	crd.x[pnt.locx[i]][pnt.locy[i]][pnt.locz[i]],crd.y[pnt.locx[i]][pnt.locy[i]][pnt.locz[i]],crd.z[pnt.locx[i]][pnt.locy[i]][pnt.locz[i]]);
	}
	/*
	//FILE *fp;
	//fp = fopen("./Rrecv_cu.txt","w");
	Real offset,Xo,Yo,Zo;
	for(i=0;i<npnt;i++)
	{
		Xo = crd.x[pnt.locx[i]][pnt.locy[i]][pnt.locz[i]]-crd.x[pnt.locx[8]][pnt.locy[8]][pnt.locz[8]];   	
		Yo = crd.y[pnt.locx[i]][pnt.locy[i]][pnt.locz[i]]-crd.y[pnt.locx[8]][pnt.locy[8]][pnt.locz[8]];   	
		Zo = crd.z[pnt.locx[i]][pnt.locy[i]][pnt.locz[i]]-crd.z[pnt.locx[8]][pnt.locy[8]][pnt.locz[8]];
		offset = sqrt( Xo*Xo + Yo*Yo + Zo*Zo);
		printf("for Receiver %d (%g,%g,%g) locates at(%d,%d,%d) and corresponding position is (%gm, %gm, %gm) related offset is %gm (%gm, %gm ,%gm)\n",
			i+1,	pnt.posx[i],pnt.posy[i],pnt.posz[i],	pnt.locx[i],pnt.locy[i],pnt.locz[i],
			crd.x[pnt.locx[i]][pnt.locy[i]][pnt.locz[i]],crd.y[pnt.locx[i]][pnt.locy[i]][pnt.locz[i]],crd.z[pnt.locx[i]][pnt.locy[i]][pnt.locz[i]],
			offset,Xo,Yo,Zo);
		//fprintf(fp,"%2d  %6g  %3d  %9.3f  %9.3f\n",i+1,pnt.posz[i],pnt.locz[i],
		//	crd.z[pnt.locx[i]][pnt.locy[i]][pnt.locz[i]],offset); 
	}
	//fclose(fp);
	*/
}

void seisplot::M2CSnapPick(cindx cdx, int *Xs, int *Xe, int cpn)
{
	int i,j,k,m,n;
	int numP;
	int location;

	for(i=0;i<nsnap;i++)
	{
		for(j=0;j<cpn;j++)
		{
			numP = 0;
			for(k=0;k<snp.xn[i];k++)
			{
				location = snp.xs[i]-1 + LenFD+k*snp.xi[i];
				//printf("at xs=%d, xi=%d, No.%d, location=%d, ranges(%d,%d)\n",snp.xs[i],snp.xi[i],k,location,Xs[j],Xe[j]);
				if(location >= Xs[j] && location <= Xe[j])
					numP++;
			}
			
			Snp[j*nsnap+i] = numP*snp.yn[i]*snp.zn[i];
			//printf("at PCS[%d]snap[%d] ranges(%d,%d) numP=%d, have %d point\n",j+1,i+1,Xs[j],Xe[j], numP,Snp[j*nsnap+i]);
			HSpt[i].Rsn[j] = new int[ Snp[j*nsnap+i] ]();
			HSpt[i].Gsn[j] = new int[ Snp[j*nsnap+i] ]();
			HSpt[i].locx[j] = new int[ Snp[j*nsnap+i] ]();
			HSpt[i].locy[j] = new int[ Snp[j*nsnap+i] ]();
			HSpt[i].locz[j] = new int[ Snp[j*nsnap+i] ]();
		}
	}

	for(j=0;j<cpn;j++)
	{
		for(i=0;i<nsnap;i++)
		{
			numP = 0;
			for(k=0;k<snp.xn[i];k++)
			{
				location = snp.xs[i]-1 + LenFD+k*snp.xi[i];
				if(location >= Xs[j] && location <= Xe[j])
				{
					for(m=0;m<snp.yn[i];m++)
						for(n=0;n<snp.zn[i];n++)
						{
							HSpt[i].Rsn[j][numP] = numP;
							HSpt[i].Gsn[j][numP] = k*snp.yn[i]*snp.zn[i] + m*snp.zn[i] + n;
							HSpt[i].locx[j][numP] = location;
							HSpt[i].locy[j][numP] = snp.ys[i]-1 + LenFD + m*snp.yi[i];
							HSpt[i].locz[j][numP] = snp.zs[i]-1 + LenFD + n*snp.zi[i]; 
							numP++;
						}
				}
			}
			HSpt[i].tinv = snp.tinv[i];
			HSpt[i].cmp = snp.cmp[i];
		}
	}

#ifndef PointOnly
	int size,nTime;
	for(i=0;i<nsnap;i++)
	{
		nTime = ceil(1.0*this->nt/HSpt[i].tinv);
		size = snp.xn[i]*snp.yn[i]*snp.zn[i];
		//printf("for snap[%d], temporal points is %d spatial points is %d\n",i+1,nTime,size);

		if(HSpt[i].cmp==1 || HSpt[i].cmp==3)
		{
			MSW[i].Vx = new Real[size*nTime]();	MSW[i].Vy = new Real[size*nTime]();	MSW[i].Vz = new Real[size*nTime]();
		}
		if(HSpt[i].cmp==2 || HSpt[i].cmp==3)
		{
			MSW[i].Txx = new Real[size*nTime]();	MSW[i].Tyy = new Real[size*nTime]();	MSW[i].Tzz = new Real[size*nTime]();
			MSW[i].Txy = new Real[size*nTime]();	MSW[i].Txz = new Real[size*nTime]();	MSW[i].Tyz = new Real[size*nTime]();
		}
	}
#endif

	/*
	for(i=0;i<nsnap;i++)
	{
		for(j=0;j<cpn;j++)
		{
			location = j*nsnap+i;
			for(k=0;k<Snp[location];k++)
				printf("inMas-Snapshot[%d],PCS[%d]->Rsn[%3d],Gsn[%3d]->(%3d,%3d,%3d),tinv=%d,cmp=%d\n",i+1,j+1,
				HSpt[i].Rsn[j][k],HSpt[i].Gsn[j][k],HSpt[i].locx[j][k],HSpt[i].locy[j][k],HSpt[i].locz[j][k],HSpt[i].tinv,HSpt[i].cmp);
		}
	}
	*/
	
}

void seisplot::M2CPointPick(cindx cdx, int *Xs, int *Xe, int *np, int cpn)
{
	int i,j;
	int numP;
	
	for(j=0;j<cpn;j++)
	{
		numP = 0;
		for(i=0;i<this->npnt;i++)
			if(pnt.locx[i]>=Xs[j] && pnt.locx[i]<=Xe[j])
				numP++;
		np[j] = numP;

		Hpt.Rsn[j] = new int[numP]();
		Hpt.Gsn[j] = new int[numP]();
		Hpt.locx[j] = new int[numP]();
		Hpt.locy[j] = new int[numP]();
		Hpt.locz[j] = new int[numP]();
	}

	for(j=0;j<cpn;j++)
	{
		numP = 0;
		for(i=0;i<this->npnt;i++)
			if(pnt.locx[i]>=Xs[j] && pnt.locx[i]<=Xe[j])
			{
				Hpt.Rsn[j][numP] = numP;
				Hpt.Gsn[j][numP] = pnt.SN[i];
				Hpt.locx[j][numP] = pnt.locx[i];
				Hpt.locy[j][numP] = pnt.locy[i];
				Hpt.locz[j][numP] = pnt.locz[i];
				numP++;
			}
	}
/*
	for(j=0;j<cpn;j++)
	{
		for(i=0;i<np[j];i++)
			printf("Rsn[%d],Gsn[%d]->(%d,%d,%d)\n",Hpt.Rsn[j][i],Hpt.Gsn[j][i],Hpt.locx[j][i],Hpt.locy[j][i],Hpt.locz[j][i]);
	}
*/	
	
}

void seisplot::data_export_def(int *currT, Real stept, cindx cdx, point *pnt, snap snp, wavebuffer *wbuffer, wfield wsnap)
{
	//   type               parameters-Var                   value-Var
	//   point-wave-field   pnt[defstruct::point]            wpoint[defstruct::wfield]
	//   snap-wave-field    snp[defstruct::snap]             wsnap[defstruct::wfield]
	//   wave-buffer        wbuffer[defstruct::wavebuffer]   wsnap[defstruct::wfield]       Abandoned
	//---------------------------------------------------------------------------------
	//   value array for snap and wave is same, 
	//   the snap do sorting work only when storing,
	//   so the full size of wave field is wsnap;
	//
	//   pnt and wbuffer were received in pointer type, here just do gathering and sending work.
	if(this->Rwork)
	{
		fprintf(stdout,"---There needs to read previous stored wave field and parameters, then restart\n");
		snc.point_import(pnt);//here pnt is already a pointer
		snc.snap_import(this->nsnap, snp);
		//snc.wavebuffer_import(currT, stept, wbuffer, cdx, wsnap);//here wbuffer is already a pointer
	}
	else
	{
		snc.point_export_def(this->npnt, pnt);//pnt is a pointer
		snc.snap_export_def(this->nsnap, snp);
		//snc.wavebuffer_def(wbuffer, cdx);//pnt is a pointer
	}

}

void seisplot::data_export(int currT, int prepauseT, Real stept, cindx cdx, point pnt, snap snp, wavebuffer wbuffer, wfield wpoint, wfield wsnap)
{
	if(currT % pnt.tinv == 0 && currT >= prepauseT)
	{
		point_extract(pnt, cdx, wpoint, wsnap);//extract point from fullwave field
		snc.point_export(currT, stept, this->npnt, pnt, wpoint);
	}

	if(currT >= prepauseT)
		snc.snap_export(currT, stept, this->nsnap, snp, cdx, wsnap);
	
	//if( currT && currT%FWSI==0)
	//	snc.wavebuffer_export(currT, stept, wbuffer, cdx, wsnap);
}

void seisplot::data_export_end(point pnt, snap snp, wavebuffer wbuffer)
{
	snc.point_export_end(pnt);
	snc.snap_export_end(this->nsnap, snp);
	//snc.wavebuffer_end(wbuffer);
}

void seisplot::point_extract(point pnt, cindx cdx, wfield wpoint, wfield wsnap)
{
	int index;
	int i;
	for(i=0;i<this->npnt;i++)
	{
		index = pnt.locx[i]*cdx.ny*cdx.nz + pnt.locy[i]*cdx.nz + pnt.locz[i];
		wpoint.Vx[i] = wsnap.Vx[index];
		wpoint.Vy[i] = wsnap.Vy[index];
		wpoint.Vz[i] = wsnap.Vz[index];
		wpoint.Txx[i] = wsnap.Txx[index];
		wpoint.Tyy[i] = wsnap.Tyy[index];
		wpoint.Tzz[i] = wsnap.Tzz[index];
		wpoint.Txy[i] = wsnap.Txy[index];
		wpoint.Txz[i] = wsnap.Txz[index];
		wpoint.Tyz[i] = wsnap.Tyz[index];
	}

}


void seisplot::point_export(int TotalTime, int prepauseT, Real stept)
{
	int current;
	int i;
	//for(current=0; current<1; current++)
	for(current=0; current<TotalTime; current++)
	{
		point_extract(current, wpoint, MPW);//extract point from point wave buffer
		//for(int j=0;j<this->npnt;j++)
		//	printf("point[%d],locates(%d,%d,%d), at time %d, Tyy-value=%f\n",
		//		j, pnt.locx[j],pnt.locy[j],pnt.locz[j], current, wpoint.Tyy[j]);
		//printf("---------------------------------------------------------------\n");
		snc.point_export(current, stept, this->npnt, pnt, wpoint);
	}
}

void seisplot::point_extract(int current, wfield wpoint, wfield MPW)
{
	int src;
	int i;
	for(i=0;i<this->npnt;i++)
	{
		src = i*this->nt+current;
		wpoint.Vx[i] =  MPW.Vx [src];
		wpoint.Vy[i] =  MPW.Vy [src];
		wpoint.Vz[i] =  MPW.Vz [src];
		wpoint.Txx[i] = MPW.Txx[src];
		wpoint.Tyy[i] = MPW.Tyy[src];
		wpoint.Tzz[i] = MPW.Tzz[src];
		wpoint.Txy[i] = MPW.Txy[src];
		wpoint.Txz[i] = MPW.Txz[src];
		wpoint.Tyz[i] = MPW.Tyz[src];
	}
}

void seisplot::snap_export(int TotalTime, int prepauseT, Real stept)
{
	//output
	//snc.snap_export(time, this->nsnap, snp, MSW);
#ifndef PointOnly
	int current;
	int i,j,k,m;
	int Tlen,nTime;
	Real time;

	Real **var;
	var = new Real*[9];
	
	for(i=0;i<nsnap;i++)
	{
		Tlen = ceil(1.0*TotalTime/HSpt[i].tinv);
		nTime = ceil(1.0*this->nt/HSpt[i].tinv);

		for(j=0;j<9;j++)
			var[j] = new Real[ snp.xn[i]*snp.yn[i]*snp.zn[i] ]();
		
		/*
		for(current=0; current<Tlen; current++)
			printf("for snap[%d], output NO.%d temporal slice(means %d) --->its' tinv is %d, total length is %d, valid length is %d, stop time is %d\n",
					i+1, current, current*HSpt[i].tinv, HSpt[i].tinv, nTime, Tlen, TotalTime);
		printf("\n");
		*/
		
		//output
		for(current=0; current<Tlen; current++)
		{
			//extract
			for(k=0;k<snp.xn[i]*snp.yn[i]*snp.zn[i];k++)
			{
				if(HSpt[i].cmp==1 || HSpt[i].cmp==3)
				{
					var[0][k] = MSW[i].Vx[ k*nTime+current ];
					var[1][k] = MSW[i].Vy[ k*nTime+current ];
					var[2][k] = MSW[i].Vz[ k*nTime+current ];
				}
				if(HSpt[i].cmp==2 || HSpt[i].cmp==3)
				{
					var[3][k] = MSW[i].Txx[ k*nTime+current ];
					var[4][k] = MSW[i].Tyy[ k*nTime+current ];
					var[5][k] = MSW[i].Tzz[ k*nTime+current ];
					var[6][k] = MSW[i].Txy[ k*nTime+current ];
					var[7][k] = MSW[i].Txz[ k*nTime+current ];
					var[8][k] = MSW[i].Tyz[ k*nTime+current ];
				}
			}

			snc.snap_export(i, current, stept, this->nsnap, snp, var);

			//print
			/*
			for(k=0;k<snp.xn[i]*snp.yn[i]*snp.zn[i];k++)
			{
				if(HSpt[i].cmp==1 || HSpt[i].cmp==3)
				{
					printf("transfer-snap[%d]->k=%4d,time=%d, Vx=%f\n",i+1,k,current,var[0][k]);
				}

				if(HSpt[i].cmp==2 || HSpt[i].cmp==3)
				{
					printf("transfer-snap[%d]->k=%4d,time=%d, Txx=%f\n",i+1,k,current,var[3][k]);
				}
			}
			*/
		}
#endif		

		/*
		//display
		for(j=0;j<CPN;j++)
			for(k=0;k<Snp[j*nsnap+i];k++)
			{
				for(current=0; current<Tlen; current++)
				{
					
					if(HSpt[i].cmp==1 || HSpt[i].cmp==3)
					{
						printf("back-snap[%d]->Rsn=%4d,Gsn=%4d,locates(%3d,%3d,%3d),Vx=%f\n",
								i+1,HSpt[i].Rsn[j][k],HSpt[i].Gsn[j][k],HSpt[i].locx[j][k],HSpt[i].locy[j][k],HSpt[i].locz[j][k],
								MSW[i].Vx[ HSpt[i].Gsn[j][k]*nTime+current ]);

					}

					if(HSpt[i].cmp==2 || HSpt[i].cmp==3)
					{
						printf("back-snap[%d]->Rsn=%4d,Gsn=%4d,locates(%3d,%3d,%3d),Txx=%f\n",
								i+1,HSpt[i].Rsn[j][k],HSpt[i].Gsn[j][k],HSpt[i].locx[j][k],HSpt[i].locy[j][k],HSpt[i].locz[j][k],
								MSW[i].Txx[ HSpt[i].Gsn[j][k]*nTime+current ]);
					}
					
				}
			}
		*/

		for(j=0;j<9;j++)
			delete [] var[j];
	
	}
	delete [] var;
	
}


void seisplot::export_pv(const char *path, cindx cdx)
{
	if(PVflag==0)
		return;

	char pvfile[SeisStrLen];
	sprintf(pvfile, "%s/peakvel.nc",path);

	snc.PV_export(pv, pvfile, cdx);

}



























