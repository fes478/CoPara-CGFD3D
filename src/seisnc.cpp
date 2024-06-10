#include "typenew.h"
#include<string.h>

using namespace constant;
using namespace defstruct;
using namespace std;

#define errprt(...) errorprint(__FILE__, __LINE__, __VA_ARGS__)

// NOTICE: retval = the Return Value
//         subs = subStart       sube = subEnd
//         subn = subNumber      subi = subInterval

//================== private ===================================================
long int seisnc::getVarSize(const char *filenm, int ncid, const char *varnm, int varid)
{
  int ndims, *dimid, retval;
  size_t *dimlen;
  long int size;

  retval = nc_inq_varndims(ncid, varid, &ndims);
    errprt("inquire variable dimension number", varnm, filenm, retval, Fail2Read);
  dimid = new int[ndims] ();
  dimlen = new size_t[ndims] ();
  retval = nc_inq_vardimid(ncid, varid, dimid);
    errprt("inquire variable dimension ID", varnm, filenm, retval, Fail2Read);
  for(int i = 0; i < ndims; i++)
  {
    retval = nc_inq_dimlen(ncid, dimid[i], &dimlen[i]);
      errprt("inquire dimension length", filenm, retval, Fail2Read);
  }

  size = mathf.accumulation(dimlen, ndims);

  delete [] dimid;
  delete [] dimlen;
  return size;
}
long int seisnc::getVarSize(const char *filenm, int ncid, const char *varnm, int varid, size_t *subc)
{
  int ndims, retval;
  long int size;

  retval = nc_inq_varndims(ncid, varid, &ndims);
    errprt("inquire variable dimension number", varnm, filenm,
      retval, Fail2Read);
  size = mathf.accumulation(subc, ndims);
  return size;
}
//================== public ====================================================
seisnc::seisnc()
{
  nf4int = NC_FILL_INT;
  nf4float = NC_FILL_FLOAT;
  nf4double = NC_FILL_DOUBLE;
}

void seisnc::errorprint(const char *filenm, int numline, const char *what, const char *who, int retncode, int errorcode)
{
  if(retncode != NC_NOERR)
  {
    fprintf(stdout,"FATAL netcdf error at %s(%d): %s [%s].\n", filenm, numline, what, who);
    fprintf(stdout,"Supplementary information: %s\n", nc_strerror(retncode));
    exit(errorcode);
  }
}
void seisnc::errorprint(const char *filenm, int numline, const char *what, const char *who1, const char *who2, int retncode, int errorcode)
{
  if(retncode != NC_NOERR)
  {
    fprintf(stdout,"FATAL netcdf error at %s(%d): %s [%s] in/to/from <%s>.\n", filenm, numline, what, who1, who2);
    fprintf(stdout,"Supplementary information: %s\n", nc_strerror(retncode));
    exit(errorcode);
  }
}

size_t seisnc::dimsize(const char *filename, const char *dimname)
{
	int ncid,dimid,retval;
	size_t dimsize;
	retval = nc_open(filename, NC_NOWRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_dimid(ncid, dimname, &dimid);
	errprt("extracting dimension ID", dimname, filename, retval, Fail2Read);
	retval = nc_inq_dimlen(ncid, dimid, &dimsize);
	errprt("extracting dimension Size", dimname, filename, retval, Fail2Read);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
	return dimsize;
}

void seisnc::attrget(const char *filename, const char *attname, int *attr)
{
	int ncid,retval;
	retval = nc_open(filename, NC_NOWRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_get_att_int(ncid, NC_GLOBAL, attname, attr);
	errprt("extracting global attributes", attname, filename, retval, Fail2Read);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}
void seisnc::attrget(const char *filename, const char *attname, float *attr)
{
	int ncid,retval;
	retval = nc_open(filename, NC_NOWRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_get_att_float(ncid, NC_GLOBAL, attname, attr);
	errprt("extracting global attributes", attname, filename, retval, Fail2Read);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}
void seisnc::attrget(const char *filename, const char *attname, double *attr)
{
	int ncid,retval;
	retval = nc_open(filename, NC_NOWRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_get_att_double(ncid, NC_GLOBAL, attname, attr);
	errprt("extracting global attributes", attname, filename, retval, Fail2Read);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}

void seisnc::varget(const char *filename, const char *varname, int *var)
{//getVar
	int ncid,varid,retval;
	retval = nc_open(filename, NC_NOWRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_varid(ncid, varname, &varid);
	errprt("extracting variables ID", varname, filename, retval, Fail2Read);
	retval = nc_get_var_int(ncid, varid, var);
	errprt("extracting variables Value", varname, filename, retval, Fail2Read);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}
void seisnc::varget(const char *filename, const char *varname, float *var)
{
	int ncid,varid,retval;
	retval = nc_open(filename, NC_NOWRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_varid(ncid, varname, &varid);
	errprt("extracting variables ID", varname, filename, retval, Fail2Read);
	retval = nc_get_var_float(ncid, varid, var);
	errprt("extracting variables Value", varname, filename, retval, Fail2Read);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}
void seisnc::varget(const char *filename, const char *varname, double *var)
{
	int ncid,varid,retval;
	retval = nc_open(filename, NC_NOWRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_varid(ncid, varname, &varid);
	errprt("extracting variables ID", varname, filename, retval, Fail2Read);
	retval = nc_get_var_double(ncid, varid, var);
	errprt("extracting variables Value", varname, filename, retval, Fail2Read);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}
void seisnc::varget(const char *filename, const char *varname, int *var, size_t *subs, size_t *subn, ptrdiff_t *subi)
{//getVars
	int ncid,varid,retval;
	retval = nc_open(filename, NC_NOWRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_varid(ncid, varname, &varid);
	errprt("extracting variables ID", varname, filename, retval, Fail2Read);
	retval = nc_get_vars_int(ncid, varid, subs, subn, subi, var);
	errprt("extracting variables Value", varname, filename, retval, Fail2Read);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}
void seisnc::varget(const char *filename, const char *varname, float *var, size_t *subs, size_t *subn, ptrdiff_t *subi)
{//getVars
	int ncid,varid,retval;
	retval = nc_open(filename, NC_NOWRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_varid(ncid, varname, &varid);
	errprt("extracting variables ID", varname, filename, retval, Fail2Read);
	retval = nc_get_vars_float(ncid, varid, subs, subn, subi, var);
	errprt("extracting variables Value", varname, filename, retval, Fail2Read);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}
void seisnc::varget(const char *filename, const char *varname, double *var, size_t *subs, size_t *subn, ptrdiff_t *subi)
{//getVars
	int ncid,varid,retval;
	retval = nc_open(filename, NC_NOWRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_varid(ncid, varname, &varid);
	errprt("extracting variables ID", varname, filename, retval, Fail2Read);
	retval = nc_get_vars_double(ncid, varid, subs, subn, subi, var);
	errprt("extracting variables Value", varname, filename, retval, Fail2Read);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}

void seisnc::varput(const char *filename, const char *varname, int *var)
{//putVar
	int ncid,varid,retval;
	retval = nc_open(filename, NC_WRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_varid(ncid, varname, &varid);
	errprt("extracting variables ID", varname, filename, retval, Fail2Read);
#ifdef GMTCOARDS
	long int size;
	size = getVarSize(filename, ncid, varname, varid);
	int vbound[2];
	vbound[0] = mathf.min(var,size);
	vbound[1] = mathf.max(var,size);
	retval = nc_redef(ncid);
	errprt("redefine NC file", filename, retval, Fail2File);
	retval = nc_put_att_int(ncid, varid, "actual_range", NC_INT, 2, vbound);
	errprt("writing attributes of value boundary for", varname, filename, retval, Fail2Write);
	retval = nc_put_att_int(ncid, varid, _FillValue, NC_INT, 1, &nf4int);
	errprt("writing attributes' name for", varname, filename, retval, Fail2Write);
	retval = nc_enddef(ncid);
	errprt("completing define mode", filename, retval, Fail2File);
#endif
	retval = nc_put_var_int(ncid, varid, var);
	errprt("writing variables Value", varname, filename, retval, Fail2Write);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}
void seisnc::varput(const char *filename, const char *varname, float *var)
{//putVar
	int ncid,varid,retval;
	retval = nc_open(filename, NC_WRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_varid(ncid, varname, &varid);
	errprt("extracting variables ID", varname, filename, retval, Fail2Read);
#ifdef GMTCOARDS
	long int size;
	size = getVarSize(filename, ncid, varname, varid);
	float vbound[2];
	vbound[0] = mathf.min(var,size);
	vbound[1] = mathf.max(var,size);
	retval = nc_redef(ncid);
	errprt("redefine NC file", filename, retval, Fail2File);
	retval = nc_put_att_float(ncid, varid, "actual_range", NC_FLOAT, 2, vbound);
	errprt("writing attributes of value boundary for", varname, filename, retval, Fail2Write);
	retval = nc_put_att_float(ncid, varid, _FillValue, NC_FLOAT, 1, &nf4float);
	errprt("writing attributes' name for", varname, filename, retval, Fail2Write);
	retval = nc_enddef(ncid);
	errprt("completing define mode", filename, retval, Fail2File);
#endif
	retval = nc_put_var_float(ncid, varid, var);
	errprt("writing variables Value", varname, filename, retval, Fail2Write);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}
void seisnc::varput(const char *filename, const char *varname, double *var)
{//putVar
	int ncid,varid,retval;
	retval = nc_open(filename, NC_WRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_varid(ncid, varname, &varid);
	errprt("extracting variables ID", varname, filename, retval, Fail2Read);
#ifdef GMTCOARDS
	long int size;
	size = getVarSize(filename, ncid, varname, varid);
	double vbound[2];
	vbound[0] = mathf.min(var,size);
	vbound[1] = mathf.max(var,size);
	retval = nc_redef(ncid);
	errprt("redefine NC file", filename, retval, Fail2File);
	retval = nc_put_att_double(ncid, varid, "actual_range", NC_DOUBLE, 2, vbound);
	errprt("writing attributes of value boundary for", varname, filename, retval, Fail2Write);
	retval = nc_put_att_double(ncid, varid, _FillValue, NC_DOUBLE, 1, &nf4double);
	errprt("writing attributes' name for", varname, filename, retval, Fail2Write);
	retval = nc_enddef(ncid);
	errprt("completing define mode", filename, retval, Fail2File);
#endif
	retval = nc_put_var_double(ncid, varid, var);
	errprt("writing variables Value", varname, filename, retval, Fail2Write);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}
void seisnc::varput(const char *filename, const char *varname, int *var, size_t *subs, size_t *subn, ptrdiff_t *subi, bool isgmt)
{//putVars
	int ncid,varid,retval;
	retval = nc_open(filename, NC_WRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_varid(ncid, varname, &varid);
	errprt("extracting variables ID", varname, filename, retval, Fail2Read);
#ifdef GMTCOARDS
	if(isgmt)
	{
		long int size;
		size = getVarSize(filename, ncid, varname, varid);
		int vbound[2];
		vbound[0] = mathf.min(var,size);
		vbound[1] = mathf.max(var,size);
		retval = nc_redef(ncid);
		errprt("redefine NC file", filename, retval, Fail2File);
		retval = nc_put_att_int(ncid, varid, "actual_range", NC_INT, 2, vbound);
		errprt("writing attributes of value boundary for", varname, filename, retval, Fail2Write);
		retval = nc_put_att_int(ncid, varid, _FillValue, NC_INT, 1, &nf4int);
		errprt("writing attributes' name for", varname, filename, retval, Fail2Write);
		retval = nc_enddef(ncid);
		errprt("completing define mode", filename, retval, Fail2File);
	}
#endif
	retval = nc_put_vars_int(ncid, varid, subs, subn, subi, var);
	errprt("writing variables Value", varname, filename, retval, Fail2Write);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}
void seisnc::varput(const char *filename, const char *varname, float *var, size_t *subs, size_t *subn, ptrdiff_t *subi, bool isgmt)
{//putVars
	int ncid,varid,retval;
	retval = nc_open(filename, NC_WRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_varid(ncid, varname, &varid);
	errprt("extracting variables ID", varname, filename, retval, Fail2Read);
#ifdef GMTCOARDS
	if(isgmt)
	{
		long int size;
		size = getVarSize(filename, ncid, varname, varid);
		float vbound[2];
		vbound[0] = mathf.min(var,size);
		vbound[1] = mathf.max(var,size);
		retval = nc_redef(ncid);
		errprt("redefine NC file", filename, retval, Fail2File);
		retval = nc_put_att_float(ncid, varid, "actual_range", NC_FLOAT, 2, vbound);
		errprt("writing attributes of value boundary for", varname, filename, retval, Fail2Write);
		retval = nc_put_att_float(ncid, varid, _FillValue, NC_FLOAT, 1, &nf4float);
		errprt("writing attributes' name for", varname, filename, retval, Fail2Write);
		retval = nc_enddef(ncid);
		errprt("completing define mode", filename, retval, Fail2File);
	}
#endif
	retval = nc_put_vars_float(ncid, varid, subs, subn, subi, var);
	errprt("writing variables Value", varname, filename, retval, Fail2Write);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}
void seisnc::varput(const char *filename, const char *varname, double *var, size_t *subs, size_t *subn, ptrdiff_t *subi, bool isgmt)
{//putVars
	int ncid,varid,retval;
	retval = nc_open(filename, NC_WRITE, &ncid);
	errprt("openning file", filename, retval, Fail2Open);
	retval = nc_inq_varid(ncid, varname, &varid);
	errprt("extracting variables ID", varname, filename, retval, Fail2Read);
#ifdef GMTCOARDS
	if(isgmt)
	{
		long int size;
		size = getVarSize(filename, ncid, varname, varid);
		double vbound[2];
		vbound[0] = mathf.min(var,size);
		vbound[1] = mathf.max(var,size);
		retval = nc_redef(ncid);
		errprt("redefine NC file", filename, retval, Fail2File);
		retval = nc_put_att_double(ncid, varid, "actual_range", NC_DOUBLE, 2, vbound);
		errprt("writing attributes of value boundary for", varname, filename, retval, Fail2Write);
		retval = nc_put_att_double(ncid, varid, _FillValue, NC_DOUBLE, 1, &nf4double);
		errprt("writing attributes' name for", varname, filename, retval, Fail2Write);
		retval = nc_enddef(ncid);
		errprt("completing define mode", filename, retval, Fail2File);
	}
#endif
	retval = nc_put_vars_double(ncid, varid, subs, subn, subi, var);
	errprt("writing variables Value", varname, filename, retval, Fail2Write);
	retval = nc_close(ncid);
	errprt("closing file", filename, retval, Fail2Close);
}

void seisnc::grid_export(coord crd, deriv drv, cindx cdx, const char *crdfile, const char *drvfile)
{
	char errstr[256];
	int ncid,ncid2; 
	int dimid[SeisGeo], varid[13], retval, oldmode;
	Real *var[10];
	size_t size;
	int i,j,k;

	size = cdx.ni*cdx.nj*cdx.nk;
	for(i=0;i<10;i++)
		var[i] = new Real[size]();
	//coord export
	fprintf(stdout,"***Start to build the curvilinear coordinate file %s in the seisnc program\n",crdfile);

	com.flatten21D(crd.x, &var[0][0], cdx, 1);
	com.flatten21D(crd.y, &var[1][0], cdx, 1);
	com.flatten21D(crd.z, &var[2][0], cdx, 1);

	retval = nc_create(crdfile, NC_NETCDF4, &ncid2);
	errprt("create coord.nc file", crdfile, retval, Fail2Open);
	if(retval != NC_NOERR) printf("--------->file grid create err\n");
	
	nc_set_fill(ncid2, NC_NOFILL, &oldmode);
	nc_def_dim(ncid2, "i", cdx.ni, &dimid[0]);
	nc_def_dim(ncid2, "j", cdx.nj, &dimid[1]);
	nc_def_dim(ncid2, "k", cdx.nk, &dimid[2]);
	nc_def_var(ncid2, "x", SeisNCType, SeisGeo, dimid, &varid[0]);
	nc_def_var(ncid2, "y", SeisNCType, SeisGeo, dimid, &varid[1]);
	nc_def_var(ncid2, "z", SeisNCType, SeisGeo, dimid, &varid[2]);
	nc_enddef(ncid2);
	nc_put_var(ncid2,varid[0],&var[0][0]);
	nc_put_var(ncid2,varid[1],&var[1][0]);
	nc_put_var(ncid2,varid[2],&var[2][0]);
	retval = nc_close(ncid2);
	if(retval != NC_NOERR) printf("----->file grid close err: %s\n",nc_strerror(retval));
	
	fprintf(stdout,"---accomplished writing the curvilinear coordinate file\n");

	//deriv export
	fprintf(stdout,"***Start to build the partial derivative file %s in the seisnc program\n",drvfile);
	
	com.flatten21D(drv.xi_x, &var[0][0], cdx, 1);
	com.flatten21D(drv.xi_y, &var[1][0], cdx, 1);
	com.flatten21D(drv.xi_z, &var[2][0], cdx, 1);
	com.flatten21D(drv.eta_x, &var[3][0], cdx, 1);
	com.flatten21D(drv.eta_y, &var[4][0], cdx, 1);
	com.flatten21D(drv.eta_z, &var[5][0], cdx, 1);
	com.flatten21D(drv.zeta_x, &var[6][0], cdx, 1);
	com.flatten21D(drv.zeta_y, &var[7][0], cdx, 1);
	com.flatten21D(drv.zeta_z, &var[8][0], cdx, 1);
	com.flatten21D(drv.jac, &var[9][0], cdx, 1);
	
	//retval = nc_create(drvfile, NC_CLOBBER, &ncid);//classic
	//retval = nc_create(drvfile, NC_64BIT_OFFSET, &ncid);
	//retval = nc_create(drvfile, NC_64BIT_DATA, &ncid);
	retval = nc_create(drvfile, NC_NETCDF4, &ncid);
	errprt("create deriv.nc file", drvfile, retval, Fail2Open);
	if(retval != NC_NOERR) printf("--------->file metric create err\n");
	
	retval = nc_set_fill(ncid, NC_NOFILL, &oldmode);
	if(retval) printf("---->Error at set fill, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_dim(ncid, "i", cdx.ni, &dimid[0]);
	if(retval) printf("---->Error at def dim 0, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_dim(ncid, "j", cdx.nj, &dimid[1]);
	if(retval) printf("---->Error at def dim 1, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_dim(ncid, "k", cdx.nk, &dimid[2]);
	if(retval) printf("---->Error at def dim2, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_var(ncid, "xi_x", SeisNCType, SeisGeo, dimid, &varid[0]);
	if(retval) printf("---->Error at def var 0, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_var(ncid, "xi_y", SeisNCType, SeisGeo, dimid, &varid[1]);
	if(retval) printf("---->Error at def var 1, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_var(ncid, "xi_z", SeisNCType, SeisGeo, dimid, &varid[2]);
	if(retval) printf("---->Error at def var 2, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_var(ncid, "eta_x", SeisNCType, SeisGeo, dimid, &varid[3]);
	if(retval) printf("---->Error at def var 3, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_var(ncid, "eta_y", SeisNCType, SeisGeo, dimid, &varid[4]);
	if(retval) printf("---->Error at def var 4, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_var(ncid, "eta_z", SeisNCType, SeisGeo, dimid, &varid[5]);
	if(retval) printf("---->Error at def var 5, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_var(ncid, "zeta_x", SeisNCType, SeisGeo, dimid, &varid[6]);
	if(retval) printf("---->Error at def var 6, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_var(ncid, "zeta_y", SeisNCType, SeisGeo, dimid, &varid[7]);
	if(retval) printf("---->Error at def var 7, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_var(ncid, "zeta_z", SeisNCType, SeisGeo, dimid, &varid[8]);
	if(retval) printf("---->Error at def var 8, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_def_var(ncid, "jac", SeisNCType, SeisGeo, dimid, &varid[9]);
	if(retval) printf("---->Error at def var 9, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));

#ifdef GMTCOARDS
	Real x[cdx.ni],y[cdx.nj],z[cdx.nk];
	for(i=0;i<cdx.ni;i++)
		for(j=0;j<cdx.nj;j++)
			for(k=0;k<cdx.nk;k++)
			{
				x[i] = crd.x[cdx.ni1+i][cdx.nj1][cdx.nk1];
				y[i] = crd.x[cdx.ni1][cdx.nj1+j][cdx.nk1];
				z[i] = crd.x[cdx.ni1][cdx.nj1][cdx.nk1+k];
			}
	nc_def_var(ncid, "x", SeisNCType, 1, &dimid[0], &varid[10]);
	nc_def_var(ncid, "y", SeisNCType, 1, &dimid[1], &varid[11]);
	nc_def_var(ncid, "z", SeisNCType, 1, &dimid[2], &varid[12]);

	Real vbound[2];
	vbound[0] = mathf.min(&var[0][0], size);
	vbound[1] = mathf.max(&var[0][0], size);
	nc_put_att(ncid, varid[0], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[0], _FillValue, SeisNCType, 1, &NFillType);
	vbound[0] = mathf.min(&var[1][0], size);
	vbound[1] = mathf.max(&var[1][0], size);
	nc_put_att(ncid, varid[1], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[1], _FillValue, SeisNCType, 1, &NFillType);
	vbound[0] = mathf.min(&var[2][0], size);
	vbound[1] = mathf.max(&var[2][0], size);
	nc_put_att(ncid, varid[2], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[2], _FillValue, SeisNCType, 1, &NFillType);
	vbound[0] = mathf.min(&var[3][0], size);
	vbound[1] = mathf.max(&var[3][0], size);
	nc_put_att(ncid, varid[3], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[3], _FillValue, SeisNCType, 1, &NFillType);
	vbound[0] = mathf.min(&var[4][0], size);
	vbound[1] = mathf.max(&var[4][0], size);
	nc_put_att(ncid, varid[4], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[4], _FillValue, SeisNCType, 1, &NFillType);
	vbound[0] = mathf.min(&var[5][0], size);
	vbound[1] = mathf.max(&var[5][0], size);
	nc_put_att(ncid, varid[5], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[5], _FillValue, SeisNCType, 1, &NFillType);
	vbound[0] = mathf.min(&var[6][0], size);
	vbound[1] = mathf.max(&var[6][0], size);
	nc_put_att(ncid, varid[6], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[6], _FillValue, SeisNCType, 1, &NFillType);
	vbound[0] = mathf.min(&var[7][0], size);
	vbound[1] = mathf.max(&var[7][0], size);
	nc_put_att(ncid, varid[7], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[7], _FillValue, SeisNCType, 1, &NFillType);
	vbound[0] = mathf.min(&var[8][0], size);
	vbound[1] = mathf.max(&var[8][0], size);
	nc_put_att(ncid, varid[8], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[8], _FillValue, SeisNCType, 1, &NFillType);
	vbound[0] = mathf.min(&var[9][0], size);
	vbound[1] = mathf.max(&var[9][0], size);
	nc_put_att(ncid, varid[9], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[9], _FillValue, SeisNCType, 1, &NFillType);
	vbound[0] = mathf.min(x,cdx.ni);
	vbound[1] = mathf.max(x,cdx.ni);
	nc_put_att(ncid, varid[10], "actual_range", SeisNCType, 2, vbound);
	vbound[0] = mathf.min(y,cdx.nj);
	vbound[1] = mathf.max(y,cdx.nj);
	nc_put_att(ncid, varid[11], "actual_range", SeisNCType, 2, vbound);
	vbound[0] = mathf.min(z,cdx.nk);
	vbound[1] = mathf.max(z,cdx.nk);
	nc_put_att(ncid, varid[12], "actual_range", SeisNCType, 2, vbound);
#endif

	retval = nc_enddef(ncid);
	if(retval != NC_NOERR) printf("----->file metric enddef err: errcode=%d, errstr=%s\n",retval,nc_strerror(retval));

	retval = nc_put_var(ncid, varid[0], &var[0][0]);
	if(retval) printf("---->Error at put_var 0, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_put_var(ncid, varid[1], &var[1][0]);
	if(retval) printf("---->Error at put_var 1, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_put_var(ncid, varid[2], &var[2][0]);
	if(retval) printf("---->Error at put_var 2, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_put_var(ncid, varid[3], &var[3][0]);
	if(retval) printf("---->Error at put_var 3, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_put_var(ncid, varid[4], &var[4][0]);
	if(retval) printf("---->Error at put_var 4, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_put_var(ncid, varid[5], &var[5][0]);
	if(retval) printf("---->Error at put_var 5, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_put_var(ncid, varid[6], &var[6][0]);
	if(retval) printf("---->Error at put_var 6, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_put_var(ncid, varid[7], &var[7][0]);
	if(retval) printf("---->Error at put_var 7, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_put_var(ncid, varid[8], &var[8][0]);
	if(retval) printf("---->Error at put_var 8, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));
	retval = nc_put_var(ncid, varid[9], &var[9][0]);
	if(retval) printf("---->Error at put_var 9, errcode=%d, errstr=%s\n",retval,nc_strerror(retval));

#ifdef GMTCOARDS
	nc_put_var(ncid, varid[10], x);
	nc_put_var(ncid, varid[11], y);
	nc_put_var(ncid, varid[12], z);
#endif
	retval = nc_close(ncid);
	if(retval != NC_NOERR) printf("----->file metric close err: %s\n",nc_strerror(retval));

	for(i=0;i<10;i++)
		delete [] var[i];
	
	fprintf(stdout,"---accomplished writing the partial derivative file\n");
}
void seisnc::grid_import(coord crd, deriv drv, cindx cdx, const char *crdfile, const char *drvfile)
{
	int ncid, dimid[SeisGeo], varid[13], retval, oldmode;
	Real *var[10];
	size_t size;
	int i,j,k;

	size = cdx.ni*cdx.nj*cdx.nk;
	for(i=0;i<10;i++)
		var[i] = new Real[size]();
	//coord import
	fprintf(stdout,"***Start to reading the curvilinear coordinate file %s in the seisnc program\n",crdfile);

	retval = nc_open(crdfile, NC_NOWRITE, &ncid);
	errprt("read coord.nc file", crdfile, retval, Fail2Read);
	nc_inq_varid(ncid, "x", &varid[0]);
	nc_inq_varid(ncid, "y", &varid[1]);
	nc_inq_varid(ncid, "z", &varid[2]);
	nc_get_var(ncid,varid[0],&var[0][0]);
	nc_get_var(ncid,varid[1],&var[1][0]);
	nc_get_var(ncid,varid[2],&var[2][0]);
	nc_close(ncid);
	
	com.compress23D(&var[0][0], crd.x, cdx, 1);
	com.compress23D(&var[1][0], crd.y, cdx, 1);
	com.compress23D(&var[2][0], crd.z, cdx, 1);

	com.interpolated_extend(crd.x,cdx);
	com.interpolated_extend(crd.y,cdx);
	com.interpolated_extend(crd.z,cdx);
	
	fprintf(stdout,"---accomplished reading the curvilinear coordinate file\n");

	//deriv import
	fprintf(stdout,"***Start to reading the partial derivative file %s in the seisnc program\n",drvfile);
	
	retval = nc_open(drvfile, NC_NOWRITE, &ncid);
	errprt("read deriv.nc file", drvfile, retval, Fail2Read);
	nc_inq_varid(ncid, "xi_x", &varid[0]);
	nc_inq_varid(ncid, "xi_y", &varid[1]);
	nc_inq_varid(ncid, "xi_z", &varid[2]);
	nc_inq_varid(ncid, "eta_x", &varid[3]);
	nc_inq_varid(ncid, "eta_y", &varid[4]);
	nc_inq_varid(ncid, "eta_z", &varid[5]);
	nc_inq_varid(ncid, "zeta_x", &varid[6]);
	nc_inq_varid(ncid, "zeta_y", &varid[7]);
	nc_inq_varid(ncid, "zeta_z", &varid[8]);
	nc_inq_varid(ncid, "jac", &varid[9]);
	nc_get_var(ncid, varid[0], &var[0][0]);
	nc_get_var(ncid, varid[1], &var[1][0]);
	nc_get_var(ncid, varid[2], &var[2][0]);
	nc_get_var(ncid, varid[3], &var[3][0]);
	nc_get_var(ncid, varid[4], &var[4][0]);
	nc_get_var(ncid, varid[5], &var[5][0]);
	nc_get_var(ncid, varid[6], &var[6][0]);
	nc_get_var(ncid, varid[7], &var[7][0]);
	nc_get_var(ncid, varid[8], &var[8][0]);
	nc_get_var(ncid, varid[9], &var[9][0]);
	nc_close(ncid);
	
	com.compress23D(&var[0][0], drv.xi_x, cdx, 1);
	com.compress23D(&var[1][0], drv.xi_y, cdx, 1);
	com.compress23D(&var[2][0], drv.xi_z, cdx, 1);
	com.compress23D(&var[3][0], drv.eta_x, cdx, 1);
	com.compress23D(&var[4][0], drv.eta_y, cdx, 1);
	com.compress23D(&var[5][0], drv.eta_z, cdx, 1);
	com.compress23D(&var[6][0], drv.zeta_x, cdx, 1);
	com.compress23D(&var[7][0], drv.zeta_y, cdx, 1);
	com.compress23D(&var[8][0], drv.zeta_z, cdx, 1);
	com.compress23D(&var[9][0], drv.jac, cdx, 1);

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
	
	for(i=0;i<10;i++)
		delete [] var[i];
	
	fprintf(stdout,"---accomplished reading the partial derivative file\n");
}

void seisnc::media_export(mdpar mpa, cindx cdx, coord crd, const char *mediafile)
{
	int ncid, dimid[SeisGeo], varid[6], retval, oldmode;
	Real *var[3];
	size_t size;
	int i,j,k;

	size = cdx.ni*cdx.nj*cdx.nk;
	for(i=0;i<3;i++)
		var[i] = new Real[size]();
	//coord export
	fprintf(stdout,"***Start to build the media parameters file %s in the seisnc program\n",mediafile);

	com.flatten21D(mpa.alpha, &var[0][0], cdx, 1);
	com.flatten21D(mpa.beta, &var[1][0], cdx, 1);
	com.flatten21D(mpa.rho, &var[2][0], cdx, 1);

	retval = nc_create(mediafile, NC_NETCDF4, &ncid);
	errprt("create media.nc file", mediafile, retval, Fail2Open);
	nc_set_fill(ncid, NC_NOFILL, &oldmode);
	nc_def_dim(ncid, "i", cdx.ni, &dimid[0]);
	nc_def_dim(ncid, "j", cdx.nj, &dimid[1]);
	nc_def_dim(ncid, "k", cdx.nk, &dimid[2]);
	nc_def_var(ncid, "vp", SeisNCType, SeisGeo, dimid, &varid[0]);
	nc_def_var(ncid, "vs", SeisNCType, SeisGeo, dimid, &varid[1]);
	nc_def_var(ncid, "rho", SeisNCType, SeisGeo, dimid, &varid[2]);
	
#ifdef GMTCOARDS
	Real x[cdx.ni],y[cdx.nj],z[cdx.nk];
	for(i=0;i<cdx.ni;i++)
		for(j=0;j<cdx.nj;j++)
			for(k=0;k<cdx.nk;k++)
			{
				x[i] = crd.x[cdx.ni1+i][cdx.nj1][cdx.nk1];
				y[i] = crd.x[cdx.ni1][cdx.nj1+j][cdx.nk1];
				z[i] = crd.x[cdx.ni1][cdx.nj1][cdx.nk1+k];
			}
	nc_def_var(ncid, "x", SeisNCType, 1, &dimid[0], &varid[3]);
	nc_def_var(ncid, "y", SeisNCType, 1, &dimid[1], &varid[4]);
	nc_def_var(ncid, "z", SeisNCType, 1, &dimid[2], &varid[5]);

	Real vbound[2];
	
	vbound[0] = mathf.min(&var[0][0], size);
	vbound[1] = mathf.max(&var[0][0], size);
	nc_put_att(ncid, varid[0], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[0], _FillValue, SeisNCType, 1, &NFillType);
	vbound[0] = mathf.min(&var[1][0], size);
	vbound[1] = mathf.max(&var[1][0], size);
	nc_put_att(ncid, varid[1], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[1], _FillValue, SeisNCType, 1, &NFillType);
	vbound[0] = mathf.min(&var[2][0], size);
	vbound[1] = mathf.max(&var[2][0], size);
	nc_put_att(ncid, varid[2], "actual_range", SeisNCType, 2, vbound);
	nc_put_att(ncid, varid[2], _FillValue, SeisNCType, 1, &NFillType);
	
	vbound[0] = mathf.min(x,cdx.ni);
	vbound[1] = mathf.max(x,cdx.ni);
	nc_put_att(ncid, varid[10], "actual_range", SeisNCType, 2, vbound);
	vbound[0] = mathf.min(y,cdx.nj);
	vbound[1] = mathf.max(y,cdx.nj);
	nc_put_att(ncid, varid[11], "actual_range", SeisNCType, 2, vbound);
	vbound[0] = mathf.min(z,cdx.nk);
	vbound[1] = mathf.max(z,cdx.nk);
	nc_put_att(ncid, varid[12], "actual_range", SeisNCType, 2, vbound);
#endif
	nc_enddef(ncid);

	nc_put_var(ncid,varid[0],&var[0][0]);
	nc_put_var(ncid,varid[1],&var[1][0]);
	nc_put_var(ncid,varid[2],&var[2][0]);
#ifdef GMTCOARDS
	nc_put_var(ncid, varid[3], x);
	nc_put_var(ncid, varid[4], y);
	nc_put_var(ncid, varid[5], z);
#endif
	nc_close(ncid);

	for(i=0;i<3;i++)
		delete [] var[i];
	
	fprintf(stdout,"---accomplished writing the media parameters file\n");
}
void seisnc::media_import(mdpar mpa, cindx cdx, coord crd, const char *mediafile)
{
	int ncid, varid[3], retval, oldmode;
	Real *var[3];
	size_t size;
	int i,j,k;

	size = cdx.ni*cdx.nj*cdx.nk;
	for(i=0;i<3;i++)
		var[i] = new Real[size]();
	//coord import
	fprintf(stdout,"***Start to reading the media parameter file %s in the seisnc program\n",mediafile);

	retval = nc_open(mediafile, NC_NOWRITE, &ncid);
	errprt("read media.nc file", mediafile, retval, Fail2Read);
	nc_inq_varid(ncid, "vp", &varid[0]);
	nc_inq_varid(ncid, "vs", &varid[1]);
	nc_inq_varid(ncid, "rho", &varid[2]);
	nc_get_var(ncid,varid[0],&var[0][0]);
	nc_get_var(ncid,varid[1],&var[1][0]);
	nc_get_var(ncid,varid[2],&var[2][0]);
	nc_close(ncid);
	
	com.compress23D(&var[0][0], mpa.alpha, cdx, 1);
	com.compress23D(&var[1][0], mpa.beta, cdx, 1);
	com.compress23D(&var[2][0], mpa.rho, cdx, 1);

	com.equivalence_extend(mpa.alpha,cdx);
	com.equivalence_extend(mpa.beta,cdx);
	com.equivalence_extend(mpa.rho,cdx);
	
	fprintf(stdout,"---accomplished reading the media parameters file\n");

	for(i=0;i<3;i++)
		delete [] var[i];
	
}

void seisnc::force_export(force frc, const char *frcfile, int nfrc, int nstf)
{
	int ncid, dimid1[1], dimid2[2],dimid3[4],varid[11];
	size_t size;
	int i,j,k,m;
	int retval, oldmode;
	Real *var1;

	//flatten data
	size = nfrc*nstf;
	var1 = new Real[size];
	for(i=0;i<nfrc;i++)
		for(j=0;j<nstf;j++)
			var1[i*nstf+j] = frc.stf[i][j];
#ifdef SrcSmooth
	Real *var2;
	size = nfrc*LenNorm*LenNorm*LenNorm;
	var2 = new Real[size];
	for(i=0;i<nfrc;i++)
		for(j=0;j<LenNorm;j++)
			for(k=0;k<LenNorm;k++)
				for(m=0;m<LenNorm;m++)
					var2[i*LenNorm*LenNorm*LenNorm+j*LenNorm*LenNorm+k*LenNorm+m] = frc.dnorm[i][j][k][m];
#endif
	//store force data;
	fprintf(stdout,"***Start to build the force data file %s in the seisnc program\n",frcfile);
	retval = nc_create(frcfile, NC_NETCDF4, &ncid);
	errprt("create force.nc file", frcfile, retval, Fail2Open);
	nc_set_fill(ncid, NC_NOFILL, &oldmode);
	nc_def_dim(ncid, "ForceNumF", nfrc, &dimid1[0]);
	nc_def_var(ncid, "locx", NC_INT, 1, dimid1, &varid[0]);
	nc_def_var(ncid, "locy", NC_INT, 1, dimid1, &varid[1]);
	nc_def_var(ncid, "locz", NC_INT, 1, dimid1, &varid[2]);
	nc_def_var(ncid, "posx", SeisNCType, 1, dimid1, &varid[3]);
	nc_def_var(ncid, "posy", SeisNCType, 1, dimid1, &varid[4]);
	nc_def_var(ncid, "posz", SeisNCType, 1, dimid1, &varid[5]);
	nc_def_var(ncid, "fx", SeisNCType, 1, dimid1, &varid[6]);
	nc_def_var(ncid, "fy", SeisNCType, 1, dimid1, &varid[7]);
	nc_def_var(ncid, "fz", SeisNCType, 1, dimid1, &varid[8]);

	nc_def_dim(ncid, "ForceNumS", nfrc, &dimid2[0]);
	nc_def_dim(ncid, "StfNum", nstf, &dimid2[1]);
	nc_def_var(ncid, "stf", SeisNCType, 2, dimid2, &varid[9]);
#ifdef SrcSmooth
	nc_def_dim(ncid, "ForceNumD", nfrc, &dimid3[0]);
	nc_def_dim(ncid, "LenNormD2", LenNorm, &dimid3[1]);
	nc_def_dim(ncid, "LenNormD3", LenNorm, &dimid3[2]);
	nc_def_dim(ncid, "LenNormD4", LenNorm, &dimid3[3]);
	nc_def_var(ncid, "dnorm", SeisNCType, 4, dimid3, &varid[10]);
#endif
	nc_enddef(ncid);

	nc_put_var(ncid,varid[0],&frc.locx[0]);
	nc_put_var(ncid,varid[1],&frc.locy[0]);
	nc_put_var(ncid,varid[2],&frc.locz[0]);
	nc_put_var(ncid,varid[3],&frc.posx[0]);
	nc_put_var(ncid,varid[4],&frc.posy[0]);
	nc_put_var(ncid,varid[5],&frc.posz[0]);
	nc_put_var(ncid,varid[6],&frc.fx[0]);
	nc_put_var(ncid,varid[7],&frc.fy[0]);
	nc_put_var(ncid,varid[8],&frc.fz[0]);
	nc_put_var(ncid,varid[9],&var1[0]);
#ifdef SrcSmooth
	nc_put_var(ncid,varid[10],&var2[0]);
#endif
	nc_close(ncid);

#ifdef SrcSmooth
	delete [] var2;
#endif
	delete [] var1;
	
	fprintf(stdout,"---accomplished writing the force data file\n");

}
void seisnc::force_import(force frc, const char *frcfile, int nfrc, int nstf)
{
	int ncid,varid[11];
	size_t size;
	int i,j,k,m;
	int retval, oldmode;
	Real *var1;

	//force import
	fprintf(stdout,"***Start to reading the force data file %s in the seisnc program\n",frcfile);

	size = nfrc*nstf;
	var1 = new Real[size];
#ifdef SrcSmooth
	Real *var2;
	size = nfrc*LenNorm*LenNorm*LenNorm;
	var2 = new Real[size];
#endif
	retval = nc_open(frcfile, NC_NOWRITE, &ncid);
	errprt("read force.nc file", frcfile, retval, Fail2Read);
	nc_inq_varid(ncid, "locx", &varid[0]);
	nc_inq_varid(ncid, "locy", &varid[1]);
	nc_inq_varid(ncid, "locz", &varid[2]);
	nc_inq_varid(ncid, "posx", &varid[3]);
	nc_inq_varid(ncid, "posy", &varid[4]);
	nc_inq_varid(ncid, "posz", &varid[5]);
	nc_inq_varid(ncid, "fx", &varid[6]);
	nc_inq_varid(ncid, "fy", &varid[7]);
	nc_inq_varid(ncid, "fz", &varid[8]);
	nc_inq_varid(ncid, "stf", &varid[9]);
#ifdef SrcSmooth
	nc_inq_varid(ncid, "dnorm", &varid[10]);
#endif
	nc_get_var(ncid, varid[0], &frc.locx[0]);
	nc_get_var(ncid, varid[1], &frc.locy[0]);
	nc_get_var(ncid, varid[2], &frc.locz[0]);
	nc_get_var(ncid, varid[3], &frc.posx[0]);
	nc_get_var(ncid, varid[4], &frc.posy[0]);
	nc_get_var(ncid, varid[5], &frc.posz[0]);
	nc_get_var(ncid, varid[6], &frc.fx[0]);
	nc_get_var(ncid, varid[7], &frc.fy[0]);
	nc_get_var(ncid, varid[8], &frc.fz[0]);
	nc_get_var(ncid, varid[9], &var1[0]);
#ifdef SrcSmooth
	nc_get_var(ncid, varid[10], &var2[0]);
#endif
	nc_close(ncid);

	for(i=0;i<nfrc;i++)
		for(j=0;j<nstf;j++)
			frc.stf[i][j] = var1[i*nstf+j];
#ifdef SrcSmooth
	for(i=0;i<nfrc;i++)
		for(j=0;j<LenNorm;j++)
			for(k=0;k<LenNorm;k++)
				for(m=0;m<LenNorm;m++)
					frc.dnorm[i][j][k][m] = var2[i*LenNorm*LenNorm*LenNorm+j*LenNorm*LenNorm+k*LenNorm+m];
	delete [] var2;
#endif
	delete [] var1;

	fprintf(stdout,"---accomplished reading the force data file\n");

}
void seisnc::moment_export(moment mnt, const char *mntfile, int nmnt, int nstf)
{
	int ncid, dimid1[1], dimid2[2],dimid3[4],varid[14];
	size_t size;
	int i,j,k,m;
	int retval, oldmode;
	Real *var3;

	//flatten data
	size = nmnt*nstf;
	var3 = new Real[size];
	for(i=0;i<nmnt;i++)
		for(j=0;j<nstf;j++)
			var3[i*nstf+j] = mnt.stf[i][j];
#ifdef SrcSmooth
	Real *var4;
	size = nmnt*LenNorm*LenNorm*LenNorm;
	var4 = new Real[size];
	for(i=0;i<nmnt;i++)
		for(j=0;j<LenNorm;j++)
			for(k=0;k<LenNorm;k++)
				for(m=0;m<LenNorm;m++)
					var4[i*LenNorm*LenNorm*LenNorm+j*LenNorm*LenNorm+k*LenNorm+m] = mnt.dnorm[i][j][k][m];
#endif
	//store moment data;
	fprintf(stdout,"***Start to build the moment data file %s in the seisnc program\n",mntfile);
	retval = nc_create(mntfile, NC_NETCDF4, &ncid);
	errprt("create force.nc file", mntfile, retval, Fail2Open);
	nc_set_fill(ncid, NC_NOFILL, &oldmode);
	nc_def_dim(ncid, "MomentNumM", nmnt, &dimid1[0]);
	nc_def_var(ncid, "locx", NC_INT, 1, dimid1, &varid[0]);
	nc_def_var(ncid, "locy", NC_INT, 1, dimid1, &varid[1]);
	nc_def_var(ncid, "locz", NC_INT, 1, dimid1, &varid[2]);
	nc_def_var(ncid, "posx", SeisNCType, 1, dimid1, &varid[3]);
	nc_def_var(ncid, "posy", SeisNCType, 1, dimid1, &varid[4]);
	nc_def_var(ncid, "posz", SeisNCType, 1, dimid1, &varid[5]);
	nc_def_var(ncid, "mxx", SeisNCType, 1, dimid1, &varid[6]);
	nc_def_var(ncid, "myy", SeisNCType, 1, dimid1, &varid[7]);
	nc_def_var(ncid, "mzz", SeisNCType, 1, dimid1, &varid[8]);
	nc_def_var(ncid, "mxy", SeisNCType, 1, dimid1, &varid[9]);
	nc_def_var(ncid, "mxz", SeisNCType, 1, dimid1, &varid[10]);
	nc_def_var(ncid, "myz", SeisNCType, 1, dimid1, &varid[11]);

	nc_def_dim(ncid, "MomentNumS", nmnt, &dimid2[0]);
	nc_def_dim(ncid, "StfNum", nstf, &dimid2[1]);
	nc_def_var(ncid, "stf", SeisNCType, 2, dimid2, &varid[12]);
#ifdef SrcSmooth
	nc_def_dim(ncid, "MomentNumD", nmnt, &dimid3[0]);
	nc_def_dim(ncid, "LenNormD2", LenNorm, &dimid3[1]);
	nc_def_dim(ncid, "LenNormD3", LenNorm, &dimid3[2]);
	nc_def_dim(ncid, "LenNormD4", LenNorm, &dimid3[3]);
	nc_def_var(ncid, "dnorm", SeisNCType, 4, dimid3, &varid[13]);
#endif
	nc_enddef(ncid);

	nc_put_var(ncid,varid[0],&mnt.locx[0]);
	nc_put_var(ncid,varid[1],&mnt.locy[0]);
	nc_put_var(ncid,varid[2],&mnt.locz[0]);
	nc_put_var(ncid,varid[3],&mnt.posx[0]);
	nc_put_var(ncid,varid[4],&mnt.posy[0]);
	nc_put_var(ncid,varid[5],&mnt.posz[0]);
	nc_put_var(ncid,varid[6],&mnt.mxx[0]);
	nc_put_var(ncid,varid[7],&mnt.myy[0]);
	nc_put_var(ncid,varid[8],&mnt.mzz[0]);
	nc_put_var(ncid,varid[9],&mnt.mxy[0]);
	nc_put_var(ncid,varid[10],&mnt.mxz[0]);
	nc_put_var(ncid,varid[11],&mnt.myz[0]);
	nc_put_var(ncid,varid[12],&var3[0]);
#ifdef SrcSmooth
	nc_put_var(ncid,varid[13],&var4[0]);
#endif
	nc_close(ncid);

#ifdef SrcSmooth
	delete [] var4;
#endif
	delete [] var3;

	fprintf(stdout,"---accomplished writing the moment data file\n");
}
void seisnc::moment_import(moment mnt, const char *mntfile, int nmnt, int nstf)
{
	int ncid,varid[14];
	size_t size;
	int i,j,k,m;
	int retval, oldmode;
	Real *var3;

	//moment import
	fprintf(stdout,"***Start to reading the moment data file %s in the seisnc program\n",mntfile);

	size = nmnt*nstf;
	var3 = new Real[size];
#ifdef SrcSmooth
	Real *var4;
	size = nmnt*LenNorm*LenNorm*LenNorm;
	var4 = new Real[size];
#endif

	retval = nc_open(mntfile, NC_NOWRITE, &ncid);
	errprt("read moment.nc file", mntfile, retval, Fail2Read);
	nc_inq_varid(ncid, "locx", &varid[0]);
	nc_inq_varid(ncid, "locy", &varid[1]);
	nc_inq_varid(ncid, "locz", &varid[2]);
	nc_inq_varid(ncid, "posx", &varid[3]);
	nc_inq_varid(ncid, "posy", &varid[4]);
	nc_inq_varid(ncid, "posz", &varid[5]);
	nc_inq_varid(ncid, "mxx", &varid[6]);
	nc_inq_varid(ncid, "myy", &varid[7]);
	nc_inq_varid(ncid, "mzz", &varid[8]);
	nc_inq_varid(ncid, "mxy", &varid[9]);
	nc_inq_varid(ncid, "mxz", &varid[10]);
	nc_inq_varid(ncid, "myz", &varid[11]);
	nc_inq_varid(ncid, "stf", &varid[12]);
#ifdef SrcSmooth
	nc_inq_varid(ncid, "dnorm", &varid[13]);
#endif
	nc_get_var(ncid, varid[0], &mnt.locx[0]);
	nc_get_var(ncid, varid[1], &mnt.locy[0]);
	nc_get_var(ncid, varid[2], &mnt.locz[0]);
	nc_get_var(ncid, varid[3], &mnt.posx[0]);
	nc_get_var(ncid, varid[4], &mnt.posy[0]);
	nc_get_var(ncid, varid[5], &mnt.posz[0]);
	nc_get_var(ncid, varid[6], &mnt.mxx[0]);
	nc_get_var(ncid, varid[7], &mnt.myy[0]);
	nc_get_var(ncid, varid[8], &mnt.mzz[0]);
	nc_get_var(ncid, varid[9], &mnt.mxy[0]);
	nc_get_var(ncid, varid[10], &mnt.mxz[0]);
	nc_get_var(ncid, varid[11], &mnt.myz[0]);
	nc_get_var(ncid, varid[12], &var3[0]);
#ifdef SrcSmooth
	nc_get_var(ncid, varid[13], &var4[0]);
#endif
	nc_close(ncid);

	for(i=0;i<nmnt;i++)
		for(j=0;j<nstf;j++)
			mnt.stf[i][j] = var3[i*nstf+j];

#ifdef SrcSmooth
	for(i=0;i<nmnt;i++)
		for(j=0;j<LenNorm;j++)
			for(k=0;k<LenNorm;k++)
				for(m=0;m<LenNorm;m++)
					mnt.dnorm[i][j][k][m] = var4[i*LenNorm*LenNorm*LenNorm+j*LenNorm*LenNorm+k*LenNorm+m];
	delete [] var4;
#endif
	delete [] var3;
	fprintf(stdout,"---accomplished reading the moment data file\n");

}

void seisnc::focal_export(Rmom mnt, const char *mntfile)
{
	int ncid, dimid[2],dimid3[4],varid[13];
	size_t size1,size2;
	int i,j,k,m;
	int retval, oldmode;
	Real *var3;

	//flatten data
	size1 = mnt.np*mnt.nt;
	var3 = new Real[size1]();

#ifdef SrcSmooth
	Real *var4;
	size2 = mnt.np*LenNorm*LenNorm*LenNorm;
	var4 = new Real[size2];
	for(i=0;i<mnt.np;i++)
		for(j=0;j<LenNorm;j++)
			for(k=0;k<LenNorm;k++)
				for(m=0;m<LenNorm;m++)
					var4[i*LenNorm*LenNorm*LenNorm+j*LenNorm*LenNorm+k*LenNorm+m] = mnt.dnorm[i][j][k][m];
#endif
	//store focal data;
	fprintf(stdout,"***Start to build the focal data file %s in the seisnc program\n",mntfile);
	retval = nc_create(mntfile, NC_NETCDF4, &ncid);
	errprt("create focal.nc file", mntfile, retval, Fail2Open);
	nc_set_fill(ncid, NC_NOFILL, &oldmode);
	nc_def_dim(ncid, "FocalPointNum", mnt.np, &dimid[0]);
	nc_def_dim(ncid, "FocalTimeNum", mnt.nt, &dimid[1]);
	
	nc_def_var(ncid, "locx", NC_INT, 1, dimid, &varid[0]);
	nc_def_var(ncid, "locy", NC_INT, 1, dimid, &varid[1]);
	nc_def_var(ncid, "locz", NC_INT, 1, dimid, &varid[2]);
	nc_def_var(ncid, "posx", SeisNCType, 1, dimid, &varid[3]);
	nc_def_var(ncid, "posy", SeisNCType, 1, dimid, &varid[4]);
	nc_def_var(ncid, "posz", SeisNCType, 1, dimid, &varid[5]);
	nc_def_var(ncid, "mxx", SeisNCType, 2, dimid, &varid[6]);
	nc_def_var(ncid, "myy", SeisNCType, 2, dimid, &varid[7]);
	nc_def_var(ncid, "mzz", SeisNCType, 2, dimid, &varid[8]);
	nc_def_var(ncid, "mxy", SeisNCType, 2, dimid, &varid[9]);
	nc_def_var(ncid, "mxz", SeisNCType, 2, dimid, &varid[10]);
	nc_def_var(ncid, "myz", SeisNCType, 2, dimid, &varid[11]);
#ifdef SrcSmooth
	nc_def_dim(ncid, "F_Damp_Num", mnt.np, &dimid3[0]);
	nc_def_dim(ncid, "LenNormD2", LenNorm, &dimid3[1]);
	nc_def_dim(ncid, "LenNormD3", LenNorm, &dimid3[2]);
	nc_def_dim(ncid, "LenNormD4", LenNorm, &dimid3[3]);
	nc_def_var(ncid, "dnorm", SeisNCType, 4, dimid3, &varid[12]);
#endif
	nc_enddef(ncid);

	nc_put_var(ncid,varid[0],&mnt.locx[0]);
	nc_put_var(ncid,varid[1],&mnt.locy[0]);
	nc_put_var(ncid,varid[2],&mnt.locz[0]);
	nc_put_var(ncid,varid[3],&mnt.posx[0]);
	nc_put_var(ncid,varid[4],&mnt.posy[0]);
	nc_put_var(ncid,varid[5],&mnt.posz[0]);
	
	for(i=0;i<mnt.np;i++) for(j=0;j<mnt.nt;j++) var3[i*mnt.nt+j] = mnt.mxx[i][j];
	nc_put_var(ncid,varid[6],var3);
	
	for(i=0;i<mnt.np;i++) for(j=0;j<mnt.nt;j++) var3[i*mnt.nt+j] = mnt.myy[i][j];
	nc_put_var(ncid,varid[7],var3);
	
	for(i=0;i<mnt.np;i++) for(j=0;j<mnt.nt;j++) var3[i*mnt.nt+j] = mnt.mzz[i][j];
	nc_put_var(ncid,varid[8],var3);
	
	for(i=0;i<mnt.np;i++) for(j=0;j<mnt.nt;j++) var3[i*mnt.nt+j] = mnt.mxy[i][j];
	nc_put_var(ncid,varid[9],var3);
	
	for(i=0;i<mnt.np;i++) for(j=0;j<mnt.nt;j++) var3[i*mnt.nt+j] = mnt.mxz[i][j];
	nc_put_var(ncid,varid[10],var3);
	
	for(i=0;i<mnt.np;i++) for(j=0;j<mnt.nt;j++) var3[i*mnt.nt+j] = mnt.myz[i][j];
	nc_put_var(ncid,varid[11],var3);

#ifdef SrcSmooth
	nc_put_var(ncid,varid[12],&var4[0]);
#endif
	nc_close(ncid);

#ifdef SrcSmooth
	delete [] var4;
#endif
	delete [] var3;

	fprintf(stdout,"---accomplished writing the focal data file\n");
}
void seisnc::focal_import(Rmom mnt, const char *mntfile)
{
	int ncid,varid[13];
	size_t size;
	int i,j,k,m;
	int retval, oldmode;
	Real *var3;

	//focal import
	fprintf(stdout,"***Start to reading the focal data file %s in the seisnc program\n",mntfile);

	size = mnt.np*mnt.nt;
	var3 = new Real[size];
#ifdef SrcSmooth
	Real *var4;
	size = mnt.np*LenNorm*LenNorm*LenNorm;
	var4 = new Real[size];
#endif

	retval = nc_open(mntfile, NC_NOWRITE, &ncid);
	errprt("read moment.nc file", mntfile, retval, Fail2Read);
	nc_inq_varid(ncid, "locx", &varid[0]);
	nc_inq_varid(ncid, "locy", &varid[1]);
	nc_inq_varid(ncid, "locz", &varid[2]);
	nc_inq_varid(ncid, "posx", &varid[3]);
	nc_inq_varid(ncid, "posy", &varid[4]);
	nc_inq_varid(ncid, "posz", &varid[5]);
	nc_inq_varid(ncid, "mxx", &varid[6]);
	nc_inq_varid(ncid, "myy", &varid[7]);
	nc_inq_varid(ncid, "mzz", &varid[8]);
	nc_inq_varid(ncid, "mxy", &varid[9]);
	nc_inq_varid(ncid, "mxz", &varid[10]);
	nc_inq_varid(ncid, "myz", &varid[11]);
#ifdef SrcSmooth
	nc_inq_varid(ncid, "dnorm", &varid[12]);
#endif

	nc_get_var(ncid, varid[0], &mnt.locx[0]);
	nc_get_var(ncid, varid[1], &mnt.locy[0]);
	nc_get_var(ncid, varid[2], &mnt.locz[0]);
	nc_get_var(ncid, varid[3], &mnt.posx[0]);
	nc_get_var(ncid, varid[4], &mnt.posy[0]);
	nc_get_var(ncid, varid[5], &mnt.posz[0]);

	nc_get_var(ncid, varid[6],var3);
	for(i=0;i<mnt.np;i++) for(j=0;j<mnt.nt;j++) mnt.mxx[i][j] = var3[i*mnt.nt+j];
	
	nc_get_var(ncid, varid[7],var3);
	for(i=0;i<mnt.np;i++) for(j=0;j<mnt.nt;j++) mnt.myy[i][j] = var3[i*mnt.nt+j];
	
	nc_get_var(ncid, varid[8],var3);
	for(i=0;i<mnt.np;i++) for(j=0;j<mnt.nt;j++) mnt.mzz[i][j] = var3[i*mnt.nt+j];
	
	nc_get_var(ncid, varid[9],var3);
	for(i=0;i<mnt.np;i++) for(j=0;j<mnt.nt;j++) mnt.mxy[i][j] = var3[i*mnt.nt+j];
	
	nc_get_var(ncid, varid[10],var3);
	for(i=0;i<mnt.np;i++) for(j=0;j<mnt.nt;j++) mnt.mxz[i][j] = var3[i*mnt.nt+j];
	
	nc_get_var(ncid, varid[11],var3);
	for(i=0;i<mnt.np;i++) for(j=0;j<mnt.nt;j++) mnt.myz[i][j] = var3[i*mnt.nt+j];
	
#ifdef SrcSmooth
	nc_get_var(ncid, varid[12], &var4[0]);
#endif
	nc_close(ncid);


#ifdef SrcSmooth
	for(i=0;i<mnt.np;i++)
		for(j=0;j<LenNorm;j++)
			for(k=0;k<LenNorm;k++)
				for(m=0;m<LenNorm;m++)
					mnt.dnorm[i][j][k][m] = var4[i*LenNorm*LenNorm*LenNorm+j*LenNorm*LenNorm+k*LenNorm+m];
	delete [] var4;
#endif
	delete [] var3;
	fprintf(stdout,"---accomplished reading the focal data file\n");

}

void seisnc::point_export_def(int npnt, point *pnt)
{
	int dimid[2],varid[5],retval,oldmode;
	char str[]="Velocity and Stress seismogram";

	fprintf(stdout,"***Start to create Point-wave-field NC file %s in the seisnc porgram\n",pnt->file);
	
	retval = nc_create(pnt->file, NC_NETCDF4, &pnt->ncid);
	errprt("create point wave field file", pnt->file, retval, Fail2Open);
	nc_set_fill(pnt->ncid, NC_NOFILL, &oldmode);
	
	nc_def_dim(pnt->ncid, "TimeNum", NC_UNLIMITED, &dimid[0]);
	nc_def_dim(pnt->ncid, "PointNum", npnt, &dimid[1]);
	
	//wave field value
	nc_def_var(pnt->ncid, "Vx", SeisNCType, 2, dimid, &pnt->vid[0]);
	nc_def_var(pnt->ncid, "Vy", SeisNCType, 2, dimid, &pnt->vid[1]);
	nc_def_var(pnt->ncid, "Vz", SeisNCType, 2, dimid, &pnt->vid[2]);
	nc_def_var(pnt->ncid, "Txx", SeisNCType, 2, dimid, &pnt->vid[3]);
	nc_def_var(pnt->ncid, "Tyy", SeisNCType, 2, dimid, &pnt->vid[4]);
	nc_def_var(pnt->ncid, "Tzz", SeisNCType, 2, dimid, &pnt->vid[5]);
	nc_def_var(pnt->ncid, "Txy", SeisNCType, 2, dimid, &pnt->vid[6]);
	nc_def_var(pnt->ncid, "Txz", SeisNCType, 2, dimid, &pnt->vid[7]);
	nc_def_var(pnt->ncid, "Tyz", SeisNCType, 2, dimid, &pnt->vid[8]);
	nc_def_var(pnt->ncid, "currTime", SeisNCType, 1, &dimid[0], &pnt->vid[9]);
	//point pars value
	nc_def_var(pnt->ncid, "Pnum", NC_INT, 1, &dimid[1], &varid[0]);
	nc_def_var(pnt->ncid, "Lnum", NC_INT, 1, &dimid[1], &varid[1]);
	nc_def_var(pnt->ncid, "posx", SeisNCType, 1, &dimid[1], &varid[2]);
	nc_def_var(pnt->ncid, "posy", SeisNCType, 1, &dimid[1], &varid[3]);
	nc_def_var(pnt->ncid, "posz", SeisNCType, 1, &dimid[1], &varid[4]);
	
	nc_put_att_text(pnt->ncid, NC_GLOBAL, "Content", sizeof(str), str);

	nc_enddef(pnt->ncid);

	//input point pars
	nc_put_var(pnt->ncid, varid[0], pnt->Pnum);
	nc_put_var(pnt->ncid, varid[1], pnt->Lnum);
	nc_put_var(pnt->ncid, varid[2], pnt->posx);
	nc_put_var(pnt->ncid, varid[3], pnt->posy);
	nc_put_var(pnt->ncid, varid[4], pnt->posz);

	fprintf(stdout,"---accomplished create Point-wave-field file and input Point-pars!\n");

}
void seisnc::snap_export_def(int nsnap, snap snp)
{
	int dimid[4],retval,oldmode;
	int i;
	int buffer[SeisGeo];

	char str1[]="Velocity field";
	char str2[]="Stress field";
	char str3[]="Velocity and Stress field";

	for(i=0;i<nsnap;i++)
	{
		fprintf(stdout,"***Start to create Snap-wave-field NC file %s in the seisnc porgram\n",snp.file[i]);
		
		retval = nc_create(snp.file[i], NC_NETCDF4, &snp.ncid[i]);
		errprt("create snap wave field file", snp.file[i], retval, Fail2Open);
		nc_set_fill(snp.ncid[i], NC_NOFILL, &oldmode);
		
		nc_def_dim(snp.ncid[i], "TimeNum", NC_UNLIMITED, &dimid[0]);
		nc_def_dim(snp.ncid[i], "i", snp.xn[i], &dimid[1]);
		nc_def_dim(snp.ncid[i], "j", snp.yn[i], &dimid[2]);
		nc_def_dim(snp.ncid[i], "k", snp.zn[i], &dimid[3]);

		if(snp.cmp[i] == 1 || snp.cmp[i] == 3)
		{
			nc_def_var(snp.ncid[i], "Vx", SeisNCType, 4, dimid, &snp.vid[i][0]);
			nc_def_var(snp.ncid[i], "Vy", SeisNCType, 4, dimid, &snp.vid[i][1]);
			nc_def_var(snp.ncid[i], "Vz", SeisNCType, 4, dimid, &snp.vid[i][2]);
		}
		if(snp.cmp[i] == 2 || snp.cmp[i] == 3)
		{
			nc_def_var(snp.ncid[i], "Txx", SeisNCType, 4, dimid, &snp.vid[i][3]);
			nc_def_var(snp.ncid[i], "Tyy", SeisNCType, 4, dimid, &snp.vid[i][4]);
			nc_def_var(snp.ncid[i], "Tzz", SeisNCType, 4, dimid, &snp.vid[i][5]);
			nc_def_var(snp.ncid[i], "Txy", SeisNCType, 4, dimid, &snp.vid[i][6]);
			nc_def_var(snp.ncid[i], "Txz", SeisNCType, 4, dimid, &snp.vid[i][7]);
			nc_def_var(snp.ncid[i], "Tyz", SeisNCType, 4, dimid, &snp.vid[i][8]);
		}
		nc_def_var(snp.ncid[i], "currTime", SeisNCType, 1, &dimid[0], &snp.vid[i][9]);
		
		buffer[0] = snp.xs[i]; buffer[1] = snp.ys[i]; buffer[2] = snp.zs[i];
		nc_put_att(snp.ncid[i], NC_GLOBAL, "IdxStart", NC_INT, SeisGeo, buffer);
		buffer[0] = snp.xn[i]; buffer[1] = snp.yn[i]; buffer[2] = snp.zn[i];
		nc_put_att(snp.ncid[i], NC_GLOBAL, "IdxNumber", NC_INT, SeisGeo, buffer);
		buffer[0] = snp.xi[i]; buffer[1] = snp.yi[i]; buffer[2] = snp.zi[i];
		nc_put_att(snp.ncid[i], NC_GLOBAL, "IdxInterval", NC_INT, SeisGeo, buffer);
		nc_put_att(snp.ncid[i], NC_GLOBAL, "TimeInterval", NC_INT, 1, &snp.tinv[i]);

		if(snp.cmp[i] == 1)
			nc_put_att_text(snp.ncid[i], NC_GLOBAL, "Content", sizeof(str1), str1);
		else if(snp.cmp[i] == 2)
			nc_put_att_text(snp.ncid[i], NC_GLOBAL, "Content", sizeof(str2), str2);
		else
			nc_put_att_text(snp.ncid[i], NC_GLOBAL, "Content", sizeof(str3), str3);

		nc_enddef(snp.ncid[i]);
	
		fprintf(stdout,"---accomplished create Snap-wave-field file!\n");
	}

}
void seisnc::point_export(int it, Real stept, int npnt, point pnt, wfield wpoint)
{
	size_t idxs[2],idxn[2];
	ptrdiff_t idxi[2];
	Real t;
	// [ntime][npoint] == [low][fast]
	// time1 point[0] point[1] point[2] ... point[npnt-1]
	// time2 point[0] point[1] point[2] ... point[npnt-1]

      //idxs[0] = it/pnt.tinv -1; idxs[1] = 0;//1-1
	idxs[0] = it/pnt.tinv;    idxs[1] = 0;//1-1, input time step is zero, start from zero
	//current recording time index euqals to (total time step) divided (record intervals) and minus one;
	idxn[0] = 1;              idxn[1] = npnt;
	idxi[0] = 1;              idxi[1] = 1;

	t = it*stept;

	nc_put_vars(pnt.ncid, pnt.vid[0], idxs, idxn, idxi, wpoint.Vx);
	nc_put_vars(pnt.ncid, pnt.vid[1], idxs, idxn, idxi, wpoint.Vy);
	nc_put_vars(pnt.ncid, pnt.vid[2], idxs, idxn, idxi, wpoint.Vz);
	nc_put_vars(pnt.ncid, pnt.vid[3], idxs, idxn, idxi, wpoint.Txx);
	nc_put_vars(pnt.ncid, pnt.vid[4], idxs, idxn, idxi, wpoint.Tyy);
	nc_put_vars(pnt.ncid, pnt.vid[5], idxs, idxn, idxi, wpoint.Tzz);
	nc_put_vars(pnt.ncid, pnt.vid[6], idxs, idxn, idxi, wpoint.Txy);
	nc_put_vars(pnt.ncid, pnt.vid[7], idxs, idxn, idxi, wpoint.Txz);
	nc_put_vars(pnt.ncid, pnt.vid[8], idxs, idxn, idxi, wpoint.Tyz);
	nc_put_vars(pnt.ncid, pnt.vid[9], &idxs[0], &idxn[0], &idxi[0], &t);

}
void seisnc::snap_export(int it, Real stept, int nsnap, snap snp, cindx cdx, wfield wsnap)
{
	//formal full size output
	int i,j,k,m;
	size_t idxs[4],idxn[4];
	ptrdiff_t idxi[4];
	Real t;
	Real *var[9];
	int tid,size;

	//   nx1            ni1                                    ni2             nx2 
        //   0    1    2    3    4    5    6    7    8    9   10   11   12   13    14
        //   F    F    F    A    A    A    A    A    A    A    A    F    F    F
	//                  xs
	//                  1    2    3    4    5    6    7    8    
	for(m=0;m<nsnap;m++)
	{
		if(it%snp.tinv[m]!=0)
			continue;
		
		for(j=0;j<9;j++)
			var[j] = new Real[ snp.xn[m]*snp.yn[m]*snp.zn[m] ]();

		for(i=0;i<snp.xn[m];i++)
			for(j=0;j<snp.yn[m];j++)
				for(k=0;k<snp.zn[m];k++)
				{
					tid = (cdx.ni1 + snp.xs[m]-1 + i*snp.xi[m])*cdx.ny*cdx.nz +
					      (cdx.nj1 + snp.ys[m]-1 + j*snp.yi[m])*cdx.nz +
					      (cdx.nk1 + snp.zs[m]-1 + k*snp.zi[m]);
					size = i*snp.yn[m]*snp.zn[m] + j*snp.zn[m] + k;
					
					if(snp.cmp[m] == 1 || snp.cmp[m] == 3)
					{
						var[0][size] = wsnap.Vx[tid];
						var[1][size] = wsnap.Vy[tid];
						var[2][size] = wsnap.Vz[tid];
					}
					if(snp.cmp[m] == 2 || snp.cmp[m] == 3)
					{
						var[3][size] = wsnap.Txx[tid];
						var[4][size] = wsnap.Tyy[tid];
						var[5][size] = wsnap.Tzz[tid];
						var[6][size] = wsnap.Txy[tid];
						var[7][size] = wsnap.Txz[tid];
						var[8][size] = wsnap.Tyz[tid];
					}
				}

	      //idxs[0] = it/snp.tinv[m]-1; idxs[1] = 0;           idxs[2] = 0;             idxs[3] = 0;
		idxs[0] = it/snp.tinv[m];   idxs[1] = 0;           idxs[2] = 0;             idxs[3] = 0;//input time step is 0, start from 0
		idxn[0] = 1;                idxn[1] = snp.xn[m];   idxn[2] = snp.yn[m];     idxn[3] = snp.zn[m];
	      //idxi[0] = 1;                idxi[1] = snp.xi[m];   idxi[2] = snp.yi[m];     idxi[3] = snp.zi[m];//without extraction, directly from orginal array
		idxi[0] = 1;                idxi[1] = 1;   	   idxi[2] = 1;     	    idxi[3] = 1;//has already extracted

		t = it*stept;

		if(snp.cmp[m] == 1 || snp.cmp[m] == 3)
		{
			nc_put_vars(snp.ncid[m], snp.vid[m][0], idxs, idxn, idxi, var[0]);
			nc_put_vars(snp.ncid[m], snp.vid[m][1], idxs, idxn, idxi, var[1]);
			nc_put_vars(snp.ncid[m], snp.vid[m][2], idxs, idxn, idxi, var[2]);
		}
		if(snp.cmp[m] == 2 || snp.cmp[m] == 3)
		{
			nc_put_vars(snp.ncid[m], snp.vid[m][3], idxs, idxn, idxi, var[3]);
			nc_put_vars(snp.ncid[m], snp.vid[m][4], idxs, idxn, idxi, var[4]);
			nc_put_vars(snp.ncid[m], snp.vid[m][5], idxs, idxn, idxi, var[5]);
			nc_put_vars(snp.ncid[m], snp.vid[m][6], idxs, idxn, idxi, var[6]);
			nc_put_vars(snp.ncid[m], snp.vid[m][7], idxs, idxn, idxi, var[7]);
			nc_put_vars(snp.ncid[m], snp.vid[m][8], idxs, idxn, idxi, var[8]);
		}
		
		nc_put_vars(snp.ncid[m], snp.vid[m][9], &idxs[0], &idxn[0], &idxi[0], &t);
	
		for(i=0;i<9;i++)
			delete [] var[i];

	}

}
void seisnc::snap_export(int m, int it, Real stept, int nsnap, snap snp, Real **var)
{
	//new valid size output
	//m is snap number
	size_t idxs[4],idxn[4];
	ptrdiff_t idxi[4];
	Real t;

	//   nx1            ni1                                    ni2             nx2 
        //   0    1    2    3    4    5    6    7    8    9   10   11   12   13    14
        //   F    F    F    A    A    A    A    A    A    A    A    F    F    F
	//                  xs
	//                  1    2    3    4    5    6    7    8    


      //idxs[0] = it/snp.tinv[m]-1; idxs[1] = 0;           idxs[2] = 0;             idxs[3] = 0;
	idxs[0] = it;               idxs[1] = 0;           idxs[2] = 0;             idxs[3] = 0;//input time step is 0, start from 0
	idxn[0] = 1;                idxn[1] = snp.xn[m];   idxn[2] = snp.yn[m];     idxn[3] = snp.zn[m];
      //idxi[0] = 1;                idxi[1] = snp.xi[m];   idxi[2] = snp.yi[m];     idxi[3] = snp.zi[m];//without extraction, directly from orginal array
	idxi[0] = 1;                idxi[1] = 1;   	   idxi[2] = 1;     	    idxi[3] = 1;//has already extracted

	t = it*snp.tinv[m]*stept;

	if(snp.cmp[m] == 1 || snp.cmp[m] == 3)
	{
		nc_put_vars(snp.ncid[m], snp.vid[m][0], idxs, idxn, idxi, var[0]);
		nc_put_vars(snp.ncid[m], snp.vid[m][1], idxs, idxn, idxi, var[1]);
		nc_put_vars(snp.ncid[m], snp.vid[m][2], idxs, idxn, idxi, var[2]);
	}
	if(snp.cmp[m] == 2 || snp.cmp[m] == 3)
	{
		nc_put_vars(snp.ncid[m], snp.vid[m][3], idxs, idxn, idxi, var[3]);
		nc_put_vars(snp.ncid[m], snp.vid[m][4], idxs, idxn, idxi, var[4]);
		nc_put_vars(snp.ncid[m], snp.vid[m][5], idxs, idxn, idxi, var[5]);
		nc_put_vars(snp.ncid[m], snp.vid[m][6], idxs, idxn, idxi, var[6]);
		nc_put_vars(snp.ncid[m], snp.vid[m][7], idxs, idxn, idxi, var[7]);
		nc_put_vars(snp.ncid[m], snp.vid[m][8], idxs, idxn, idxi, var[8]);
	}

	nc_put_vars(snp.ncid[m], snp.vid[m][9], &idxs[0], &idxn[0], &idxi[0], &t);


}
void seisnc::point_import(point *pnt)
{
	int retval;

	fprintf(stdout,"***Start to read Point-wave-field NC file parameters %s in the seisnc porgram\n",pnt->file);

	retval = nc_open(pnt->file, NC_WRITE, &pnt->ncid);
	errprt("read point wave field file", pnt->file, retval, Fail2Read);

	nc_inq_varid(pnt->ncid, "Vx", &pnt->vid[0]);
	nc_inq_varid(pnt->ncid, "Vy", &pnt->vid[1]);
	nc_inq_varid(pnt->ncid, "Vz", &pnt->vid[2]);
	nc_inq_varid(pnt->ncid, "Txx", &pnt->vid[3]);
	nc_inq_varid(pnt->ncid, "Tyy", &pnt->vid[4]);
	nc_inq_varid(pnt->ncid, "Tzz", &pnt->vid[5]);
	nc_inq_varid(pnt->ncid, "Txy", &pnt->vid[6]);
	nc_inq_varid(pnt->ncid, "Txz", &pnt->vid[7]);
	nc_inq_varid(pnt->ncid, "Tyz", &pnt->vid[8]);
	nc_inq_varid(pnt->ncid, "currTime", &pnt->vid[9]);

	fprintf(stdout,"---accomplished reading point nc file pars\n");

}
void seisnc::snap_import(int nsnap, snap snp)
{
	int retval;
	int i;
	
	
	for(i=0;i<nsnap;i++)
	{
		retval = nc_open(snp.file[i], NC_WRITE, &snp.ncid[i]);
		errprt("read snap wave field file", snp.file[i], retval, Fail2Read);
		
		fprintf(stdout,"***Start to read Snap-wave-field NC file parameters %s in the seisnc porgram\n",snp.file[i]);
		
		if(snp.cmp[i] == 1 || snp.cmp[i] == 3)
		{
			nc_inq_varid(snp.ncid[i], "Vx", &snp.vid[i][0]);
			nc_inq_varid(snp.ncid[i], "Vy", &snp.vid[i][1]);
			nc_inq_varid(snp.ncid[i], "Vz", &snp.vid[i][2]);
		}
		if(snp.cmp[i] == 2 || snp.cmp[i] == 3)
		{
			nc_inq_varid(snp.ncid[i], "Txx", &snp.vid[i][3]);
			nc_inq_varid(snp.ncid[i], "Tyy", &snp.vid[i][4]);
			nc_inq_varid(snp.ncid[i], "Tzz", &snp.vid[i][5]);
			nc_inq_varid(snp.ncid[i], "Txy", &snp.vid[i][6]);
			nc_inq_varid(snp.ncid[i], "Txz", &snp.vid[i][7]);
			nc_inq_varid(snp.ncid[i], "Tyz", &snp.vid[i][8]);
		}
		nc_inq_varid(snp.ncid[i], "currTime", &snp.vid[i][9]);

		fprintf(stdout,"---accomplished read snap nc file pars\n");
	}

}
void seisnc::point_export_end(point pnt)
{
	nc_close(pnt.ncid);
	fprintf(stdout,"---accomplished store Point-wave-field file!\n");
}
void seisnc::snap_export_end(int nsnap, snap snp)
{
	int i;
	for(i=0;i<nsnap;i++)
		nc_close(snp.ncid[i]);
	fprintf(stdout,"---accomplished store Snap-wave-field file!\n");
}

void seisnc::wavebuffer_def(wavebuffer *wbuffer, cindx cdx)
{
	int dimid[SeisGeo],dimid2,retval,oldmode;
	int i;
	int buffer[SeisGeo];

	fprintf(stdout,"***Start to create Buffer-wave-field NC file %s in the seisnc porgram\n",wbuffer->file);

	retval = nc_create(wbuffer->file, NC_NETCDF4, &wbuffer->ncid);
	errprt("create buff wave field file", wbuffer->file, retval, Fail2Open);
	nc_set_fill(wbuffer->ncid, NC_NOFILL, &oldmode);

	nc_def_dim(wbuffer->ncid, "TimeNum", 1, &dimid2);
	nc_def_dim(wbuffer->ncid, "i", cdx.nx, &dimid[0]);
	nc_def_dim(wbuffer->ncid, "j", cdx.ny, &dimid[1]);
	nc_def_dim(wbuffer->ncid, "k", cdx.nz, &dimid[2]);

	nc_def_var(wbuffer->ncid, "Vx", SeisNCType, SeisGeo, dimid, &wbuffer->vid[0]);
	nc_def_var(wbuffer->ncid, "Vy", SeisNCType, SeisGeo, dimid, &wbuffer->vid[1]);
	nc_def_var(wbuffer->ncid, "Vz", SeisNCType, SeisGeo, dimid, &wbuffer->vid[2]);
	nc_def_var(wbuffer->ncid, "Txx", SeisNCType, SeisGeo, dimid, &wbuffer->vid[3]);
	nc_def_var(wbuffer->ncid, "Tyy", SeisNCType, SeisGeo, dimid, &wbuffer->vid[4]);
	nc_def_var(wbuffer->ncid, "Tzz", SeisNCType, SeisGeo, dimid, &wbuffer->vid[5]);
	nc_def_var(wbuffer->ncid, "Txy", SeisNCType, SeisGeo, dimid, &wbuffer->vid[6]);
	nc_def_var(wbuffer->ncid, "Txz", SeisNCType, SeisGeo, dimid, &wbuffer->vid[7]);
	nc_def_var(wbuffer->ncid, "Tyz", SeisNCType, SeisGeo, dimid, &wbuffer->vid[8]);
	nc_def_var(wbuffer->ncid, "currTime", SeisNCType, 1, &dimid2, &wbuffer->vid[9]);

	nc_put_att_text(wbuffer->ncid, NC_GLOBAL, "Content", SeisStrLen2, "Full velocity and stress field");
	
	nc_enddef(wbuffer->ncid);

	fprintf(stdout,"---accomplished create Buffer-wave-field file!\n");

}
void seisnc::wavebuffer_export(int it, Real stept, wavebuffer wbuffer, cindx cdx, wfield wsnap)
{
	int dimid[SeisGeo];
	int i;
	Real time;
	size_t idxs[3];
	size_t idxn[3];
	ptrdiff_t idxi[3];

	idxs[0] = 0;      idxs[1] = 0;      idxs[2] = 0;
	idxn[0] = cdx.nx; idxn[1] = cdx.ny; idxn[2] = cdx.nz;
	idxi[0] = 1;      idxi[1] = 1;      idxi[2] = 1;

	nc_put_vars(wbuffer.ncid, wbuffer.vid[0], idxs, idxn, idxi, wsnap.Vx);
	nc_put_vars(wbuffer.ncid, wbuffer.vid[1], idxs, idxn, idxi, wsnap.Vy);
	nc_put_vars(wbuffer.ncid, wbuffer.vid[2], idxs, idxn, idxi, wsnap.Vz);
	nc_put_vars(wbuffer.ncid, wbuffer.vid[3], idxs, idxn, idxi, wsnap.Txx);
	nc_put_vars(wbuffer.ncid, wbuffer.vid[4], idxs, idxn, idxi, wsnap.Tyy);
	nc_put_vars(wbuffer.ncid, wbuffer.vid[5], idxs, idxn, idxi, wsnap.Tzz);
	nc_put_vars(wbuffer.ncid, wbuffer.vid[6], idxs, idxn, idxi, wsnap.Txy);
	nc_put_vars(wbuffer.ncid, wbuffer.vid[7], idxs, idxn, idxi, wsnap.Txz);
	nc_put_vars(wbuffer.ncid, wbuffer.vid[8], idxs, idxn, idxi, wsnap.Tyz);
	
	time = it*stept;
	nc_put_var(wbuffer.ncid, wbuffer.vid[9], &time);
	
	fprintf(stdout,"---accomplished update Buffer-wave-field at time %f (step %d)\n",time,it);
	
}
void seisnc::wavebuffer_import(int *restartT, Real stept, wavebuffer *wbuffer, cindx cdx, wfield wsnap)
{
	int i;
	int retval;
	Real time;

	fprintf(stdout,"***Start to read Buffer-wave-field NC file %s in the seisnc porgram\n",wbuffer->file);
	
	retval = nc_open(wbuffer->file, NC_WRITE, &wbuffer->ncid);
	errprt("read wavebuffer.nc file", wbuffer->file, retval, Fail2Read);
	
	nc_inq_varid(wbuffer->ncid, "Vx", &wbuffer->vid[0]);
	nc_inq_varid(wbuffer->ncid, "Vy", &wbuffer->vid[1]);
	nc_inq_varid(wbuffer->ncid, "Vz", &wbuffer->vid[2]);
	nc_inq_varid(wbuffer->ncid, "Txx", &wbuffer->vid[3]);
	nc_inq_varid(wbuffer->ncid, "Tyy", &wbuffer->vid[4]);
	nc_inq_varid(wbuffer->ncid, "Tzz", &wbuffer->vid[5]);
	nc_inq_varid(wbuffer->ncid, "Txy", &wbuffer->vid[6]);
	nc_inq_varid(wbuffer->ncid, "Txz", &wbuffer->vid[7]);
	nc_inq_varid(wbuffer->ncid, "Tyz", &wbuffer->vid[8]);
	nc_inq_varid(wbuffer->ncid, "currTime", &wbuffer->vid[9]);

	nc_get_var(wbuffer->ncid, wbuffer->vid[0], wsnap.Vx);
	nc_get_var(wbuffer->ncid, wbuffer->vid[1], wsnap.Vy);
	nc_get_var(wbuffer->ncid, wbuffer->vid[2], wsnap.Vz);
	nc_get_var(wbuffer->ncid, wbuffer->vid[3], wsnap.Txx);
	nc_get_var(wbuffer->ncid, wbuffer->vid[4], wsnap.Tyy);
	nc_get_var(wbuffer->ncid, wbuffer->vid[5], wsnap.Tzz);
	nc_get_var(wbuffer->ncid, wbuffer->vid[6], wsnap.Txy);
	nc_get_var(wbuffer->ncid, wbuffer->vid[7], wsnap.Txz);
	nc_get_var(wbuffer->ncid, wbuffer->vid[8], wsnap.Tyz);
	nc_get_var(wbuffer->ncid, wbuffer->vid[9], &time);

	*restartT = time/stept;
	
	fprintf(stdout,"---accomplished read Buffer-wave-field file and restart at %d step (%f seconds)!\n",*restartT, time);

}
void seisnc::wavebuffer_end(wavebuffer wbuffer)
{
	nc_close(wbuffer.ncid);
	fprintf(stdout,"---accomplished store Buffer-wave-field file!\n");
}

void seisnc::PV_export(PeakVel pv, const char *pvname, cindx cdx)
{
	char errstr[256];
	int ncid;
	int dimid[2],varid[3],retval;
	size_t size;

	int i,j;

	Real *var;
	var = new Real[cdx.ni*cdx.nj]();

	fprintf(stdout,"***Start to build the peak velocity file %s in the seisnc program\n",pvname);

	retval = nc_create(pvname, NC_NETCDF4, &ncid);
	errprt("create peakvel.nc file", pvname, retval, Fail2Open);
	
	nc_def_dim(ncid, "i", cdx.ni, &dimid[0]);
	nc_def_dim(ncid, "j", cdx.nj, &dimid[1]);
	nc_def_var(ncid, "Vx", SeisNCType, 2, dimid, &varid[0]);
	nc_def_var(ncid, "Vy", SeisNCType, 2, dimid, &varid[1]);
	nc_def_var(ncid, "Vz", SeisNCType, 2, dimid, &varid[2]);
	nc_enddef(ncid);

	for(i=0;i<cdx.ni;i++)
		for(j=0;j<cdx.nj;j++)
			var[i*cdx.nj+j] = pv.Vx[ (i+cdx.ni1)*cdx.ny + j+cdx.nj1 ];
	nc_put_var(ncid,varid[0],var);

	for(i=0;i<cdx.ni;i++)
		for(j=0;j<cdx.nj;j++)
			var[i*cdx.nj+j] = pv.Vy[ (i+cdx.ni1)*cdx.ny + j+cdx.nj1 ];
	nc_put_var(ncid,varid[1],var);

	for(i=0;i<cdx.ni;i++)
		for(j=0;j<cdx.nj;j++)
			var[i*cdx.nj+j] = pv.Vy[ (i+cdx.ni1)*cdx.ny + j+cdx.nj1 ];
	nc_put_var(ncid,varid[2],var);

	retval = nc_close(ncid);
	if(retval != NC_NOERR) printf("----->file peakvel close err: %s\n",nc_strerror(retval));

	delete [] var;
	var = NULL;
	
	fprintf(stdout,"---accomplished writing the peak velocity file\n");

}







































