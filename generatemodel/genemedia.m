clc;clear;

writegpu = 1;


%% interp topo

NX = 50;
NY = 50;
NZ = 200;

mx = linspace(-5e3,15e3, NX);%lon
my = linspace(-5e3,15e3, NY);%lat
Zget = linspace(0,-10,200);

Onx = NX;
Ony = NY;
Onz = NZ;



%%  emprical relation

Vp = ones(Onx,Ony,Onz)*6.4;
Vs = ones(Onx,Ony,Onz)*3.7;
rho = ones(Onx,Ony,Onz)*2.81;

Vp = Vp*1e3;
Vs = Vs*1e3;
rho = rho*1e3;
Zget = Zget*1e3;
velmax = max(max(max(Vp)));
velmin = min(min(min(Vs)));

%%
figure(1)
vsc = squeeze(Vs(:,10,:));
imagesc(mx/1e3,Zget/1e3,vsc');
set(gca,'YDir','normal');
colorbar();

%%

if writegpu
    Vs = flip(Vs,3);
    Vp = flip(Vp,3);
    rho = flip(rho,3);
end

figure(2)
vsc = squeeze(Vs(:,10,:));
imagesc(mx/1e3,Zget/1e3,vsc');
colorbar();
set(gca,'YDir','normal');


%% write

if writegpu

    disp('start to write GPU');

    Zget = Zget(end:-1:1);

    ncid = netcdf.create('./Vsim_gpu.nc','netcdf4');%rewrite

    dimid = ones(1,3);%fast mid low
    dimid(1) = netcdf.defDim(ncid,'depth',Onz);
    dimid(2) = netcdf.defDim(ncid,'y',Ony);
    dimid(3) = netcdf.defDim(ncid,'x',Onx);

    varid = ones(1,7);
    varid(1) = netcdf.defVar(ncid, 'x', 'NC_FLOAT', dimid(3));
    varid(2) = netcdf.defVar(ncid, 'y', 'NC_FLOAT', dimid(2));
    varid(3) = netcdf.defVar(ncid, 'depth', 'NC_FLOAT', dimid(1));

    varid(4) = netcdf.defVar(ncid, 'depth2sealevel', 'NC_FLOAT', dimid(1));
    varid(5) = netcdf.defVar(ncid, 'Vp', 'NC_FLOAT', dimid);
    varid(6) = netcdf.defVar(ncid, 'Vs', 'NC_FLOAT', dimid);
    varid(7) = netcdf.defVar(ncid, 'rho', 'NC_FLOAT', dimid);

    % varid1 = netcdf.getConstant('Global');
    % netcdf.putAtt(ncid, varid1, 'X_lon_end', vellon2);
    % varid2 = netcdf.getConstant('Global');
    % netcdf.putAtt(ncid, varid2, 'Y_lat_end', Flat2);
    % varid3 = netcdf.getConstant('Global');
    % netcdf.putAtt(ncid, varid3, 'X_lon_start', vellon1);
    % varid4 = netcdf.getConstant('Global');
    % netcdf.putAtt(ncid, varid4, 'Y_lat_start', vellat1);
    varid5 = netcdf.getConstant('Global');
    netcdf.putAtt(ncid, varid5, 'sealevel', 0.0);
    varid6 = netcdf.getConstant('Global');
    netcdf.putAtt(ncid, varid6, 'velmax', velmax);
    varid7 = netcdf.getConstant('Global');
    netcdf.putAtt(ncid, varid7, 'velmin', velmin);

    netcdf.endDef(ncid);

    netcdf.putVar(ncid, varid(1), [0],[Onx],[1], mx);
    netcdf.putVar(ncid, varid(2), [0],[Ony],[1], my);
    netcdf.putVar(ncid, varid(3), [0],[Onz],[1], Zget);
    netcdf.putVar(ncid, varid(4), [0],[Onz],[1], -Zget);
    VAR = permute(Vp, [3,2,1]);%should according to dimension define
    netcdf.putVar(ncid, varid(5), VAR);
    VAR = permute(Vs, [3,2,1]);
    netcdf.putVar(ncid, varid(6), VAR);
    VAR = permute(rho, [3,2,1]);
    netcdf.putVar(ncid, varid(7), VAR);

    netcdf.close(ncid)

    disp('pass for_output');

end