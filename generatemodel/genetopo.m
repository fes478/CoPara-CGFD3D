clear; clc;

nx=201;
ny=201;
nz=200;
nlayer=1;

%% interp topo

mx = linspace(0,10e3, nx);%lon
my = linspace(0,10e3, ny);%lat

[gridX,gridY]=meshgrid(mx,my);
gridX = gridX';
gridY = gridY';

r=sqrt( (gridX-5e3).^2+(gridY-5e3).^2 );
a=1.2e3;
ztopo=a*exp(-r.^2/a^2);

surf(gridX,gridY,ztopo);


%%

Zdep = 10e3;%here doesnot need to exceed bounds

ninterface = 2; %nlayer = ninterface-1
zlayer= zeros(ninterface);
zlayer(1)=0; % to be refresh
zlayer(2)=-10e3; % to be refresh

disp('start write');

writeGPU = 1;

if writeGPU
	%write for gpu
	fp2 = fopen('./gridvmap_G.dat','w');

	fprintf(fp2,'#grid vmap ascii data\n');
	fprintf(fp2,'vmap_dims = %d %d %d\n',nx,ny,ninterface);

	fprintf(fp2,'vmap_gridpoints = %d\n',nz-1);
	fprintf(fp2,'vmap_equalspacing = %d\n',1);

	fprintf(fp2,'#from top to bottom, Xcord Ycord Zsurf1 Zsurf2 Zsurf3 Zsurf4\n');
	fprintf(fp2,'<vmap_anchor>\n');

	for i=1:nx
		for j=1:ny
			% fprintf(fp2,'%f %f %f %f',gridX(i,j),gridY(i,j),topo(i,j),topo(i,j)-5e3);
			% for k=ninterface-1:ninterface
			% 	fprintf(fp2,' %f',zlayer(k));
            % end
            % fprintf(fp2,'%f %f %f %f',gridX(i,j),gridY(i,j),zlayer(1),zlayer(2));
            fprintf(fp2,'%f %f %f %f',gridX(i,j),gridY(i,j),ztopo(i,j),zlayer(2));
			fprintf(fp2,'\n');
		end
	end

	fclose(fp2);

end


disp('ok');

