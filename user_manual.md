## NOTICE:
1. This is a **GPU parallel** program that support **multi nodes** execution.

2. This program is suitable for complex interface and 3D medium situations.
3. The actual model construction an numerical calculation are carried out on the Cartesian coordinate (automatically transformed from curve coordinate).

4. The coordinate reference frame follows **East-North-Up**  as  X-Y-Z (right-hand spiral coordinate).


	//                          
	//        Z-dir          O-----------------------------O
	//        |           x  |                          x  |
	//        |        x     |                       x     |
	//        |     x        |                    x        |
	//        |  x           |                 x           |
	//        O-----------------------------O              |
	//        |              |              |              |
	//        |              |              |              |
	//        |              |              |              |
	//        |              |        Y-dir |              |
	//        |              |     x        |              |
	//        |              |  x           |              |
	//        |              0--------------|--------------O
	//        |           x                 |           x
	//        |        x                    |        x
	//        |     x                       |     x
	//        |  x                          |  x
	//        O-----------------------------O-------------X-dir
	//
	//

5. All units follow the International System of Units (meter, second, pascal).
6. PML and exponential absorbing boundary layer is supported (**PML** is recommended). 
7. Input source should be **velocity** or **moment rate**, and volume-type fault rupture process input is supported.
8. Output is velocity and stress in point, line, surface, and volume type.
8.  The **minimum grid points per wavelength(PPW) is 8**. If the maximum grid spacing is dh, the minimum velocity is Vmin, the maximum velocity is Vmax, then the maximum reliable frequency (f) is f=Vmin/(PPW\*dh) = Vmin/(8\*dh), the maximum time interval (dt) allowed is dt=0.76\*dh/Vmax. Example: Vmin=1000 m/s, Vmax = 3000m/s, dh = 100m, f = 1000/8/100=1.25, dt = 0.76*100/3000=0.025. So the simulating `dt` should small than 0.025, and the maximum frequency of synthetic waveform will be 1.25.



## Folder description

There are 5 folders and 3 files in the program directory.

### 1. Execution use

Folder src: souce code;

Folder obj:  compiled intermediate files;

Folder bin:  executable program (named wavesim)

File Makefile: compile file

### 2. Example use

Folder EXAMPLE: an example model and corresponding visualization programs.

Folder generatemodel: programs used for model construction.

### 3. Documentation use

File README.md:  a brief introduction to this method.

File user_manual.md: this file, program instructions.



## Program compile

Hardware requirements: NVIDIA GPU with compute capability 6.0 (compute\_60,sm\_60) or higher.

Program language: CUDA, C++, MPI .

Software required: CUDA10.0, mpich3.2.0, netcdf-c4.7.3 .



1. set environment variables for cuda, netcdf, and mpi.
2. set the path of cuda, netcdf, mpi in `Makefile`.

3. compile the `Makefile`.



The `HYindex` Macro in Makefile is used for hybrid gird implementation, which is suitable for box-shaped structure with undulating surface. This macro is enabled by default (`HYindex := ON`).

If your target geologic body requires the use of completely non-orthogonal grids, you should cancel this macro (`HYindex := `).



## Model construct

Program here are used to construct models, and the default output is the hillmodel demo used in EXAMPLE  folder.

`genetopo.m` is used to generate grid file in ASCII type.

`genemedia.m` is used to generate medium file in netcdf type.

`genesource.py` is used to generate moment rate source file in netcdf type.

These are just sample programs that can provide reference for building complex practical models. The following points should also be noted:

1. Be sure to pay attention to the writing of NC-type velocity files. When building a real model in the later stage, it is recommended to directly apply the code used to write files in `genemedia.m` .
2. The spatial range of the velocity model must be greater than or equal to the grid model.
3. If you want to use 200 grid points to build a one-layer-model in `genetopo.m` , that means there will be 199 grid intervals and 2 interfaces along Z direction. The `vmap_gridpoints` should be the number of intervals. The `vmap_equalspacing` means whether to apply same space for adjacent grid points in this layer ( 1 to enable, 0 for disable). 
4. When you use the netcdf-type input  moment rate source, please confirm whether moment rate has been multiplied by the sub fault area. If not, set the area as a constant in `source.dat` (will introduce in wavefieldsimulation par) file to `simulate_damp` parameter.



## Wavefield simulation

The default wavefield simulation parameters are set in Folder hillmodel, and executed under Folder EXAMPLE path by File `run.sh`. The result can be viewed by python programs in Folder EXAMPLE.

### set parameters in hillmodel

create 3 folder: checkpoint, input, and output.

SeisFD3D.conf: main parameters.

​				seispath: project path shouled set first.

​				BlockPerGrid: GPU kernel launch parameter for block. 

​				ThreadPerBlock: GPU kernel launch parameter for block. 

​				hybrid_grid: set 1 to enable hygrid, needed complie first by HYindex marco.

​				conversion_interface_depth: use curve grid above this interface, and cartisian grid below this interface.

​				reference_velocity: the minimal Vs.

​				output_peakvel: set 1 to output top surface peak velocity (Vx, Vy, and Vz).

source.dat: source parameters.

​				source_hyper_height: the Maximum reference height, exceeding this value will be applied to the surface.

​				distance2meter: distance unit, used for single force (type 1) or moment tensor (type 3).

​				source_type: choose source type, 1 for extrenal volume type input, 2 for single force, 3 for moment tensor.
​					If you use explosive source, you should input moment tensor (type 1 or 3) with Mxx, Myy, Mzz euqals to 1,
​					and Mxz, Mxy, Myz equals to 0. But in the oil exploration, you shoule use single force (type 2) with Vx, Vy and Vz equals to 1.
​					The unit of X,Y,Z in source_type 2 and 3 is km, and in source type 1 is m.

​				notice: M0=miu*D*A. miu(rigidity, 1e10) will be read from extermal input medium file and mutiplied in program automatically.
​					The input D should be rate(velocity), not slip(displacement),
​					Pay more attention to A (the area of subfault), recommand to mutiplied it when writing the source file, if not, you
​					should set a constant (average) area to `simulate_damp'.

station.dat: station parameters.

​				topo_hyper_height: the Maximum reference height, exceeding this value will be applied to the surface.

​				tinv_of_seismo: output interval for line and point type.

​				snap output: outputID Xstart Ystart Zstart Xnum Ynum Znum Xint Yint Zint tinv cmp
​					     snap_001    1     1     200    201  201  1    1    1    1    20   3
​			            the prevous pars means, to output the full horizontal grid topo surface snapshot of velocity and stress.
​				    in SeisFD3D.conf, we arranged number of X,Y,Z equals to 201,201,200.
​				    the Start series to confirm plot which interface, here Zstart=200 means to plot the topo surface..
​				    the Count and Stride series to confirm the dense of output grid.
​				    the Tinv confirms output time interval.
​				    the CMP confirms output file type.
​				    Volume type output are supported (Xnum, Ynum ,Znum are not 1).

​				line output: outputID Xstart Ystart  Zstart  Xint  Yint   Zint Num
​					    line_001 3.0e3  0.5e3   0.0e3   0.0   1.0e3  0.0  9
​				    the previous pars means, to record a seismoline from (3km,0.5km,0) to (3km, 8.5km,0), totally 9 stations.

​				point output: outputID  X       Y      Z
​					      recv_001  5.0e3   5.0e3  -5.0e3   #  to record (5km,5km,-5km) station
​					      recv_002  5.0e3   5.0e3  9000e3   #  to record surface station on (5km, 5km, surface)

​				output notice:
​					peakvel.nc: full suface grid Vx, Vy, and Vz, dimension as (NX,NY)
​					seismo.nc: point and line output. Lnum is line number, Pnum is point number. Lnum=0, is point output.
​					snap_001.nc: velocity and stress snapshot output.

absorb.dat: absorbing parameters

​				abs_number: Xmin Xmax Ymin Ymax Zmin Zmax
​					grid point arranged for absorbing layer, 12 points is enough.
​					sequencial as X, Y ,Z order, form index small to large.
​					The LAST one represent the top surface, set 0 mean free surface.

​				abs_velocity: each absorbing layer's reference velocity, should use the maximum Vp. 

​				NOTICE: as a trick, directly use the default par, except for Zmax(free surface use).
​					detail usage should refer to ZhangWei and ShenYang(GJI, 2010), or Master Thesis of YangXina(2010, USTC, in Chinese)

device.dat:

​				used_device_number_garray3: set how many devices hired for computing in GPU node.

​				device_ydims_garray3: how many devices hired for Y direction.

​				used_device_id_garray3: which device are available for use.

​				NOTICE:
​					here should change garray3 as your GPU node name.
​					one node may have many devices, such as TITAN V has 8 devices.
​					multi-node are supported.
​					EXAMPLE:
​					used_device_number_garray4 = 6
​					device_ydims_garray4 = 2
​					used_device_id_garray4 = 2 3 4 5 6 7
​					means, we hired 6 devices on GPU node garray4 for computation. the device serial number is 2,3,4,5,6,7.
​					The data of this node will be divided into 3 parts in the X direction and 2 parts in the Y direction.

machinefile:

​				garray4:2 : means use 2 processes in node garray4. 
​					1 for master process(data arrangement, control all child process)
​					1 for child process (computing only, control all hired GPU device)
​					that means, when lanuch the mpirun, at least 2 processes are needed.



### run the demo

After you set all the simulating pars. use `run.sh` to preform the simulation.

use `nohup sh run.sh 2 hillmodel > 1.out 2>&1 &` to run in background, `1.out` is the output log.

use `sh run.sh 2 hillmodel` to directly run.



### visualize the output

Four python programs are used for view the output, and the instructions was written in the program.

`figgrid3d.py` use to display gridmesh.

`figmedia3d.py` use to display medium.

`figseismo3d.py` use to display point and line.

`figsnap3d.py` use to display snapshot.



