#!/bin/bash

# before run
# sh run.sh 3 ( one for master and two for child, distributed by sequence )
#restrict the least processes is two, one for master another for child
# Demos:  nohup sh run.sh 2 hillmodel > 1.out 2>&1 &


#currtDir=$(cd $(dirname $0) && pwd)
currtDir=/shdisk/ary4/yzj/CoPara-CGFD3D/EXAMPLE
binDir=/shdisk/ary4/yzj/CoPara-CGFD3D/bin
nodefile=${currtDir}/$2/machinefile
devfile=${currtDir}/$2/device.dat
confile=${currtDir}/$2/SeisFD3D.conf
numP=$1

#information output
echo Project is $2 under path $currtDir
echo will hire $numP Process, while the 1st is master-CPU
echo
echo Process information:----------
cat $nodefile | grep garray | awk '!/#/'
echo
echo Device information:----------
cat $devfile | grep device | awk '!/#/'
echo


#-----------------execute------------------------------------
mpirun -np $numP -f $nodefile ${binDir}/wavesim $confile




#-------------------nvprof---------------------------------------

#mpirun -np $numP -f $nodefile nvprof ./bin/wavesim $confile

#mpirun -np $numP -f $nodefile nvprof --output-profile timeline.nvprof ./bin/wavesim $confile
#mpirun -np $numP -f $nodefile nvprof --metrics achieved_occupancy,executed_ipc -o metric.nvprof ./bin/wavesim $confile
#mpirun -np $numP -f $nodefile nvprof --events warps_launched,branch -o events1.nvprof ./bin/wavesim $confile
#mpirun -np $numP -f $nodefile nvprof --kernels ":::" --analysis-metrics -o analysis.nvprof ./bin/wavesim $confile

#sh run 4



# convert -delay 100 *.jpg heiheihei.gif






#if [ $# -lt 1 ]; then
#	numP="2"
#else
#	numP="$1"
#fi
#mpirun -np $numP -f ./machinefile ./bin/wavesim SeisFD3D.conf

