CXX := g++
CCU := nvcc
MPI := mpicxx

SRCDIR := ./src
OBJDIR := ./obj
BINDIR := ./bin

DisBug := 
HYindex := ON
TypeDouble := 
SrcSmooth := 
withABS := ON
CFSPML := ON
CondFree := ON
CondFreeVLOW := ON
CondFreeVUCD :=
CondFreeTIMG := ON
MPI_DEBUG := 
PointOnly := 
ZWmedia := ON   
DevicePick :=


DFLAG_LIST = DisBug TypeDouble SrcSmooth \
             CondFreeVLOW CondFreeTIMG \
	     MPI_DEBUG CondFreeVUCD CondFree \
	     withABS CFSPML PointOnly ZWmedia \
	     HYindex DevicePick

NCLIBS := -L/public/software/netcdf-gnu/lib -lnetcdf
NCINC := -I/public/software/netcdf-gnu/include
CUDALIBS := -L/usr/local/cuda/lib64 -lcudart -lcudadevrt
CUDAINC := -I/usr/local/cuda/include
CUDAHELPERINC := -I/usr/local/cuda/samples/common/inc
MPILIBS := -L /public/software/mpich-gnu/lib -lmpi
MPIINC := -I /public/software/mpich-gnu/include

LIBFLAGS := $(NCLIBS) $(CUDALIBS) $(MPILIBS)
INCFLAGS := -I$(SRCDIR) $(NCINC) $(MPIINC)

CUDAARC := -arch=sm_60
# cuda link using -rdc=true -dlink, cuda compile -dc
CXXFLAGS := $(INCFLAGS) -std=c++11
CCUFLAGS := $(INCFLAGS) $(CUDAINC) $(CUDAHELPERINC) $(CUDAARC)
MPIFLAGS := $(INCFLAGS) $(CUDAINC) -std=c++11

DFLAGS := $(foreach flag,$(DFLAG_LIST), $(if $($(flag)),-D$(flag),)) $(DFLAGS)
DFLAGS := $(strip $(DFLAGS))

#CESHI := -g

SRC_CXX := common.cpp mathfunc.cpp seisnc.cpp gridmesh.cpp mediaparas.cpp source.cpp \
		seisplot.cpp ceshi.cpp absorb.cpp
SRC_CCU := calculate.cu
SRC_MPI := datadis.cc
SRC_WAVE := simulation.cc
EXE_WAVE := wavesim
CUDA_LINK := cudalink.o

#SRC_CXX := common.cpp mathfunc.cpp gridmesh.cpp mediapara.cpp source.cpp \
#            seisplot.cpp abspml.cpp seisnc.cpp
#SRC_CCU := propker.cu
#SRC_WAVE := wavemain.cu
#EXE_WAVE := wave

OBJ_CXX := $(foreach file, $(SRC_CXX), $(OBJDIR)/cxx/$(file:.cpp=.o))
OBJ_CCU := $(foreach file, $(SRC_CCU), $(OBJDIR)/cuda/$(file:.cu=.o))
OBJ_MPI := $(foreach file, $(SRC_MPI), $(OBJDIR)/mpi/$(file:.cc=.o))
OBJ_WAVE := $(OBJDIR)/mpi/$(SRC_WAVE:.cc=.o)

all: prep link

prep:
	@mkdir -p $(BINDIR) $(OBJDIR)
	@mkdir -p $(OBJDIR)/cxx $(OBJDIR)/cuda $(OBJDIR)/mpi

link: $(BINDIR)/$(EXE_WAVE)

$(OBJDIR)/cuda/$(CUDA_LINK) : $(OBJ_CCU)
	$(CCU) $(CUDAARC) -rdc=true -dlink -maxrregcount=64 $< -o $@
#$(OBJDIR)/cuda/$(CUDA_LINK) : $(OBJ_CCU)
#	$(CCU) $(CUDAARC) -rdc=true -dlink $< -o $@

$(BINDIR)/$(EXE_WAVE): $(OBJ_CXX) $(OBJ_CCU) $(OBJDIR)/cuda/$(CUDA_LINK) $(OBJ_MPI) $(OBJ_WAVE)
	$(CXX) $^ $(LIBFLAGS) -o $@

clear: clean
	-rm -f $(BINDIR)/*

clean:
	-rm -f $(OBJDIR)/cxx/*.o
	-rm -f $(OBJDIR)/cuda/*.o
	-rm -f $(OBJDIR)/mpi/*.o
	-rm -f $(BINDIR)/$(EXE_WAVE)

$(OBJDIR)/cxx/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(DFLAGS) $(CXXFLAGS) $(CESHI) -c $< -o $@

$(OBJDIR)/cuda/%.o : $(SRCDIR)/%.cu
	$(CCU) $(DFLAGS) $(CCUFLAGS) $(CESHI) -G -dc $< -o $@

$(OBJDIR)/mpi/%.o : $(SRCDIR)/%.cc
	$(MPI) $(DFLAGS) $(MPIFLAGS) $(CESHI) -c $< -o $@

#vim:ft=make:ts=4:sw=4:nu:et:at:noet
