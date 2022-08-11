
TARGET = testSPH testSPH2 testSPH4 testSPH5 testSPH6 testDEM testDEM2 testDEM3 testDEM4 testDEM5 testDEM5D testDEM6 testDEM6m \
	testMD1 testMD2 testMD3 testMD4 testMD4n calcMD_CO2

CUDA_SRC_FILES = kernelfuncs.cu \
	CUDAenv.cu cudaFrame.cu cudaParticleBase.cu \
	cudaCutoffBlock.cu cudaCalcGR.cu \
	cudaParticleVV.cu cudaSortedPositions.cu cudaParticleMD.cu \
	GaussianThermo.cu \
	cudaParticleRotation.cu cudaParticleDEM.cu \
	cudaParticleSPHBase.cu cudaParticleSPH_NS.cu \
	cudaParticleLF.cu cudaSelectedBlock.cu putTMPselected.cu \
	cudaSparseMat.cu
CUDA_OBJ_FILES = $(CUDA_SRC_FILES:.cu=.o)


SRC_FILES = 
OBJ_FILES = $(CUDA_OBJ_FILES) 


NVCC		= nvcc
NVCCFLAGS	= -O3 -m64 -Xcompiler -fPIC -std=c++14 --expt-extended-lambda
LINK		= nvcc
LDFLAGS		+= -m64
CXXFLAGS0	+= -DREAL=float
#CXXFLAGS0	+= -DREAL=double
NVCCFLAGS	+= -DDOT=cublasSdot
#NVCCFLAGS	+= -DGEMV=cublasSgemv -DDOT=cublasSdot -DAXPY=cublasSaxpy -arch=sm_37
#NVCCFLAGS	+= -DGEMV=cublasDgemv -DDOT=cublasDdot -DAXPY=cublasDaxpy -arch=sm_20 --use_fast_math

# for DEBUG mode (checking error with cudaThreadSync.)
#NVCCFLAGS	+= -DDEBUG
#NVCCFLAGS	+= -DDUMPCOLLISION

# if use OpenMP in .cu
ifneq ($(SYSTEM),CLANG)
NVCCFLAGS	+= -Xcompiler -fopenmp
LDFLAGS		+= -lgomp
endif

CXXFLAGS0	+= $(LOCAL_INCLUDE)

LDFLAGS		+= -lpthread
LDFLAGS		+= -lcublas

LIBS0		+= -lboost_serialization
LIBS		+= -L. -lcudaParticles $(LIBS0)

NVCCFLAGS	+= $(CXXFLAGS0)
CXXFLAGS	+= $(CXXFLAGS0)


##
## make depend using .d
##
#PREFLAGS	+= $(LOCAL_INCLUDE)
SUFFIXES	+= .d .cu

ifndef OBJ_FILES
OBJ_FILES = $(SRC_FILES:.cc=.o) $(C_SRC_FILES:.c=.o) $(CUDA_SRC_FILES:.cu=.o) 
endif

include $(subst .o,.d,$(OBJ_FILES))

#%.d: %.cc
#	$(CXX) -std=c++11 $(PREFLAGS) $(BLAS_TARGET) $< > $@
#%.d: %.c
#	$(CC) $(PREFLAGS) $(BLAS_TARGET) $< > $@
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(LOCAL_INCLUDE) -o $@ -c $<
%.d: %.cu
	$(NVCC) -M $(LOCAL_INCLUDE) $< > $@

##
## targets
##

all: $(TARGET) _cudaParticles.so

clean:
	rm -f $(OBJ_FILES) *~ $(TARGET) $(TARGET:=.o) libcudaParticles.*
	rm -f _cudaParticles.so cudaParticles.py* cudaParticles_wrap.*


libcudaParticles.a: $(CUDA_OBJ_FILES)
	ar -sr $@ $?

libcudaParticles.so: $(CUDA_OBJ_FILES)
	$(NVCC) -o $@ $^ $(LDFLAGS) -shared


testMD1: testMD1.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testMD2: testMD2.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testMD3: testMD3.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testMD4: testMD4.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testMD4n: testMD4n.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

calcMD_CO2: calcMD_CO2.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

calcgr1: calcgr1.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testSPH: testSPH.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testSPH2: testSPH2.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testSPH3: testSPH3.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testSPH4: testSPH4.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testSPH5: testSPH5.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testSPH6: testSPH6.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testDEM: testDEM.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testDEM2: testDEM2.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testDEM3: testDEM3.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testDEM4: testDEM4.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testDEM5: testDEM5.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testDEM5D: testDEM5D.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testDEM6: testDEM6.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)

testDEM6m: testDEM6m.o libcudaParticles.so
	$(LINK) $(LDFLAGS) -o $@ $< $(LIBS)
testDEM6m.o: testDEM6.cu
	$(NVCC) $(NVCCFLAGS) $(LOCAL_INCLUDE) -DUSEMULTIPARTICLEBLOCK -o $@ -c $<

##
## for the SWIG interface
##
_cudaParticles.so: cudaParticles_wrap.o $(OBJ_FILES)
	$(NVCC) $(LDFLAGS) -shared -o $@ $^ $(LIBS0) -lpython2.7

cudaParticles_wrap.o: cudaParticles_wrap.cu
	$(NVCC) $(NVCCFLAGS) -I/usr/include/python2.7/ -c $^ -o $@

cudaParticles_wrap.cu: cudaParticles.i 
	swig -c++ -python $(CXXFLAGS0) -o $@ cudaParticles.i

