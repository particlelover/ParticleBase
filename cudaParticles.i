%module cudaParticles
%include "carrays.i"
%include "std_string.i"
%include "std_vector.i"
%include "carrays.i"
%include "stdint.i"


%{
#include "CUDAenv.hh"
#include "cudaFrame.hh"
#include "cudaParticleBase.hh"
#include "putTMPselected.hh"
#include "cudaCutoffBlock.hh"
#include "cudaSelectedBlock.hh"
#include "cudaParticleLF.hh"
#include "cudaParticleRotation.hh"
#include "cudaParticleSPHBase.hh"
#include "cudaParticleSPH_NS.hh"
#include "cudaSparseMat.hh"
#include "cudaParticleDEM.hh"
#include "ParticleBase.hh"
#include "AdaptiveTime.hh"
%}

%include "cudaFrame.hh"
%include "cudaParticleBase.hh"
%include "putTMPselected.hh"
%include "cudaCutoffBlock.hh"
%include "cudaSelectedBlock.hh"
%include "cudaParticleLF.hh"
%include "cudaParticleRotation.hh"
%include "cudaParticleSPHBase.hh"
%include "cudaParticleSPH_NS.hh"
%include "cudaSparseMat.hh"
%include "cudaParticleDEM.hh"
%include "ParticleBase.hh"
%include "AdaptiveTime.hh"

%include "CUDAenv.hh"
%template (CudaParticleSPH_NS) CUDAenv<cudaParticleSPH_NS>;
%template (CudaParticleDEM) CUDAenv<cudaParticleDEM>;

%template (AdaptiveTimeDEM) AdaptiveTime<CUDAenv<cudaParticleDEM> >;

%template (globalTable) std::vector<ParticleBase>;

%array_functions(real, realArray)
%template (VectorInt) std::vector<int>;

%extend CUDAenv<cudaParticleSPH_NS> {
    cudaParticleSPH_NS & __getitem__(int i) {
            return (self->operator[](i));
    }
};
%extend CUDAenv<cudaParticleDEM> {
    cudaParticleDEM & __getitem__(int i) {
            return (self->operator[](i));
    }
};
