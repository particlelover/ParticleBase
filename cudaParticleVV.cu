#include "cudaParticleVV.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"

cudaParticleVV::~cudaParticleVV()
{
  if (Fold != NULL)
    cudaFree(Fold);
}

void cudaParticleVV::setup(int n)
{
  cudaParticleBase::setup(n);

  // alloc Fx, Fy, Fz
  cudaMalloc((void **)&Fold, sizeof(real) * 3 * N);
  if (withInfo)
    ErrorInfo("malloc Fold[] on GPU");
}

void cudaParticleVV::clearForce(void)
{
  cudaParticleBase::clearForce();

  clearArray<<<MPnum, THnum1D>>>(Fold, N * 3);
  if (withInfo)
    ErrorInfo("cudaParticleVV::clearForce");
}

void cudaParticleVV::TimeEvolution(real dt)
{
  /**
   *
   * full scheme of Velocity Verlet at time t in our way is
   * 
   * + SHAKE constraints
   * + calc F(t) from r(t)
   * + calc v(t) from v(t-dt/2), F(t)
   * + RATTLE constraints
   * + calc v(t+dt/2) from v(t), F(t)
   * + calc r(t+dt) from v(t+dt/2)
   *
   * in another style, itertion loop starts from calc v(t+dt/2)
   */
  propagateVelocityVerlet<<<MPnum, THnum1D>>>(r, dt, v, F, Fold, minv, N);

  if (withInfo)
    ErrorInfo("cudaParticleVV::TimeEvolution");
}
