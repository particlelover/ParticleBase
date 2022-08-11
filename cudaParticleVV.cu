#include "cudaParticleVV.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"

void cudaParticleVV::setup(int n) {
  cudaParticleBase::setup(n);
  Fold = a;
}

void cudaParticleVV::clearForce(void) {
  cudaParticleBase::clearForce();

  clearArray_F4<<<MPnum, THnum1D>>>(Fold, N);
  if (withInfo) ErrorInfo("cudaParticleVV::clearForce");
}

void cudaParticleVV::TimeEvolution(real dt) {
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
  propagateVelocityVerlet_F4<<<MPnum, THnum1D>>>(r, dt, v, F, Fold, minv, N);

  if (withInfo) ErrorInfo("cudaParticleVV::TimeEvolution");
}
