#include "cudaParticleLF.hh"
#include "kernelfuncs.h"

void cudaParticleLF::calcVinit(real dt) {
  clearForce();
  calcForce();  // calculates F(0) from r(0) first

  if (omitlist.count(ArrayType::force)==0) {
    // v(-1/2) = v(0) - dt/2 F(0)/m
    calcLFVinit_F4<<<MPnum, THnum1D>>>(v, dt/2, F, minv, N);
  }
}

void cudaParticleLF::TimeEvolution(real dt) {
  propagateLeapFrog_F4<<<MPnum, THnum1D>>>(r, dt, v, a, minv, move, N);

  if (withInfo) ErrorInfo("cudaParticleLF::TimeEvolution");
}

void cudaParticleLF::rollback(real dt) {
  rollbackLeapFrog_F4<<<MPnum, THnum1D>>>(r, dt, v, a, minv, move, N);
}
