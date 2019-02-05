#include "cudaParticleLF.hh"
#include "kernelfuncs.h"

void cudaParticleLF::calcVinit(real dt)
{
  clearForce();
  calcForce(); // calculates F(0) from r(0) first

  // v(-1/2) = v(0) - dt/2 F(0)/m
  calcLFVinit<<<MPnum, THnum1D>>>(v, dt / 2, F, minv, N);
}

void cudaParticleLF::TimeEvolution(real dt)
{
  propagateLeapFrog<<<MPnum, THnum1D>>>(r, dt, v, a, minv, move, N);

  if (withInfo)
    ErrorInfo("cudaParticleLF::TimeEvolution");
}

void cudaParticleLF::rollback(real dt)
{
  rollbackLeapFrog<<<MPnum, THnum1D>>>(r, dt, v, a, minv, move, N);
}
