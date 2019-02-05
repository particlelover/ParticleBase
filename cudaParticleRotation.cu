#include "cudaParticleRotation.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"

#include <iostream>

cudaParticleRotation::~cudaParticleRotation()
{
  if (w != NULL)
    cudaFree(w);
  if (m0inv != NULL)
    cudaFree(m0inv);
  if (L != NULL)
    cudaFree(L);
  if (T != NULL)
    cudaFree(T);
}

void cudaParticleRotation::setup(int n)
{

  cudaMalloc((void **)&w, sizeof(real) * 3 * N);
  if (withInfo)
    ErrorInfo("malloc w[] on GPU");

  cudaMalloc((void **)&m0inv, sizeof(real) * N);
  if (withInfo)
    ErrorInfo("malloc m0inv[] on GPU");

  cudaMalloc((void **)&L, sizeof(real) * 3 * N);
  if (withInfo)
    ErrorInfo("malloc L[] on GPU");

  cudaMalloc((void **)&T, sizeof(real) * 3 * N);
  if (withInfo)
    ErrorInfo("malloc T[] on GPU");
}

void cudaParticleRotation::TimeEvolution(real dt)
{
  propagateEulerianRotation<<<MPnum, THnum1D>>>(w, dt, L, T, m0inv, move, N);

  if (withInfo)
    ErrorInfo("do TimeEvolution Eulerian Equation of Motion");
}

void cudaParticleRotation::setInertia(real r0_all)
{
  std::vector<real> r0(N, r0_all);
  setInertia(r0);
}

void cudaParticleRotation::setInertia(const std::vector<real> &_r0)
{
  clearArray<<<MPnum, THnum1D>>>(w, N * 3);
  clearArray<<<MPnum, THnum1D>>>(L, N * 3);

  cudaMemcpy(m0inv, minv, sizeof(real) * N, cudaMemcpyDeviceToDevice);

  std::vector<real> r1(N);
  for (int i = 0; i < N; ++i)
    r1[i] = 1 / (0.4 * _r0[i] * _r0[i]);

  cudaMemcpy(tmp3N, &(r1[0]), sizeof(real) * N, cudaMemcpyHostToDevice);
  multiplies<<<MPnum, THnum1D>>>(m0inv, tmp3N, N);

  if (withInfo)
    ErrorInfo("set Initial Inertia");
}
