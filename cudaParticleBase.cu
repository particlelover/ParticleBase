#include "cudaParticleBase.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"

#include <iostream>

cudaParticleBase::~cudaParticleBase()
{
  if (r != NULL)
    cudaFree(r);
  if (minv != NULL)
    cudaFree(minv);
  if (v != NULL)
    cudaFree(v);
  if (a != NULL)
    cudaFree(a);
  if (F != NULL)
    cudaFree(F);
  if (tmp3N != NULL)
    cudaFree(tmp3N);
  if (typeID != NULL)
    cudaFree(typeID);
  if (move != NULL)
    cudaFree(move);
}

void cudaParticleBase::setup(int n)
{

  N = n;
  TMP.resize(n * 3);

  TMP2.resize(n);

  // alloc x, y, z
  cudaMalloc((void **)&r, sizeof(real) * 3 * N);
  if (withInfo)
    ErrorInfo("malloc r[] on GPU");

  // alloc mass inv
  cudaMalloc((void **)&minv, sizeof(real) * N);
  if (withInfo)
    ErrorInfo("malloc minv[] on GPU");

  // alloc vx, vy, vz
  cudaMalloc((void **)&v, sizeof(real) * 3 * N);
  if (withInfo)
    ErrorInfo("malloc v[] on GPU");

  // alloc ax, ay, az
  cudaMalloc((void **)&a, sizeof(real) * 3 * N);
  if (withInfo)
    ErrorInfo("malloc a[] on GPU");

  // alloc Fx, Fy, Fz
  cudaMalloc((void **)&F, sizeof(real) * 3 * N);
  if (withInfo)
    ErrorInfo("malloc F[] on GPU");

  // tmp3N
  cudaMalloc((void **)&tmp3N, sizeof(real) * 3 * N);
  if (withInfo)
    ErrorInfo("malloc tmp3N[] on GPU");

  // alloc typeID
  cudaMalloc((void **)&typeID, sizeof(unsigned short int) * N);
  if (withInfo)
    ErrorInfo("malloc typeID[] on GPU");

  // alloc move flag
  cudaMalloc((void **)&move, sizeof(char) * N);
  if (withInfo)
    ErrorInfo("malloc move[] on GPU");

  //  threadsMax = getNumTh();
}

void cudaParticleBase::import(const std::vector<ParticleBase> &P)
{
  const size_t sizeN = sizeof(real) * N;
  const size_t sizeN3 = sizeN * 3;
  const size_t N2 = N * 2;

#pragma omp parallel for
  for (int i = 0; i < N; ++i)
  {
    TMP[i] = P[i].r[0];
    TMP[i + N] = P[i].r[1];
    TMP[i + N2] = P[i].r[2];
  }
  cudaMemcpy(r, &(TMP[0]), sizeN3, cudaMemcpyHostToDevice);
  if (withInfo)
    ErrorInfo("copy r[] to GPU");

#pragma omp parallel for
  for (int i = 0; i < N; ++i)
  {
    TMP[i] = P[i].v[0];
    TMP[i + N] = P[i].v[1];
    TMP[i + N2] = P[i].v[2];
  }
  cudaMemcpy(v, &(TMP[0]), sizeN3, cudaMemcpyHostToDevice);
  if (withInfo)
    ErrorInfo("copy v[] to GPU");

#pragma omp parallel for
  for (int i = 0; i < N; ++i)
  {
    TMP[i] = P[i].a[0];
    TMP[i + N] = P[i].a[1];
    TMP[i + N2] = P[i].a[2];
  }
  cudaMemcpy(a, &(TMP[0]), sizeN3, cudaMemcpyHostToDevice);
  if (withInfo)
    ErrorInfo("copy a[] to GPU");

  for (size_t i = 0; i < N; ++i)
    TMP[i] = 1.0 / P[i].m;
  cudaMemcpy(minv, &(TMP[0]), sizeN, cudaMemcpyHostToDevice);
  if (withInfo)
    ErrorInfo("copy minv[] to GPU");

  std::valarray<unsigned short int> _T(N);
  for (size_t i = 0; i < N; ++i)
    _T[i] = P[i].type;
  cudaMemcpy(typeID, &(_T[0]), sizeof(unsigned short int) * N, cudaMemcpyHostToDevice);
  if (withInfo)
    ErrorInfo("copy type[] to GPU");

  std::valarray<char> _T2(N);
  for (size_t i = 0; i < N; ++i)
    _T2[i] = (P[i].isFixed) ? 0 : 1;
  cudaMemcpy(move, &(_T2[0]), sizeof(char) * N, cudaMemcpyHostToDevice);
  if (withInfo)
    ErrorInfo("copy type[] to GPU");
}

void cudaParticleBase::getTypeID(void)
{
  cudaMemcpy(&(TMP2[0]), typeID, sizeof(unsigned short) * N, cudaMemcpyDeviceToHost);
  if (withInfo)
    ErrorInfo("do getTypeID");
}
void cudaParticleBase::getPosition(void)
{
  size_t sizeN = sizeof(real) * N * 3;

  pthread_mutex_lock(&mutTMP);
  cudaMemcpy(&(TMP[0]), r, sizeN, cudaMemcpyDeviceToHost);
  pthread_mutex_unlock(&mutTMP);

  if (withInfo)
    ErrorInfo("do getPosition");
}
void cudaParticleBase::getAcceleration(void)
{
  size_t sizeN = sizeof(real) * N * 3;

  pthread_mutex_lock(&mutTMP);
  cudaMemcpy(&(TMP[0]), a, sizeN, cudaMemcpyDeviceToHost);
  pthread_mutex_unlock(&mutTMP);

  if (withInfo)
    ErrorInfo("do getAcceleration");
}
void cudaParticleBase::getForce(void)
{
  size_t sizeN = sizeof(real) * N * 3;

  pthread_mutex_lock(&mutTMP);
  cudaMemcpy(&(TMP[0]), F, sizeN, cudaMemcpyDeviceToHost);
  pthread_mutex_unlock(&mutTMP);

  if (withInfo)
    ErrorInfo("do getForce");
}

void cudaParticleBase::TimeEvolution(real dt)
{
  propagateEuler<<<MPnum, THnum1D>>>(r, dt, v, a, move, N);

  if (withInfo)
    ErrorInfo("do TimeEvolution");
}

void cudaParticleBase::clearForce(void)
{
  clearArray<<<MPnum, THnum1D>>>(F, N * 3);
  if (withInfo)
    ErrorInfo("clearForce");
}

void cudaParticleBase::clearTmp3N(void)
{
  clearArray<<<MPnum, THnum1D>>>(tmp3N, N * 3);
  if (withInfo)
    ErrorInfo("clearTmp3N");
}

void cudaParticleBase::calcAcceleration(void)
{
  calcA<<<MPnum, THnum1D>>>(a, minv, F, N);
  if (withInfo)
    ErrorInfo("calcAcceleration");
}

void cudaParticleBase::treatPeriodicCondition(void)
{
  applyPeriodicCondition<<<MPnum, THnum1D>>>(r, cell[0], cell[1], N);
  applyPeriodicCondition<<<MPnum, THnum1D>>>(&(r[N]), cell[2], cell[3], N);
  applyPeriodicCondition<<<MPnum, THnum1D>>>(&(r[N * 2]), cell[4], cell[5], N);
}

void cudaParticleBase::treatAbsoluteCondition(void)
{
  treatAbsoluteBoundary<<<MPnum, THnum1D>>>(r, cell[0], cell[1], N);
  treatAbsoluteBoundary<<<MPnum, THnum1D>>>(&(r[N]), cell[2], cell[3], N);
  treatAbsoluteBoundary<<<MPnum, THnum1D>>>(&(r[N * 2]), cell[4], cell[5], N);
}

void cudaParticleBase::addForceX(real fx)
{
  addArray<<<MPnum, THnum1D>>>(F, fx, N);
  if (withInfo)
    ErrorInfo("addForce in X");
}

void cudaParticleBase::addForceY(real fy)
{
  addArray<<<MPnum, THnum1D>>>(&(F[N]), fy, N);
  if (withInfo)
    ErrorInfo("addForce in Y");
}

void cudaParticleBase::addForceZ(real fz)
{
  addArray<<<MPnum, THnum1D>>>(&(F[N * 2]), fz, N);
  if (withInfo)
    ErrorInfo("addForce in Z");
}

void cudaParticleBase::addAccelerationX(real ax)
{
  addArray<<<MPnum, THnum1D>>>(a, ax, N);
  if (withInfo)
    ErrorInfo("addAcceleration in X");
}

void cudaParticleBase::addAccelerationY(real ay)
{
  addArray<<<MPnum, THnum1D>>>(&(a[N]), ay, N);
  if (withInfo)
    ErrorInfo("addAcceleration in Y");
}

void cudaParticleBase::addAccelerationZ(real az)
{
  addArray<<<MPnum, THnum1D>>>(&(a[N * 2]), az, N);
  if (withInfo)
    ErrorInfo("addAcceleration in Z");
}

int cudaParticleBase::inspectVelocity(real vlim, real lim_u, real lim_l, real &_r, bool debug)
{
  if (debug)
    clearArray<<<MPnum, THnum1D>>>(tmp3N, N + 2);
  else
    clearArray<<<MPnum, THnum1D>>>(&(tmp3N[N]), 2);

  inspectV<<<MPnum, THnum1D>>>(v, N, vlim, tmp3N, lim_u, lim_l, debug);

  pthread_mutex_lock(&(mutTMP));

  if (debug)
    cudaMemcpy(&(TMP[0]), &(tmp3N[0]), sizeof(real) * (N + 2), cudaMemcpyDeviceToHost);
  else
    cudaMemcpy(&(TMP[N]), &(tmp3N[N]), sizeof(real) * 2, cudaMemcpyDeviceToHost);
  /*
  std::cerr << TMP[N+0] << TMP[N+1] << TMP[N+2] << TMP[N+3]
	    << TMP[N+4] << TMP[N+5] << TMP[N+6] << TMP[N+7] << std::endl;
*/

  // enhancement; add to debug out
  if (debug)
  {
    real r1 = *std::max_element(&TMP[0], &TMP[N]); // largest *ratio*
    _r = r1;
    // std::cerr << v1 << " " << vmax << " " << v1 / vmax << std::flush;
    // _v = v1; => _v * deltaT
    // important field is time, delta t, and the *ratio*
  }
  if (TMP[N + 0] > 0.0)
  {
    pthread_mutex_unlock(&(mutTMP));
    return 1;
  }
  else if (TMP[N + 1] > 0.0)
  {
    pthread_mutex_unlock(&(mutTMP));
    return 0;
  }
  else
  {
    pthread_mutex_unlock(&(mutTMP));
    return -1;
  }
}

void cudaParticleBase::dump3Narray(real *A, std::ostream &o)
{
  pthread_mutex_lock(&mutTMP);
  cudaMemcpy(&(TMP[0]), A, sizeof(real) * N * 3, cudaMemcpyDeviceToHost);
  o << std::endl
    << std::endl;
  for (int i = 0; i < N; ++i)
    o << i << "\t" << TMP[i]
      << "\t" << TMP[i + N]
      << "\t" << TMP[i + N * 2] << std::endl;
  pthread_mutex_unlock(&mutTMP);
}
