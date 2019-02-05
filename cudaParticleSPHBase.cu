#include <assert.h>
#include "cudaParticleSPHBase.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"

cudaParticleSPHBase::~cudaParticleSPHBase()
{
  cudaFree(num);
  //  cudaFree(rho);
  //cudaFree(one);
  cudaFree(m);
  cudaFree(rhoinv);
  //  cudaFree(w2D);
  //  cudaFree(dW2D);

  cublasDestroy(hdl);
}

void cudaParticleSPHBase::setup(int n)
{
  cudaParticleLF::setup(n);

  // alloc n (number density)
  cudaMalloc((void **)&num, sizeof(real) * N * 3);
  if (withInfo)
    ErrorInfo("malloc n[] on GPU");

  // alloc rho
  rho = &(num[N]);
  //  cudaMalloc((void **)&rho, sizeof(real)*N);
  //  if (withInfo) ErrorInfo("malloc rho[] on GPU");

  // alloc one
  //  cudaMalloc((void **)&one, sizeof(real)*N);
  //  if (withInfo) ErrorInfo("malloc one[] on GPU");

  cudaMalloc((void **)&m, sizeof(real) * N);
  if (withInfo)
    ErrorInfo("malloc m[] on GPU");

  //  setArray<<<4, 256>>>(one, 1.0, N);
  //  if (withInfo) ErrorInfo("setArray by one");

  // alloc rhoinv
  cudaMalloc((void **)&rhoinv, sizeof(real) * N);
  if (withInfo)
    ErrorInfo("malloc rhoinv[] on GPU");

  // alloc kernel W
  //  cudaMalloc((void **)&w2D, sizeof(real)*N*N);
  //  if (withInfo) ErrorInfo("malloc W[] on GPU");

  // alloc kernel dW/dr
  //  cudaMalloc((void **)&dW2D, sizeof(real)*N*N);
  //  if (withInfo) ErrorInfo("malloc dW/dr[] on GPU");

  // preparation for CUBLAS
  cublasCreate(&hdl);
}

void cudaParticleSPHBase::setM(void)
{
  calcReciproc<<<MPnum, THnum1D>>>(minv, m, N);
  if (withInfo)
    ErrorInfo("calc reciprocal of minv to m");
}

void cudaParticleSPHBase::calcKernels(void)
{
#if !defined(CUDACUTOFFBLOCK)
  clearArray<<<MPnum, THnum1D>>>(w2D, N * N);
  clearArray<<<MPnum, THnum1D>>>(dW2D, N * N);

  dim3 _mpnum, _thnum;
  _thnum.x = THnum2D;
  _thnum.y = THnum2D;
  _thnum.z = 1;
  // thnum * thnum limitted by GPU's `threads per block'
  assert(_thnum.x * _thnum.y <= threadsMax);

  _mpnum.x = N / _thnum.x;
  _mpnum.y = N / _thnum.y;
  _mpnum.z = 1;
  if ((_mpnum.x * _thnum.x) < N)
    ++_mpnum.x;
  if ((_mpnum.y * _thnum.y) < N)
    ++_mpnum.y;
  //  std::cerr << _mpnum.x << "x" << _mpnum.y << " block with "
  //	    << _thnum.x << "x" << _thnum.y << " threads" << std::endl;

  calcSPHKernel<class SPHKernelLucy, class SPHKernelLucyDW><<<_mpnum, _thnum>>>(r, w2D, dW2D, N, h, w0, -12 * w0);
  if (withInfo)
    ErrorInfo("calc Kernel for SPH");
#endif
}

void cudaParticleSPHBase::calcDensity(void)
{
  clearArray<<<MPnum, THnum1D>>>(num, N);
  clearArray<<<MPnum, THnum1D>>>(rho, N);
  clearArray<<<MPnum, THnum1D>>>(rhoinv, N);
  if (withInfo)
    ErrorInfo("clear Array num, rho, rhoinv");

#if defined(CUDACUTOFFBLOCK)
  dim3 _mpnum, _thnum;
  //_thnum.x = THnum2D; _thnum.y = THnum2D; _thnum.z = 1;
  _thnum.x = THnum2D2;
  _thnum.y = 1;
  _thnum.z = 1;
  assert(_thnum.x * _thnum.y <= threadsMax);
  _mpnum.x = totalNumBlock;
  _mpnum.y = 27;
  _mpnum.z = 1;
  clearArray<<<MPnum, THnum1D>>>(tmp81N, N * 81);
  class SPHcalcDensity<SPHKernelLucy> P;
  P.m = m;
  P.h = h;
  P.w0 = w0;
  P.opt = NULL;
  calcF_IJpairWithBlock<SPHcalcDensity<SPHKernelLucy>><<<_mpnum, _thnum>>>(P, r,
                                                                           tmp81N,
                                                                           0,
                                                                           blockNeighbor, pid, bindex,
                                                                           N);
  if (withInfo)
    ErrorInfo("calc Density with Cutoff Block (SPHBase)");

  reduce27<<<MPnum, THnum1D>>>(rho, tmp81N, N, N * 3, 0);
  reduce27<<<MPnum, THnum1D>>>(num, tmp81N, N, N * 3, N);
#else
  const real alpha = 1.0;
  const real beta = 0.0;
  // calculate n = [W](1), rho = [W](m)
  GEMV(hdl, CUBLAS_OP_N, N, N, &alpha, w2D, N, one, 1, &beta, num, 1);
  GEMV(hdl, CUBLAS_OP_N, N, N, &alpha, w2D, N, m, 1, &beta, rho, 1);
  if (withInfo)
    ErrorInfo("GEMV by CUBLAS");
#endif

  /*
  cudaMemcpy(&(TMP[0]),   num, sizeof(real)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(&(TMP[N]),   rho, sizeof(real)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(&(TMP[N*2]), rhoinv, sizeof(real)*N, cudaMemcpyDeviceToHost);
  putTMP(std::cerr);
  if (withInfo) ErrorInfo("cudaMemCpy");
*/

  // rhoinv = 1.0/rho
  calcReciproc<<<MPnum, THnum1D>>>(rho, rhoinv, N);
  if (withInfo)
    ErrorInfo("calc reciprocal of rho for SPH");
}

void cudaParticleSPHBase::setSPHProperties(real _h)
{
  h = _h;
  w0 = 105 / (16 * M_PI * h * h * h * h * h * h * h);

  calcReciproc<<<MPnum, THnum1D>>>(minv, m, N);
}
