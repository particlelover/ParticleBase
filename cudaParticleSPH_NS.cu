#include "cudaParticleSPH_NS.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"
#include <assert.h>
#include <boost/lexical_cast.hpp>

cudaParticleSPH_NS::~cudaParticleSPH_NS()
{
  if (mu0 != NULL)
    cudaFree(mu0);
  if (c2 != NULL)
    cudaFree(c2);
  //  if (mu!=NULL)  cudaFree(mu);
}

void cudaParticleSPH_NS::setup(int n)
{
  cudaParticleSPHBase::setup(n);

  // for output x, y, z, num, rho, mu
  TMP.resize(n * 6);

  // alloc mu0
  cudaMalloc((void **)&mu0, sizeof(real) * N);
  cudaMalloc((void **)&c2, sizeof(real) * N);
  //  cudaMalloc((void **)&mu, sizeof(real)*N);
  mu = &(num[N * 2]);
  if (withInfo)
    ErrorInfo("malloc mu0[], c2[], mu[] on GPU");

  clearArray<<<4, 256>>>(mu0, N, 1.0);
}

void cudaParticleSPH_NS::getPosition(void)
{
  const size_t sizeN = sizeof(real) * N;
  const size_t sizeN3 = sizeN * 3;

  pthread_mutex_lock(&mutTMP);
  cudaMemcpy(&(TMP[0]), r, sizeN3, cudaMemcpyDeviceToHost);
  cudaMemcpy(&(TMP[N * 3]), num, sizeN, cudaMemcpyDeviceToHost);
  cudaMemcpy(&(TMP[N * 4]), rho, sizeN, cudaMemcpyDeviceToHost);
  cudaMemcpy(&(TMP[N * 5]), mu, sizeN, cudaMemcpyDeviceToHost);
  pthread_mutex_unlock(&mutTMP);

  if (withInfo)
    ErrorInfo("cudaParticleSPH_NS::getPosition");
}

void cudaParticleSPH_NS::setSPHProperties(const std::valarray<real> &_mu0, std::valarray<real> _c1, real _h)
{
  cudaParticleSPHBase::setSPHProperties(_h);

  const size_t sizeN = sizeof(real) * N;

  cudaMemcpy(mu0, &(_mu0[0]), sizeN, cudaMemcpyHostToDevice);

  // copy (sound velocity)^2
  _c1 *= _c1;
  cudaMemcpy(c2, &(_c1[0]), sizeN, cudaMemcpyHostToDevice);
}

std::string cudaParticleSPH_NS::additionalOutput(uint32_t i) const
{
  std::string r = " " + boost::lexical_cast<std::string>(TMP[i + N * 3]) +
                  " " + boost::lexical_cast<std::string>(TMP[i + N * 4]) +
                  " " + boost::lexical_cast<std::string>(TMP[i + N * 5]);
  return r;
}

void cudaParticleSPH_NS::calcAcceleration(bool sortedOutput)
{
  //  real n_threshold = inspectDensity()*0.96;

  // at first clear a[]
  clearArray<<<MPnum, THnum1D>>>(a, N * 3);

#if defined(CUDACUTOFFBLOCK)
  dim3 _mpnum, _thnum;
  //_thnum.x = THnum2D; _thnum.y = THnum2D; _thnum.z = 1;
  _thnum.x = 8;
  _thnum.y = 9;
  _thnum.z = 1;
  assert(_thnum.x * _thnum.y <= threadsMax);
  _mpnum.x = myBlockNum;
  _mpnum.y = 3;
  _mpnum.z = 1;
  clearArray<<<MPnum, THnum1D>>>(tmp81N, N * 81);
  class SPHNavierStokes<SPHKernelLucyDW> P;
  P.v = v;
  P.m = m;
  P.rhoinv = rhoinv;
  P.mu = mu;
  P.c2 = c2;
  P.h = h;
  P.w1 = -12 * w0;
  P.rho0 = 1.0;

  int validOffset = 0;
  uint32_t *Q = NULL;
  uint32_t blocks2calc = myBlockNum;
  if (myBlockSelected > 0)
  {
    Q = selectedBlock;
    blocks2calc = myBlockSelected;
    validOffset = totalNumBlock - numSelected;
  }
  const int dup = static_cast<int>(
      ceil(static_cast<real>(blocks2calc) / maxGrid));
  //  if (dup>1) std::cerr << "DUP" << dup << std::endl;
  for (int _i = 0; _i < dup; ++_i)
  {
    _mpnum.x = (blocks2calc > (_i + 1) * maxGrid) ? maxGrid : blocks2calc % maxGrid;
    //  std::cerr << "myBlock: " << _i << " " << _mpnum.x << std::endl;
    calcF_IJpairWithBlock<class SPHNavierStokes<SPHKernelLucyDW>><<<_mpnum, _thnum>>>(P, r,
                                                                                      tmp81N,
                                                                                      validOffset + myOffsetSelected + maxGrid * _i,
                                                                                      blockNeighbor, pid, bindex,
                                                                                      N,
                                                                                      NULL,        // torque
                                                                                      Q,           // selected block
                                                                                      false,       // sorted r[] and typeid[]
                                                                                      sortedOutput // calcluated acceleration is sorted
    );
  }

  reduce27<<<MPnum, THnum1D>>>(a, tmp81N, N * 3);
#else
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

  //  std::cerr << "calculating cuda kernel with " << _mpnum.x << ":" << _mpnum.y << " blocks" << std::endl;
  calcF_SPH_NS<<<_mpnum, _thnum>>>(r, a, typeID, N, dW2D, rhoinv, m, mu0, v, 1.5e4);
#endif
  if (withInfo)
    ErrorInfo("calc acceleration for SPH_NS on GPU");
}

void cudaParticleSPH_NS::calcDensity(bool sortedOutput)
{
  clearArray<<<MPnum, THnum1D>>>(num, N);
  clearArray<<<MPnum, THnum1D>>>(rho, N);
  clearArray<<<MPnum, THnum1D>>>(rhoinv, N);
  clearArray<<<MPnum, THnum1D>>>(mu, N);
  if (withInfo)
    ErrorInfo("clear Array num, rho, rhoinv");

#if defined(CUDACUTOFFBLOCK)
  const uint32_t calcBlock = (sortedOutput) ? myBlockNum : totalNumBlock;
  dim3 _mpnum, _thnum;
  //_thnum.x = THnum2D; _thnum.y = THnum2D; _thnum.z = 1;
  _thnum.x = 8;
  _thnum.y = 9;
  _thnum.z = 1;
  assert(_thnum.x * _thnum.y <= threadsMax);
  _mpnum.x = calcBlock;
  _mpnum.y = 3;
  _mpnum.z = 1;
  clearArray<<<MPnum, THnum1D>>>(tmp81N, N * 81);
  class SPHcalcDensity<SPHKernelLucy> P;
  P.m = m;
  P.h = h;
  P.w0 = w0;
  P.opt = mu0;

  const int dup = static_cast<int>(
      ceil(static_cast<real>(calcBlock) / maxGrid));
  for (int _i = 0; _i < dup; ++_i)
  {
    _mpnum.x = (calcBlock > (_i + 1) * maxGrid) ? maxGrid : calcBlock % maxGrid;
    //  std::cerr << "myBlock: " << _i << " " << _mpnum.x << std::endl;
    calcF_IJpairWithBlock<SPHcalcDensity<SPHKernelLucy>><<<_mpnum, _thnum>>>(P, r,
                                                                             tmp81N,
                                                                             ((sortedOutput) ? myBlockOffset : 0) + maxGrid * _i,
                                                                             blockNeighbor, pid, bindex,
                                                                             N,
                                                                             NULL,        // torque
                                                                             NULL,        // selected block
                                                                             false,       // sorted r[] and typeid[]
                                                                             sortedOutput // calcluated acceleration is sorted
    );
  }
  if (withInfo)
    ErrorInfo("calc Density with Cutoff Block (SPH_NS)");

  reduce27<<<MPnum, THnum1D>>>(num, tmp81N, N, N * 3, 0);
  reduce27<<<MPnum, THnum1D>>>(rho, tmp81N, N, N * 3, N);
  reduce27<<<MPnum, THnum1D>>>(mu, tmp81N, N, N * 3, N * 2);
  // re-pointing by num, mu is needed after the rho<->tmp3N swapping

#else
  std::cerr << "SPH_NS::calcDensity without cutoff Block is not implemented" << std::endl;
  exit(0);
#endif

  // no sorted output, no exchange
  if (!sortedOutput)
    calcDensityPost(false);
}

void cudaParticleSPH_NS::calcDensityPost(bool sortedOutput)
{
  if (sortedOutput)
  {
    RestoreByPid<<<MPnum, THnum1D>>>(tmp3N, num, N, pid);
    std::swap(num, tmp3N);
    rho = &(num[N]);
    mu = &(num[N * 2]);
  }

  // rhoinv = 1.0/rho
  calcReciproc<<<MPnum, THnum1D>>>(rho, rhoinv, N);
  if (withInfo)
    ErrorInfo("calc reciprocal of rho for SPH");

  // calculate field value mu from (\mu \rho)*\rho ^-1
  multiplies<<<MPnum, THnum1D>>>(mu, rhoinv, N);
  if (withInfo)
    ErrorInfo("\\mu\\rho *= \\rho^-1");
}

real cudaParticleSPH_NS::inspectDensity(void)
{
  inspectDense<<<1, 128, 128 * (sizeof(real) + sizeof(uint32_t))>>>(num, move, N, tmp3N);
  if (withInfo)
    ErrorInfo("inspect densities for SPH");
  real R[3] = {1.0, 1.0, 1.0};
  cudaMemcpy(&R, tmp3N, sizeof(real), cudaMemcpyDeviceToHost);
  //std::cerr << "mean num density: " << R[0] << "," << R[1] << "," << R[2] << std::endl;
  return R[0];
}

void cudaParticleSPH_NS::RestoreAcceleration(void)
{
  RestoreByPid<<<MPnum, THnum1D>>>(tmp3N, a, N, pid);
  //cudaThreadSynchronize();
  std::swap(a, tmp3N);

  if (withInfo)
    ErrorInfo("RestoreByPid from a to tmp3N");
}

void cudaParticleSPH_NS::getExchangePidRange1(void)
{
  uint32_t _p1 = myBlockOffset;
  uint32_t _p2 = _p1 + myBlockNum;

  bool _found = false;
  int trial = 0;
  int __size = ___p1.size();
  do
  {
    cudaMemcpy(&___p1[0], &(bindex[_p1]), sizeof(uint32_t) * __size, cudaMemcpyDeviceToHost);
    std::vector<uint32_t>::iterator _ix = std::find_if(___p1.begin(), ___p1.begin() + __size, [](uint32_t i) { return i != UINT_MAX; });
    if (_ix != ___p1.end())
    {
      p1 = *_ix;
      _found = true;
    }
    else
    {
      _p1 = myBlockOffset + ___p1.size() + trial * BINDEX_SEARCH_WIDTH;
      __size = BINDEX_SEARCH_WIDTH;
      ++trial;
    }
  } while (!_found);
  if (trial > 0)
    ___p1.resize(___p1.size() + (trial)*BINDEX_SEARCH_WIDTH);

  if (_p2 == totalNumBlock)
    p2 = N;
  else
  {
    _found = false;
    trial = 0;
    __size = ___p2.size();
    do
    {
      cudaMemcpy(&___p2[0], &(bindex[_p2]), sizeof(uint32_t) * __size, cudaMemcpyDeviceToHost);
      std::vector<uint32_t>::iterator _ix = std::find_if(___p2.begin(), ___p2.begin() + __size, [](uint32_t i) { return i != UINT_MAX; });
      if (_ix != ___p2.end())
      {
        p2 = *_ix;
        _found = true;
      }
      else
      {
        _p2 = myBlockOffset + myBlockNum + ___p2.size() + trial * BINDEX_SEARCH_WIDTH;
        __size = BINDEX_SEARCH_WIDTH;
        ++trial;
      }
    } while (!_found);
    if (trial > 0)
      ___p2.resize(___p2.size() + (trial)*BINDEX_SEARCH_WIDTH);
  }

  /*
#pragma omp critical
{
    std::cerr << myBlockOffset << ":" << myBlockOffset+myBlockNum << "[" << p1 << ":" << p2 << "]"
	      << "(" << ___p1.size() << ":" << ___p2.size() << ")"
	      << std::endl;
}
*/

  if (withInfo)
    ErrorInfo("getExchangePidRange1");
}
void cudaParticleSPH_NS::getExchangePidRange2(void)
{
  uint32_t _p3 = (totalNumBlock - numSelected) + myOffsetSelected;
  uint32_t _p4 = _p3 + myBlockSelected;
  const uint32_t _p5 = _p4;
  cudaMemcpy(&_p3, &(selectedBlock[_p3]), sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (_p5 < totalNumBlock)
    cudaMemcpy(&_p4, &(selectedBlock[_p4]), sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&_p3, &(bindex[_p3]), sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (_p5 < totalNumBlock)
    cudaMemcpy(&_p4, &(bindex[_p4]), sizeof(uint32_t), cudaMemcpyDeviceToHost);
  p3 = _p3;
  p4 = (_p5 < totalNumBlock) ? _p4 : N;
  //  std::cerr << p3 << ":" << p4 << std::endl;
  if (withInfo)
    ErrorInfo("getExchangePidRange2");
}

// override methods of cudaSelectedBlock
void cudaParticleSPH_NS::getForceSelected(const int typeID)
{
  if (typeID != 4)
    cudaSelectedBlock::getForceSelected(typeID);
  else
  {
    const size_t sizeN = sizeof(real) * (p2 - p1);

    pthread_mutex_lock(&mutTMP);
    cudaMemcpy(&(TMP[p1]), &(num[p1]), sizeN, cudaMemcpyDeviceToHost);
    cudaMemcpy(&(TMP[p1 + N]), &(num[p1 + N]), sizeN, cudaMemcpyDeviceToHost);
    cudaMemcpy(&(TMP[p1 + N * 2]), &(num[p1 + N * 2]), sizeN, cudaMemcpyDeviceToHost);
    pthread_mutex_unlock(&mutTMP);

    if (withInfo)
      ErrorInfo("cudaParticleSPH_NS::getForceSelected");
  }
}
void cudaParticleSPH_NS::importForceSelected(const cudaParticleSPH_NS &A, const int typeID,
                                             bool directAccess, int idMe, int idPeer)
{
  if (typeID != 4)
    cudaSelectedBlock::importForceSelected(A, typeID, directAccess, idMe, idPeer);
  else
  {
    const size_t sizeN = sizeof(real) * (A.p2 - A.p1);
    if (directAccess)
    {
      cudaMemcpyPeer(&(num[A.p1]), idMe, &(A.num[A.p1]), idPeer, sizeN);
      cudaMemcpyPeer(&(num[A.p1 + N]), idMe, &(A.num[A.p1 + N]), idPeer, sizeN);
      cudaMemcpyPeer(&(num[A.p1 + N * 2]), idMe, &(A.num[A.p1 + N * 2]), idPeer, sizeN);
      cudaThreadSynchronize();
    }
    else
    {
      cudaMemcpy(&(num[A.p1]), &(A.TMP[A.p1]), sizeN, cudaMemcpyHostToDevice);
      cudaMemcpy(&(num[A.p1 + N]), &(A.TMP[A.p1 + N]), sizeN, cudaMemcpyHostToDevice);
      cudaMemcpy(&(num[A.p1 + N * 2]), &(A.TMP[A.p1 + N * 2]), sizeN, cudaMemcpyHostToDevice);
    }

    if (withInfo)
      ErrorInfo("cudaParticleSPH_NS::importForceSelected");
  }
}
