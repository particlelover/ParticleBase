#define WITHJLIST
#include "cudaParticleDEM.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"
#include <assert.h>

cudaParticleDEM::~cudaParticleDEM()
{
  if (r0 != NULL)
    cudaFree(r0);

#if !defined(WITHJLIST)
  if (tmp81N_TRQ != NULL)
    cudaFree(tmp81N_TRQ);
#endif

  if (bid_by_pid != NULL)
    cudaFree(bid_by_pid);
}

void cudaParticleDEM::setup(int n)
{
  cudaParticleLF::setup(n);
  cudaParticleRotation::setup(n);

  cudaMalloc((void **)&r0, sizeof(real) * N);

  clearArray<<<MPnum, THnum1D>>>(r0, N, 1.0);

#if !defined(WITHJLIST)
  cudaMalloc((void **)&tmp81N_TRQ, sizeof(real) * 3 * N * 27);
  if (withInfo)
    ErrorInfo("malloc tmp81N_TRQ[] on GPU");
#endif

#if defined(WithBlock2)
  cudaMalloc((void **)&bid_by_pid, sizeof(uint32_t) * N);
#endif

  contactMat[0].setup(N, threadsMax);
  contactMat[1].setup(N, threadsMax);

  clearArray<uint32_t><<<MPnum, THnum2D>>>(contactMat[0].rowPtr, N + 1);
  clearArray<uint32_t><<<MPnum, THnum2D>>>(contactMat[1].rowPtr, N + 1);
}

void cudaParticleDEM::calcForce(real dt)
{

  // 1) calc coordination number

#if defined(WithBlock2)
  clearArray<<<MPnum, THnum1D>>>(F, N);
  //  std::cerr << p2 << ":" << p1 << ":" << MPnum*THnum1D << std::endl;
  if (SingleParticleBlock)
  {
    calcF_IJpairWithBlock4<class calcCoord><<<MPnum, THnum1D>>>(C, r,
                                                                F,
                                                                blockNeighbor, bindex,
                                                                bid, move,
                                                                N, NULL, p3, N - p4);
  }
  else
  {
    calcF_IJpairWithBlock2<class calcCoord><<<MPnum, THnum1D>>>(C, r,
                                                                F,
                                                                blockNeighbor, pid, bindex,
                                                                bid_by_pid, move,
                                                                N, NULL, p3, N - p4);
  }
  if (withInfo)
    ErrorInfo("cudaParticleDEM::calc Coordination number");
#else
  dim3 _mpnum, _thnum;
  _thnum.x = THnum2D2;
  _thnum.y = 27;
  _thnum.z = 1;
  _mpnum.x = myBlockNum;
  _mpnum.y = 1;
  _mpnum.z = 1;

  // use selected blocks if selectBlock() was performed
  uint32_t *Q = NULL;
  int validOffset = myBlockOffset;
  if (myBlockSelected > 0)
  {
    Q = selectedBlock;
    _mpnum.x = myBlockSelected;
    validOffset = totalNumBlock - numSelected;
  }
  clearArray<<<MPnum, THnum1D>>>(tmp81N, N * 81);
  calcF_IJpairWithBlock<class calcCoord><<<_mpnum, _thnum>>>(C, r,
                                                             tmp81N,
                                                             validOffset,
                                                             blockNeighbor, pid, bindex,
                                                             N, NULL, Q);
  if (withInfo)
    ErrorInfo("cudaParticleDEM::calc Coordination number");
  clearArray<<<MPnum, THnum1D>>>(F, N);
  reduce27<<<MPnum, THnum1D>>>(F, tmp81N, N, N * 3);
#endif

  const int __I = (_N++ % 2);
  //  std::cerr << "Mat: " << __I << std::endl;
  real2ulong<<<MPnum, THnum1D>>>(F, &(contactMat[__I].rowPtr[1]), N);

  // 2) make row pointer and column index
  uint32_t nnz = contactMat[__I].makeRowPtr();
  contactMat[__I].Resize(nnz);

  /*
  std::cerr << "(" << nnz << ")";

  std::vector<uint32_t> T0(N+1);
  cudaMemcpy(&(T0[0]), contactMat[__I].rowPtr, sizeof(uint32_t)*(N+1), cudaMemcpyDeviceToHost);

  uint32_t _prev=0;
  for (int i=0;i<N+1;++i) {
    if (T0[i]>_prev) {
      std::cerr << i << ":" << T0[i] << " ";
    }
    _prev = T0[i];
  }
  std::cerr << std::endl;
*/

#if defined(WithBlock2)
  if (SingleParticleBlock)
  {
    makeJlist_WithBlock4<<<MPnum, THnum1D>>>(contactMat[__I].rowPtr,
                                             contactMat[__I].colIdx,
                                             r0, r,
                                             blockNeighbor, bindex,
                                             bid, move,
                                             N, p3, N - p4);
  }
  else
  {
    makeJlist_WithBlock2<<<MPnum, THnum1D>>>(contactMat[__I].rowPtr,
                                             contactMat[__I].colIdx,
                                             r0, r,
                                             blockNeighbor, pid, bindex,
                                             bid_by_pid, move,
                                             N, p3, N - p4);
  }
#else
  dim3 _mpnum, _thnum;
  _mpnum.x = myBlockNum;
  _mpnum.y = 1;
  _mpnum.z = 1;
  makeJlist_WithBlock<<<_mpnum, THnum2D>>>(contactMat[__I].rowPtr,
                                           contactMat[__I].colIdx,
                                           r0, r,
                                           validOffset,
                                           blockNeighbor, pid, bindex,
                                           N, Q);
#endif

  sortColIdx<<<MPnum, THnum1D>>>(contactMat[__I].rowPtr,
                                 contactMat[__I].colIdx, N);
  if (withInfo)
    ErrorInfo("cudaParticleDEM::make J list");

  // 3) copy from previous Mat
  clearArray<<<MPnum, THnum1D>>>(contactMat[__I].val, nnz);
  succeedPrevState<<<MPnum, THnum1D>>>(contactMat[(__I + 1) % 2].rowPtr, contactMat[(__I + 1) % 2].colIdx, contactMat[(__I + 1) % 2].val,
                                       contactMat[__I].rowPtr, contactMat[__I].colIdx, contactMat[__I].val, N);
  if (withInfo)
    ErrorInfo("cudaParticleDEM::copy SparseMat from old");

  /*
  std::vector<uint32_t> T1(nnz);
  cudaMemcpy(&(T1[0]), contactMat[__I].colIdx, sizeof(uint32_t)*(nnz), cudaMemcpyDeviceToHost);

  for (int i=0;i<nnz;++i)
    std::cerr << i << ":" << T1[i] << " ";
  std::cerr << std::endl;
  std::cerr << std::endl;

  for (int i=0;i<N;++i) {
    if ((T0[i+1]-T0[i])>0)
      std::cerr << "\n" << i << ": ";
    for (int j=T0[i];j<T0[i+1];++j)
      std::cerr << T1[j] << " ";
  }
  std::cerr << std::endl;
*/

  // 4) calc contact forces with i-j sparse matrix

  clearArray<<<MPnum, THnum1D>>>(F, N * 3);
  clearArray<<<MPnum, THnum1D>>>(T, N * 3);

  //  real m = static_cast<real>(N)/(THnum1D*4);
  //  uint32_t _mpnum = static_cast<uint32_t>(trunc(m));

  //class DEMContact P;
  P.rowPtr = contactMat[__I].rowPtr;
  P.colIdx = contactMat[__I].colIdx;
  P.val = contactMat[__I].val;
  P.deltaT = dt;

#if defined(WITHJLIST)
  calcF_IJpairWithList<<<MPnum, (1 * THnum1D) / 2>>>(P,
                                                     contactMat[__I].rowPtr, contactMat[__I].colIdx,
                                                     r, F, N, T, p3, N - p4);
  if (withInfo)
    ErrorInfo("cudaParticleDEM::calcForce With JLIST");
#else
#if defined(CUDACUTOFFBLOCK)
  //  dim3 _mpnum, _thnum;
  _thnum.x = THnum2D2;
  _thnum.y = 9;
  _thnum.z = 1;
  _mpnum.x = myBlockNum;
  _mpnum.y = 3;
  _mpnum.z = 1;

  // use selected blocks if selectBlock() was performed
  uint32_t *Q = NULL;
  int validOffset = myBlockOffset;
  if (myBlockSelected > 0)
  {
    Q = selectedBlock;
    _mpnum.x = myBlockSelected;
    validOffset = totalNumBlock - numSelected;
  }

  clearArray<<<MPnum, THnum1D>>>(tmp81N, N * 81);
  clearArray<<<MPnum, THnum1D>>>(tmp81N_TRQ, N * 81);
  calcF_IJpairWithBlock<class DEMContact><<<_mpnum, _thnum>>>(P, r,
                                                              tmp81N,
                                                              validOffset,
                                                              blockNeighbor, pid, bindex,
                                                              N, tmp81N_TRQ, Q);
  reduce27<<<MPnum, THnum1D>>>(F, tmp81N, N * 3);
  reduce27<<<MPnum, THnum1D>>>(T, tmp81N_TRQ, N * 3);

  if (withInfo)
    ErrorInfo("cudaParticleDEM::calcForce With Block");
#else
  calcF_IJpair<class DEMContact><<<MPnum, THnum1D>>>(P, r, F, N, T);
  if (withInfo)
    ErrorInfo("calcForce for DEM");
#endif
#endif
}

void cudaParticleDEM::TimeEvolution(real dt)
{
  cudaParticleLF::TimeEvolution(dt);

  cudaParticleRotation::TimeEvolution(dt);
}

void cudaParticleDEM::setDEMProperties(real _E, real _mu, real _sigma, real _gamma, real _mu_r,
                                       const std::vector<real> &_r0)
{
  assert(_sigma < 0.5);
  E = _E * (1 - _sigma * _sigma);
  mu = _mu;
  s2 = (2 - _sigma) / 2 / (1 - _sigma);
  gamma = _gamma;
  mu_r = _mu_r;

  cudaMemcpy(r0, &(_r0[0]), sizeof(real) * N, cudaMemcpyHostToDevice);

  setFunctorMembers();
}

void cudaParticleDEM::setFunctorMembers(void)
{
  //class calcCoord C;
  C.r0 = r0;

  //class DEMContact P;
  P.v = v;
  P.w = w;
  P.move = move;
  P.r0 = r0;
  P.E = E;
  P.mu = mu;
  P.s2 = s2;
  P.gamma2 = gamma;
  P.mu_r = mu_r;
  P.minv = minv;
}

void cudaParticleDEM::setDEMProperties(real _E, real _mu, real _sigma, real _gamma, real _mu_r,
                                       real r0_all)
{
  std::vector<real> R0_All(N, r0_all);

  setDEMProperties(_E, _mu, _sigma, _gamma, _mu_r, R0_All);
}

void cudaParticleDEM::calcBlockID(void)
{
  if (SingleParticleBlock)
  {
    clearArray<<<MPnum, THnum1D>>>(bindex, totalNumBlock, UINT_MAX);

    calcBID_direct<<<MPnum, THnum1D>>>(r, bid, bindex, N, blocklen[0], blocklen[1], blocklen[2],
                                       blocklen[3], blocklen[4], blocklen[5], blocknum[0], blocknum[1], blocknum[2],
                                       blocknum[2], blocknum[1] * blocknum[2]);

    if (withInfo)
      ErrorInfo("calcBID_direct");
  }
  else
  {
    cudaCutoffBlock::calcBlockID();

    makeBIDbyPID<<<MPnum, THnum1D>>>(pid, bid, bid_by_pid, N);
  }
}

// override methods of cudaSelectedBlock
void cudaParticleDEM::getForceSelected(const int typeID)
{
  if (typeID != 2)
    cudaSelectedBlock::getForceSelected(typeID);
  else
  {
    const size_t sizeN = sizeof(real) * (p4 - p3);

    pthread_mutex_lock(&mutTMP);
    cudaMemcpy(&(TMP[p3]), &(T[p3]), sizeN, cudaMemcpyDeviceToHost);
    cudaMemcpy(&(TMP[p3 + N]), &(T[p3 + N]), sizeN, cudaMemcpyDeviceToHost);
    cudaMemcpy(&(TMP[p3 + N * 2]), &(T[p3 + N * 2]), sizeN, cudaMemcpyDeviceToHost);
    pthread_mutex_unlock(&mutTMP);

    if (withInfo)
      ErrorInfo("cudaParticleDEM::getForceSelected");
  }
}
void cudaParticleDEM::importForceSelected(const cudaParticleDEM &A, const int typeID,
                                          bool directAccess, int idMe, int idPeer)
{
  if (typeID != 2)
    cudaSelectedBlock::importForceSelected(A, typeID, directAccess, idMe, idPeer);
  else
  {
    const size_t sizeN = sizeof(real) * (A.p4 - A.p3);
    if (directAccess)
    {
      cudaMemcpyPeer(&(T[A.p3]), idMe, &(A.T[A.p3]), idPeer, sizeN);
      cudaMemcpyPeer(&(T[A.p3 + N]), idMe, &(A.T[A.p3 + N]), idPeer, sizeN);
      cudaMemcpyPeer(&(T[A.p3 + N * 2]), idMe, &(A.T[A.p3 + N * 2]), idPeer, sizeN);
      cudaThreadSynchronize();
    }
    else
    {
      // typeID==2
      cudaMemcpy(&(T[A.p3]), &(A.TMP[A.p3]), sizeN, cudaMemcpyHostToDevice);
      cudaMemcpy(&(T[A.p3 + N]), &(A.TMP[A.p3 + N]), sizeN, cudaMemcpyHostToDevice);
      cudaMemcpy(&(T[A.p3 + N * 2]), &(A.TMP[A.p3 + N * 2]), sizeN, cudaMemcpyHostToDevice);
    }

    if (withInfo)
      ErrorInfo("cudaParticleDEM::importForceSelected");
  }
}

void cudaParticleDEM::rollback(real dt)
{
  cudaParticleLF::rollback(dt);
  cudaParticleRotation::TimeEvolution(-dt);
  _N -= 1;
}
void cudaParticleDEM::rollback2(real dt)
{
  cudaParticleLF::calcVinit(dt / 2.0);
  cudaParticleRotation::TimeEvolution(-dt);
}
