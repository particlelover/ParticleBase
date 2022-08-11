#include "cudaParticleMD.hh"
#include "kernelfuncs.h"
#include <assert.h>

#include "kerneltemplate.hh"

cudaParticleMD::~cudaParticleMD() {
  if (m!=NULL) cudaFree(m);

  if (hdl!=NULL) cublasDestroy(hdl);
}

void cudaParticleMD::setup(int n) {
  cudaParticleVV::setup(n);

  // alloc m 1/(1/mass)
  cudaMalloc((void **)&m, sizeof(real)*N);
  if (withInfo) ErrorInfo("malloc m[] on GPU");

  // preparation for CUBLAS
  if (hdl==NULL) cublasCreate(&hdl);
}

void cudaParticleMD::calcForce(void) {
  clearArray_F4<<<MPnum, THnum1D>>>(F, N);

  class potentialLJ LJ;
  class MDpairForce<potentialLJ> P;
  P.cx = cell[6];
  P.cy = cell[7];
  P.cz = cell[8];
  P.c0x = sqrt(P.cx) * 2.0;
  P.c0y = sqrt(P.cy) * 2.0;
  P.c0z = sqrt(P.cz) * 2.0;
  P.rmax2 = rmax2;
  P.typeID = typeID_s;
  P.op = LJ;
#if defined(CUDACUTOFFBLOCK)
  assert(myBlockNum<maxGrid);

  if (useNlist) {
    calcF_IJpairWithList_F4<<<MPnum, THnum1DX>>>(P,
      NList.rowPtr, NList.colIdx,
      r, F, N,
      NULL, 0, 0);
      
  } else {
    clearArray<<<MPnum, THnum1D>>>(tmp81N, N*81);
    dim3 _mpnum, _thnum;
    _mpnum.x = myBlockNum ; _mpnum.y = 27; _mpnum.z = 1;
    _thnum.x = THnum2D2; _thnum.y = 1; _thnum.z = 1;
    assert(_thnum.x*_thnum.y <= threadsMax);
    calcF_IJpairWithBlock_F4<class MDpairForce<potentialLJ> ><<<_mpnum, _thnum>>>(P, r_s,
      tmp81N,
      myBlockOffset,
      blockNeighbor, pid, bindex,
      N,
      NULL,NULL, true);
    reduce27_F4<<<MPnum, THnum1D>>>(F, tmp81N, N);
  }
  if (withInfo) ErrorInfo("cudaParticleMD::calcForce With Block");
#else

  calcF_IJpair<class MDpairForce<potentialLJ> ><<<MPnum, THnum1D>>>(P, r, F, N);
  //calcF_IJpair<class potentialLJ><<<_mpnum, _thnum>>>(LJ, r, F, typeID, N, cell[6], cell[7], cell[8], rmax2);
  //calcF_LJ<<<_mpnum, _thnum>>>(r, F, typeID, N, cx*cx, cy*cy, cz*cz);
  if (withInfo) ErrorInfo("cudaParticleMD::calcForce Without Block");
#endif
}

real cudaParticleMD::calcPotentialE(void) {
  clearArray_F4<<<MPnum, THnum1D>>>(tmp3N, N);

  class potentialLJ LJ;
  class MDpairPotential<potentialLJ> P;
  P.cx = cell[6];
  P.cy = cell[7];
  P.cz = cell[8];
  P.c0x = sqrt(P.cx) * 2.0;
  P.c0y = sqrt(P.cy) * 2.0;
  P.c0z = sqrt(P.cz) * 2.0;
  P.rmax2 = rmax2;
  P.typeID = typeID_s;
  P.op = LJ;
  real f = 0;
#if defined(CUDACUTOFFBLOCK)
  assert(myBlockNum<maxGrid);

  if (useNlist) {
    calcF_IJpairWithList_F4<<<MPnum, THnum1DX>>>(P,
      NList.rowPtr, NList.colIdx,
      r, tmp3N, N,
      NULL, 0, 0);
    if (withInfo) ErrorInfo("cudaParticleMD::calcPotentialE calc");
    accumulate<<<1, threadsMax, sizeof(double)*threadsMax>>>(tmp3N, N, tmp3N);
    if (withInfo) ErrorInfo("cudaParticleMD::calcPotentialE accumulate");
    cudaMemcpy(&f, tmp3N, sizeof(float), cudaMemcpyDeviceToHost);

  } else {
    clearArray<<<MPnum, THnum1D>>>(tmp81N, N*81);
    dim3 _mpnum, _thnum;
    _mpnum.x = myBlockNum ; _mpnum.y = 27; _mpnum.z = 1;
    _thnum.x = THnum2D2; _thnum.y = 1; _thnum.z = 1;
    assert(_thnum.x*_thnum.y <= threadsMax);
    calcF_IJpairWithBlock_F4<class MDpairPotential<potentialLJ> ><<<_mpnum, _thnum>>>(P, r_s,
      tmp81N,
      myBlockOffset,
      blockNeighbor, pid, bindex,
      N,
      NULL,NULL, true);
    reduce27_F4<<<MPnum, THnum1D>>>(tmp3N, tmp81N, N);
    if (withInfo) ErrorInfo("cudaParticleMD::calcPotentialE calc");
    accumulate<<<1, threadsMax, sizeof(double)*threadsMax>>>(tmp3N, N, tmp3N);
    if (withInfo) ErrorInfo("cudaParticleMD::calcPotentialE accumulate");
    cudaMemcpy(&f, tmp3N, sizeof(float), cudaMemcpyDeviceToHost);

  }
#else
  std::cerr << "cudaParticleMD::calcPotentialE() is not implemented" << std::endl;
#endif
  return f;
}

real cudaParticleMD::calcMV2(void) {
  calcV2_F4<<<MPnum, THnum1D>>>(v, reinterpret_cast<float*>(tmp3N), N);
  if (withInfo) ErrorInfo("cudaParticleMD::calcV2");

  real t=0;
  DOT(hdl, N, reinterpret_cast<float*>(tmp3N), 1, m, 1, &t);
  if (withInfo) ErrorInfo("cudaParticleMD::DOT");

  return t;
}

void cudaParticleMD::setM(void) {
  calcReciproc<<<MPnum, THnum1D>>>(minv, m, N);
  if (withInfo) ErrorInfo("calc reciprocal of minv to m");
}

real cudaParticleMD::constTemp(void) {
  // calc kernel calculation first (before the CUBLAS funcs)
  real t = calcMV2();
  //std::cerr << "constTemp T= " << t << std::endl;

  real lambda=0;
  DOT(hdl, N*4, (float*)F, 1, (float*)v, 1, &lambda);

  lambda /= t;  // \sum F_i v_i / \sum m_i v_i^2
  std::cerr << "[l=" << lambda << "]";

  if (lambda != 0)
    correctConstTemp_F4<<<MPnum, THnum1D>>>(v, F, m, lambda, N);

  return t/(3*N*kB);
}

void cudaParticleMD::statMV2(std::ostream &o) {
  calcMV2();

  pthread_mutex_lock(&(mutTMP));
  cudaMemcpy(&(TMP[0]), tmp3N, sizeof(float)*N, cudaMemcpyDeviceToHost);
  o << std::endl << std::endl;
  for (int i=0;i<N;++i)
    o << i << "\t" << TMP[i] << std::endl;
  pthread_mutex_unlock(&(mutTMP));
}

real cudaParticleMD::scaleTemp(real Temp) {
  const real T0 = calcTemp();
  const real s = sqrt(static_cast<real>(3*N-1)/static_cast<real>(3*N)*Temp/T0);

  mulArray<<<MPnum, THnum1D>>>((float*)v, s, N*4);

  return s;
}

void cudaParticleMD::adjustVelocities(real Temp, bool debug) {
  real v1 = sqrt(kB * Temp / m0);
  uint32_t __thnum = std::min((uint32_t)1024, threadsMax);

  if (debug) {
    std::cerr << std::endl << std::endl << "adjustVelocity currentTemp\t" << calcTemp()
      << "\tTarget Temp " << Temp << std::endl;
    
    adjustVelocity_F4<<<1, __thnum, sizeof(double)*__thnum*6>>>(v, __thnum, N, v1, reinterpret_cast<float*>(tmp3N));
    cudaDeviceSynchronize();
    pthread_mutex_lock(&(mutTMP));
    cudaMemcpy(&(TMP[0]), tmp3N, sizeof(float)*6, cudaMemcpyDeviceToHost);
    std::cerr << "velocity statistics";
    for (int i=0;i<6;++i)
      std::cerr << "\t" << TMP[i];
    std::cerr << std::endl;
    pthread_mutex_unlock(&(mutTMP));
    std::cerr << "after adjusted T=" << calcTemp() << std::endl;
  } else {
    adjustVelocity_F4<<<1, __thnum, sizeof(double)*__thnum*6>>>(v, __thnum, N, v1);
  }
}

void cudaParticleMD::setLJparams(const std::vector<real> &p, uint32_t elemnum) {
  _setLJparams(p, elemnum);
}


void cudaParticleMD::initialAnnealing(uint32_t anealstep, real dt, real _rc, real _f0, real T) {
  std::cerr << "performs initial annealing by soft core potential to rmax= " << _rc << " with steps " << anealstep << std::endl;

  float3 c1, c2;
  c1.x = cell[0];
  c1.y = cell[2];
  c1.z = cell[4];
  c2.x = cell[1];
  c2.y = cell[3];
  c2.z = cell[5];

#if defined(CUDACUTOFFBLOCK)
  dim3 __mpnum, __thnum;
  __mpnum.x = myBlockNum ; __mpnum.y = 27; __mpnum.z = 1;
  __thnum.x = THnum2D2; __thnum.y = 1; __thnum.z = 1;
  std::cerr << "Init Aneal with " << __mpnum.x << "x" << __mpnum.y
            << " blocks " << __thnum.x << "x" << __thnum.y << " threads" << std::endl;
  assert(myBlockNum<maxGrid);
#else
  std::cerr << "Init Aneal with " << MPnum
            << " blocks " << THnum1D << " threads" << std::endl;
#endif

  class potentialSC SC;
  SC.rc = _rc;
  SC.f0 = _f0;

  class MDpairForce<potentialSC> P;
  P.cx = cell[6];
  P.cy = cell[7];
  P.cz = cell[8];
  P.c0x = sqrt(P.cx) * 2.0;
  P.c0y = sqrt(P.cy) * 2.0;
  P.c0z = sqrt(P.cz) * 2.0;
  P.rmax2 = rmax2;
  P.typeID = typeID;
  P.op = SC;

  for (uint32_t i=0;i<anealstep;++i) {
    clearArray_F4<<<MPnum, THnum1D>>>(F, N);
    if (withInfo) ErrorInfo("clear Forces");

    SC.rc = _rc * (i+1)/anealstep;
    std::cerr << "\r" << SC.rc << "\t" << std::flush;


#if defined(CUDACUTOFFBLOCK)
    calcBlockID();
    clearArray<<<MPnum, THnum1D>>>(tmp81N, N*81);
    P.op.rc = SC.rc;
    calcF_IJpairWithBlock_F4<class MDpairForce<potentialSC> >
      <<<__mpnum, __thnum>>>(P, r,
      tmp81N,
      myBlockOffset,
      blockNeighbor, pid, bindex,
      N);
    reduce27_F4<<<MPnum, THnum1D>>>(F, tmp81N, N);
#else
    P.op.rc = SC.rc;
    calcF_IJpair<class MDpairForce<potentialSC> ><<<MPnum, THnum1D>>>(P, r, F, N);
#endif
    if (withInfo) ErrorInfo("calc Forces by softcore");

    // correct force by velocity
    constTemp();

    calcA_F4<<<MPnum, THnum1D>>>(a, minv, F, N);
    if (withInfo) ErrorInfo("calcAcceleration");

    propagateEuler_F4<<<MPnum, THnum1D>>>(r, dt, v, a, move, N);
    if (withInfo) ErrorInfo("propagate by Euler");

    applyPeriodicCondition_F4<<<MPnum, THnum1D>>>(r, c1, c2, N);

    //adjustVelocities(MPnum, THnum1D, T);
    std::cerr << "T= " << calcTemp() << std::flush;
  }
  std::cerr << std::endl;

  std::cerr << "temperature scaling: " << scaleTemp(T) << std::endl;

  std::cerr << "with LJ " << std::endl;
  class potentialLJ LJ;
  clearArray_F4<<<MPnum, THnum1D>>>(Fold, N);

  class MDpairForce<potentialLJ> P2;
  P2.cx = cell[6];
  P2.cy = cell[7];
  P2.cz = cell[8];
  P2.c0x = sqrt(P2.cx) * 2.0;
  P2.c0y = sqrt(P2.cy) * 2.0;
  P2.c0z = sqrt(P2.cz) * 2.0;
  P2.rmax2 = rmax2;
  P2.typeID = typeID;
  P2.op = LJ;

  for (uint32_t i=0;i<anealstep;++i) {
    std::cerr << i << "\t" << calcTemp() << "\t"; 
    clearArray_F4<<<MPnum, THnum1D>>>(F, N);

#if defined(CUDACUTOFFBLOCK)
    calcBlockID();
    clearArray<<<MPnum, THnum1D>>>(tmp81N, N*81);
    calcF_IJpairWithBlock_F4<class MDpairForce<potentialLJ> >
      <<<__mpnum, __thnum>>>(P2, r,
      tmp81N,
      myBlockOffset,
      blockNeighbor, pid, bindex,
      N);
    reduce27_F4<<<MPnum, THnum1D>>>(F, tmp81N, N);
#else
    calcF_IJpair<class MDpairForce<potentialLJ> ><<<MPnum, THnum1D>>>(P2, r, F, N);
#endif
    std::cerr << constTemp() << std::endl;
    propagateVelocityVerlet_F4<<<MPnum, THnum2D*THnum2D/2>>>(r, dt, v, F, Fold, minv, N);
    //std::cerr << scaleTemp(MPnum, THnum1D, T) << std::endl;
    applyPeriodicCondition_F4<<<MPnum, THnum1D>>>(r, c1, c2, N);
  }

  std::cerr << "temperature scaling: " << scaleTemp(T) << std::endl;
}

void cudaParticleMD::import(const std::vector<ParticleBase> &P) {
  cudaParticleBase::import(P);
  
  double _m0 = std::accumulate(P.begin(), P.end(), 0.0,
    [](double acc, ParticleBase cur) {
      acc += cur.m;
      return acc;
    });
  m0 = _m0 / P.size();
}

void cudaParticleMD::makeNList(void) {
  if (!useNlist) {
    return;
  }

  real rmax2t = (sqrt(rmax2) + thickness) * (sqrt(rmax2) + thickness);
  // 1) calc coordination number
  class calcCoordMD_F4 C;
  C.cx = cell[6];
  C.cy = cell[7];
  C.cz = cell[8];
  C.rmax2 = rmax2t;
  
  calcBlockID();
  clearArray<<<MPnum, THnum1D>>>(tmp81N, N*81);

  dim3 __mpnum, __thnum;
  __mpnum.x = myBlockNum ; __mpnum.y = 27; __mpnum.z = 1;
  __thnum.x = THnum2D2; __thnum.y = 1; __thnum.z = 1;
  calcF_IJpairWithBlock_F4<class calcCoordMD_F4>
    <<<__mpnum, __thnum>>>(C, r,
    tmp81N,
    myBlockOffset,
    blockNeighbor, pid, bindex,
    N);
  clearArray_F4<<<MPnum, THnum1D>>>(F, N);
  reduce27_F4<<<MPnum, THnum1D>>>(F, tmp81N, N);

  real2ulong_F4<<<MPnum, THnum1D>>>(F, &(NList.rowPtr[1]), N);

  // 2) make row pointer and column index
  uint32_t nnz = NList.makeRowPtr();
  NList.Resize(nnz);

  class MDNeighborRegister C2;
  C2.cx = cell[6];
  C2.cy = cell[7];
  C2.cz = cell[8];
  C2.rmax2 = rmax2t;
  
  dim3 _mpnum;
  uint32_t z = 0;
  _mpnum.x = myBlockNum; _mpnum.y = 1; _mpnum.z = 1;
  makeJlist_WithBlock_F4<<<_mpnum, THnum2D>>>(C2,
    NList.rowPtr, NList.colIdx,
    r,
    z,
    blockNeighbor, pid, bindex,
    N, NULL);

  /** no need for MD neighbor list
  sortColIdx<<<MPnum, THnum1D>>>(NList.rowPtr,
    NList.colIdx, N);
   */
  if (withInfo) ErrorInfo("cudaParticleMD::make Neighbor list");

}