#include "cudaParticleBase.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"

#include <iostream>

cudaParticleBase::~cudaParticleBase() {
  if (r!=NULL) cudaFree(r);
  if (minv!=NULL) cudaFree(minv);
  if (v!=NULL) cudaFree(v);
  if (a!=NULL) cudaFree(a);
  if (F!=NULL) cudaFree(F);
  if (tmp3N!=NULL) cudaFree(tmp3N);
  if (typeID!=NULL) cudaFree(typeID);
  if (move!=NULL) cudaFree(move);
}

void cudaParticleBase::setup(int n) {

  N = n;
  TMP.resize(n*4);

  TMP2.resize(n);

  // alloc x, y, z
  cudaMalloc((void **)&r, sizeof(float4)*N);
  if (withInfo) ErrorInfo("malloc r[] on GPU");

  // alloc mass inv
  cudaMalloc((void **)&minv, sizeof(real)*N);
  if (withInfo) ErrorInfo("malloc minv[] on GPU");

  // alloc vx, vy, vz
  if (omitlist.count(ArrayType::velocity)==0) {
    cudaMalloc((void **)&v, sizeof(float4)*N);
  } else {
    std::cerr << "velocity[] not created" << std::endl;
  }
  if (withInfo) ErrorInfo("malloc v[] on GPU");

  // alloc ax, ay, az
  cudaMalloc((void **)&a, sizeof(float4)*N);
  if (withInfo) ErrorInfo("malloc a[] on GPU");

  // alloc Fx, Fy, Fz
  if (omitlist.count(ArrayType::force)==0) {
    cudaMalloc((void **)&F, sizeof(float4)*N);
  } else {
    std::cerr << "force[] not created" << std::endl;
  }
  if (withInfo) ErrorInfo("malloc F[] on GPU");

  // tmp3N
  cudaMalloc((void **)&tmp3N, sizeof(float4)*N);
  if (withInfo) ErrorInfo("malloc tmp3N[] on GPU");

  // alloc typeID
  cudaMalloc((void **)&typeID, sizeof(unsigned short int)*N);
  if (withInfo) ErrorInfo("malloc typeID[] on GPU");

  // alloc move flag
  cudaMalloc((void **)&move, sizeof(char)*N);
  if (withInfo) ErrorInfo("malloc move[] on GPU");

  //threadsMax = getNumTh();
}

void cudaParticleBase::import(const std::vector<ParticleBase> &P) {
  const size_t sizeN = sizeof(real) * N;
  const size_t sizeN3 = sizeof(float4) * N;

#pragma omp parallel for
  for (int i=0;i<N;++i) {
    TMP[i*4]   = P[i].r[0];
    TMP[i*4+1] = P[i].r[1];
    TMP[i*4+2] = P[i].r[2];
    TMP[i*4+3] = 0;
  }
  cudaMemcpy(r, &(TMP[0]), sizeN3, cudaMemcpyHostToDevice);
  if (withInfo) ErrorInfo("copy r[] to GPU");

  if (omitlist.count(ArrayType::velocity)==0) {
#pragma omp parallel for
    for (int i=0;i<N;++i) {
      TMP[i*4]   = P[i].v[0];
      TMP[i*4+1] = P[i].v[1];
      TMP[i*4+2] = P[i].v[2];
      TMP[i*4+3] = 0;
    }
    cudaMemcpy(v, &(TMP[0]), sizeN3, cudaMemcpyHostToDevice);
    if (withInfo) ErrorInfo("copy v[] to GPU");
  }

#pragma omp parallel for
  for (int i=0;i<N;++i) {
    TMP[i*4]   = P[i].a[0];
    TMP[i*4+1] = P[i].a[1];
    TMP[i*4+2] = P[i].a[2];
    TMP[i*4+3] = 0;
  }
  cudaMemcpy(a, &(TMP[0]), sizeN3, cudaMemcpyHostToDevice);
  if (withInfo) ErrorInfo("copy a[] to GPU");

  for (size_t i=0;i<N;++i) TMP[i] = 1.0/P[i].m;
  cudaMemcpy(minv,      &(TMP[0]), sizeN, cudaMemcpyHostToDevice);
  if (withInfo) ErrorInfo("copy minv[] to GPU");

  std::valarray<unsigned short int> _T(N);
  for (size_t i=0;i<N;++i) _T[i] = P[i].type;
  cudaMemcpy(typeID,    &(_T[0]), sizeof(unsigned short int)*N, cudaMemcpyHostToDevice);
  if (withInfo) ErrorInfo("copy type[] to GPU");

  std::valarray<char> _T2(N);
  for (size_t i=0;i<N;++i) _T2[i] = (P[i].isFixed) ? 0 : 1;
  cudaMemcpy(move,      &(_T2[0]), sizeof(char)*N, cudaMemcpyHostToDevice);
  if (withInfo) ErrorInfo("copy type[] to GPU");
}

void cudaParticleBase::getTypeID(void) {
  cudaMemcpy(&(TMP2[0]), typeID, sizeof(unsigned short)*N, cudaMemcpyDeviceToHost);
  if (withInfo) ErrorInfo("do getTypeID");
}
void cudaParticleBase::getPosition(void) {
  size_t sizeN = sizeof(float4) * N;

  pthread_mutex_lock(&mutTMP);
  cudaMemcpy(&(TMP[0]), r, sizeN, cudaMemcpyDeviceToHost);
  pthread_mutex_unlock(&mutTMP);

  if (withInfo) ErrorInfo("do getPosition");
}
void cudaParticleBase::getAcceleration(void) {
  size_t sizeN = sizeof(float4) * N;

  pthread_mutex_lock(&mutTMP);
  cudaMemcpy(&(TMP[0]), a, sizeN, cudaMemcpyDeviceToHost);
  pthread_mutex_unlock(&mutTMP);

  if (withInfo) ErrorInfo("do getAcceleration");
}
void cudaParticleBase::getForce(void) {
  size_t sizeN = sizeof(float4) * N;

  pthread_mutex_lock(&mutTMP);
  cudaMemcpy(&(TMP[0]), F, sizeN, cudaMemcpyDeviceToHost);
  pthread_mutex_unlock(&mutTMP);

  if (withInfo) ErrorInfo("do getForce");
}


void cudaParticleBase::TimeEvolution(real dt) {
  //propagateEuler<<<MPnum, THnum1D>>>(r, dt, v, a, move, N);

  if (withInfo) ErrorInfo("do TimeEvolution");
}

void cudaParticleBase::clearForce(void) {
  if (omitlist.count(ArrayType::force)==0) {
    clearArray_F4<<<MPnum, THnum1D>>>(F, N);
    if (withInfo) ErrorInfo("clearForce");
  }
}

void cudaParticleBase::clearTmp3N(void) {
  clearArray_F4<<<MPnum, THnum1D>>>(tmp3N, N);
  if (withInfo) ErrorInfo("clearTmp3N");
}

void cudaParticleBase::calcAcceleration(void) {
  calcA_F4<<<MPnum, THnum1D>>>(a, minv, F, N);
  if (withInfo) ErrorInfo("calcAcceleration");
}

void cudaParticleBase::treatPeriodicCondition(void) {
  float3 c1, c2;
  c1.x = cell[0];
  c1.y = cell[2];
  c1.z = cell[4];
  c2.x = cell[1];
  c2.y = cell[3];
  c2.z = cell[5];
  applyPeriodicCondition_F4<<<MPnum, THnum1D>>>(r, c1, c2, N);
}

void cudaParticleBase::treatAbsoluteCondition(void) {
  float4 c0, c1;
  c0.x = cell[0];
  c1.x = cell[1];
  c0.y = cell[2];
  c1.y = cell[3];
  c0.z = cell[4];
  c1.z = cell[5];
  treatAbsoluteBoundary_F4<<<MPnum, THnum1D>>>(r, c0, c1, N);
}

void cudaParticleBase::treatRefrectCondition(void) {
  float4 c0, c1;
  c0.x = cell[0];
  c1.x = cell[1];
  c0.y = cell[2];
  c1.y = cell[3];
  c0.z = cell[4];
  c1.z = cell[5];
  treatRefrectBoundary_F4<<<MPnum, THnum1D>>>(r, v, c0, c1, N);
}

void cudaParticleBase::addForceX(real fx) {
  const float4 _f = { fx, 0.0, 0.0, 0.0};
  addArray_F4<<<MPnum, THnum1D>>>(F, _f, N);
  if (withInfo) ErrorInfo("addForce in X");
}

void cudaParticleBase::addForceY(real fy) {
  const float4 _f = { 0.0, fy, 0.0, 0.0};
  addArray_F4<<<MPnum, THnum1D>>>(F, _f, N);
  if (withInfo) ErrorInfo("addForce in Y");
}

void cudaParticleBase::addForceZ(real fz) {
  const float4 _f = { 0.0, 0.0, fz, 0.0};
  addArray_F4<<<MPnum, THnum1D>>>(F, _f, N);
  if (withInfo) ErrorInfo("addForce in Z");
}

void cudaParticleBase::addAccelerationX(real ax) {
  const float4 _a = { ax, 0.0, 0.0, 0.0};
  addArray_F4<<<MPnum, THnum1D>>>(a, _a, N);
  if (withInfo) ErrorInfo("addAcceleration in X");
}

void cudaParticleBase::addAccelerationY(real ay) {
  const float4 _a = { 0.0, ay, 0.0, 0.0};
  addArray_F4<<<MPnum, THnum1D>>>(a, _a, N);
  if (withInfo) ErrorInfo("addAcceleration in Y");
}

void cudaParticleBase::addAccelerationZ(real az) {
  const float4 _a = { 0.0, 0.0, az, 0.0};
  addArray_F4<<<MPnum, THnum1D>>>(a, _a, N);
  if (withInfo) ErrorInfo("addAcceleration in Z");
}

int cudaParticleBase::inspectVelocity(real vlim, real lim_u, real lim_l, real &_r, uint32_t &_r1, bool debug) {
  if (debug) clearArray_F4<<<MPnum, THnum1D>>>(tmp3N, N+4);
  else clearArray_F4<<<MPnum, THnum1D>>>(tmp3N, 2);

  float2 thresh;
  thresh.x = lim_u; // upper lim
  thresh.y = lim_l; // lower lim
  inspectV_F4<<<MPnum, THnum1D>>>(v, N, vlim, tmp3N, thresh, debug);

  pthread_mutex_lock(&(mutTMP));

  if (debug) cudaMemcpy(&(TMP[0]), &(tmp3N[0]), sizeof(float)*(4+N), cudaMemcpyDeviceToHost);
  else cudaMemcpy(&(TMP[0]), &(tmp3N[0]), sizeof(float)*2, cudaMemcpyDeviceToHost);
/*
  std::cerr << TMP[1] << TMP[5] << TMP[9] << TMP[13]
            << TMP[17] << TMP[21] << TMP[25] << TMP[29] << std::endl;
*/

  // enhancement; add to debug out
  real T1 = TMP[0], T2 = TMP[1];
  if (debug) {
    auto r1 = std::max_element(&TMP[4], &TMP[4+N]); // largest *ratio*
    _r = *r1;
    _r1 = r1 - &(TMP[4]);
    // std::cerr << v1 << " " << vmax << " " << v1 / vmax << std::flush;
    // _v = v1; => _v * deltaT
    // important field is time, delta t, and the *ratio*
  }
  if (T1>0.0) {
    pthread_mutex_unlock(&(mutTMP));
    return 1;
  } else if (T2>0.0) {
    pthread_mutex_unlock(&(mutTMP));
    return 0;
  } else {
    pthread_mutex_unlock(&(mutTMP));
    return -1;
  }
}

void cudaParticleBase::dump3Narray(real *A, std::ostream &o) {
    pthread_mutex_lock(&mutTMP);
    cudaMemcpy(&(TMP[0]), A, sizeof(float4)*N, cudaMemcpyDeviceToHost);
    o << std::endl << std::endl;
    for (int i=0;i<N;++i)
      o << i << "\t" << TMP[i*4]
        << "\t" << TMP[i*4+1] 
        << "\t" << TMP[i*4+2] << std::endl;
    pthread_mutex_unlock(&mutTMP);
}
