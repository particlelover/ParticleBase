#include "cudaParticleMD.hh"
#include "kernelfuncs.h"
#include <assert.h>

#include "kerneltemplate.hh"

cudaParticleMD::~cudaParticleMD()
{
  if (m != NULL)
    cudaFree(m);

  if (hdl != NULL)
    cublasDestroy(hdl);
}

void cudaParticleMD::setup(int n)
{
  cudaParticleVV::setup(n);

  // alloc m 1/(1/mass)
  cudaMalloc((void **)&m, sizeof(real) * N);
  if (withInfo)
    ErrorInfo("malloc m[] on GPU");

  // preparation for CUBLAS
  if (hdl == NULL)
    cublasCreate(&hdl);
}

void cudaParticleMD::calcForce(void)
{
  clearArray<<<MPnum, THnum1D>>>(F, N * 3);

  dim3 _mpnum, _thnum;
  assert(_thnum.x * _thnum.y <= threadsMax);

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
#if defined(CUDACUTOFFBLOCK)
  _thnum.x = THnum2D2;
  _thnum.y = 1;
  _thnum.z = 1;
  _mpnum.x = myBlockNum;
  _mpnum.y = 27;
  _mpnum.z = 1;
  assert(myBlockNum < maxGrid);

  clearArray<<<MPnum, THnum1D>>>(tmp81N, N * 81);
  calcF_IJpairWithBlock<class MDpairPotential<potentialLJ>><<<_mpnum, _thnum>>>(P, r_s,
                                                                                tmp81N,
                                                                                myBlockOffset,
                                                                                blockNeighbor, pid, bindex,
                                                                                N,
                                                                                NULL, NULL, true);
  reduce27<<<MPnum, THnum1D>>>(F, tmp81N, N * 3);

  if (withInfo)
    ErrorInfo("cudaParticleMD::calcForce With Block");
#else

  calcF_IJpair<class MDpairPotential<potentialLJ>><<<MPnum, THnum1D>>>(P, r, F, N);
  //  calcF_IJpair<class potentialLJ><<<_mpnum, _thnum>>>(LJ, r, F, typeID, N, cell[6], cell[7], cell[8], rmax2);
  //  calcF_LJ<<<_mpnum, _thnum>>>(r, F, typeID, N, cx*cx, cy*cy, cz*cz);
  if (withInfo)
    ErrorInfo("cudaParticleMD::calcForce Without Block");
#endif
}

real cudaParticleMD::calcMV2(void)
{
  calcV2<<<MPnum, THnum1D>>>(v, tmp3N, N);
  if (withInfo)
    ErrorInfo("cudaParticleMD::calcV2");

  real t = 0;
  DOT(hdl, N, tmp3N, 1, m, 1, &t);
  if (withInfo)
    ErrorInfo("cudaParticleMD::DOT");

  return t;
}

void cudaParticleMD::setM(void)
{
  calcReciproc<<<MPnum, THnum1D>>>(minv, m, N);
  if (withInfo)
    ErrorInfo("calc reciprocal of minv to m");
}

real cudaParticleMD::constTemp(void)
{
  // calc kernel calculation first (before the CUBLAS funcs)
  real t = calcMV2();
  //std::cerr << "constTemp T= " << t << std::endl;

  real lambda = 0;
  DOT(hdl, N * 3, F, 1, v, 1, &lambda);
  if (lambda == 0)
    std::cerr << "[l=0]";

  lambda /= t; // \sum F_i v_i / \sum m_i v_i^2

  correctConstTemp<<<MPnum, THnum1D>>>(v, F, m, lambda, N);
  //  AXPY(hdl, N*3, &lambda, &(v[0]), 1, &(F[0]), 1);
  //  return lambda;
  return t / (3 * N * kB);
}
void cudaParticleMD::statMV2(std::ostream &o)
{
  calcMV2();

  pthread_mutex_lock(&(mutTMP));
  cudaMemcpy(&(TMP[0]), tmp3N, sizeof(real) * N, cudaMemcpyDeviceToHost);
  o << std::endl
    << std::endl;
  for (int i = 0; i < N; ++i)
    o << i << "\t" << TMP[i] << std::endl;
  pthread_mutex_unlock(&(mutTMP));
}

real cudaParticleMD::scaleTemp(real Temp)
{
  const real T0 = calcTemp();
  const real s = sqrt(static_cast<real>(3 * N - 1) / static_cast<real>(3 * N) * Temp / T0);

  mulArray<<<MPnum, THnum1D>>>(&(v[0]), s, N * 3);

  return s;
}

void cudaParticleMD::adjustVelocities(real Temp)
{
  // because array *a is not used in Velocity-Verlet, a is filled by 1
  clearArray<<<4, 256>>>(a, N * 3, 1.0);

  // calc v[i]*v[i] to tmp3N[i]
  calcV20<<<MPnum, THnum1D>>>(v, tmp3N, N);
  cudaThreadSynchronize();

  //std::valarray<real> _v(6, 1.0);
  real _v[6];
  DOT(hdl, N, &(v[0]), 1, &(a[0]), 1, &(_v[0]));
  DOT(hdl, N, &(v[N]), 1, &(a[N]), 1, &(_v[1]));
  DOT(hdl, N, &(v[N * 2]), 1, &(a[N * 2]), 1, &(_v[2]));
  DOT(hdl, N, &(tmp3N[0]), 1, &(a[0]), 1, &(_v[3]));
  DOT(hdl, N, &(tmp3N[N]), 1, &(a[N]), 1, &(_v[4]));
  DOT(hdl, N, &(tmp3N[N * 2]), 1, &(a[N * 2]), 1, &(_v[5]));
#if defined(DEBUG)
  std::cerr
      << _v[0] << ","
      << _v[1] << ","
      << _v[2] << ","
      << _v[3] << ","
      << _v[4] << ","
      << _v[5] << std::endl;
#endif

  //_v /= N;
  for (int i = 0; i < 6; ++i)
    _v[i] /= N;
  _v[3] -= _v[0] * _v[0];
  _v[4] -= _v[1] * _v[1];
  _v[5] -= _v[2] * _v[2];

  real m0 = 1;
  DOT(hdl, N, m, 1, a, 1, &m0);
  m0 /= N;
  real v1 = sqrt(kB * Temp / m0);
  //  std::cerr << m0 << ":" << v1 << "  " << std::flush;
  _v[3] = v1 / sqrt(_v[3]);
  _v[4] = v1 / sqrt(_v[4]);
  _v[5] = v1 / sqrt(_v[5]);

  addArray<<<1, THnum1D>>>(&(v[0]), -_v[0], N);
  addArray<<<1, THnum1D>>>(&(v[N]), -_v[1], N);
  addArray<<<1, THnum1D>>>(&(v[N * 2]), -_v[2], N);
  mulArray<<<1, THnum1D>>>(&(v[0]), _v[3], N);
  mulArray<<<1, THnum1D>>>(&(v[N]), _v[4], N);
  mulArray<<<1, THnum1D>>>(&(v[N * 2]), _v[5], N);

  cudaThreadSynchronize();
  std::cerr << "temp scaling by: "
            << _v[3] << ", " << _v[4] << ", " << _v[5] << std::endl;
}

void cudaParticleMD::setLJparams(const std::vector<real> &p, uint32_t elemnum)
{
  _setLJparams(p, elemnum);
}

void cudaParticleMD::initialAnnealing(uint32_t anealstep, real dt, real _rc, real _f0, real T)
{
  std::cerr << "performs initial annealing by soft core potential to rmax= " << _rc << " with steps " << anealstep << std::endl;

#if defined(CUDACUTOFFBLOCK)
  dim3 __mpnum, __thnum;
  __mpnum.x = myBlockNum;
  __mpnum.y = 27;
  __mpnum.z = 1;
  __thnum.x = THnum2D2;
  __thnum.y = 1;
  __thnum.z = 1;
  std::cerr << "Init Aneal with " << __mpnum.x << "x" << __mpnum.y
            << " blocks " << __thnum.x << "x" << __thnum.y << " threads" << std::endl;
  assert(myBlockNum < maxGrid);
#else
  std::cerr << "Init Aneal with " << MPnum
            << " blocks " << THnum1D << " threads" << std::endl;
#endif

  class potentialSC SC;
  SC.rc = _rc;
  SC.f0 = _f0;

  class MDpairPotential<potentialSC> P;
  P.cx = cell[6];
  P.cy = cell[7];
  P.cz = cell[8];
  P.c0x = sqrt(P.cx) * 2.0;
  P.c0y = sqrt(P.cy) * 2.0;
  P.c0z = sqrt(P.cz) * 2.0;
  P.rmax2 = rmax2;
  P.typeID = typeID;
  P.op = SC;

  for (uint32_t i = 0; i < anealstep; ++i)
  {
    clearArray<<<MPnum, THnum1D>>>(F, N * 3);
    if (withInfo)
      ErrorInfo("clear Forces");

    SC.rc = _rc * (i + 1) / anealstep;
    std::cerr << "\r" << SC.rc << "\t" << std::flush;

#if defined(CUDACUTOFFBLOCK)
    calcBlockID();
    clearArray<<<MPnum, THnum1D>>>(tmp81N, N * 81);
    P.op.rc = SC.rc;
    calcF_IJpairWithBlock<class MDpairPotential<potentialSC>>
        <<<__mpnum, __thnum>>>(P, r,
                               tmp81N,
                               myBlockOffset,
                               blockNeighbor, pid, bindex,
                               N);
    reduce27<<<MPnum, THnum1D>>>(F, tmp81N, N * 3);
#else
    P.op.rc = SC.rc;
    calcF_IJpair<class MDpairPotential<potentialSC>><<<MPnum, THnum1D>>>(P, r, F, N);
#endif
    if (withInfo)
      ErrorInfo("calc Forces by softcore");

    // correct force by velocity
    constTemp();

    calcA<<<MPnum, THnum1D>>>(a, minv, F, N);
    if (withInfo)
      ErrorInfo("calcAcceleration");

    propagateEuler<<<MPnum, THnum1D>>>(r, dt, v, a, move, N);
    if (withInfo)
      ErrorInfo("propagate by Euler");

    applyPeriodicCondition<<<MPnum, THnum1D>>>(r, cell[0], cell[1], N);
    applyPeriodicCondition<<<MPnum, THnum1D>>>(&(r[N]), cell[2], cell[3], N);
    applyPeriodicCondition<<<MPnum, THnum1D>>>(&(r[N * 2]), cell[4], cell[5], N);

    //adjustVelocities(MPnum, THnum1D, T);
    std::cerr << "T= " << calcTemp() << std::flush;
  }
  std::cerr << std::endl;

  std::cerr << "temperature scaling: " << scaleTemp(T) << std::endl;

  std::cerr << "with LJ " << std::endl;
  class potentialLJ LJ;
  clearArray<<<MPnum, THnum1D>>>(Fold, N * 3);

  class MDpairPotential<potentialLJ> P2;
  P2.cx = cell[6];
  P2.cy = cell[7];
  P2.cz = cell[8];
  P2.c0x = sqrt(P2.cx) * 2.0;
  P2.c0y = sqrt(P2.cy) * 2.0;
  P2.c0z = sqrt(P2.cz) * 2.0;
  P2.rmax2 = rmax2;
  P2.typeID = typeID;
  P2.op = LJ;

  for (uint32_t i = 0; i < anealstep; ++i)
  {
    std::cerr << i << "\t" << calcTemp() << "\t";
    clearArray<<<MPnum, THnum1D>>>(F, N * 3);

#if defined(CUDACUTOFFBLOCK)
    calcBlockID();
    clearArray<<<MPnum, THnum1D>>>(tmp81N, N * 81);
    calcF_IJpairWithBlock<class MDpairPotential<potentialLJ>>
        <<<__mpnum, __thnum>>>(P2, r,
                               tmp81N,
                               myBlockOffset,
                               blockNeighbor, pid, bindex,
                               N);
    reduce27<<<MPnum, THnum1D>>>(F, tmp81N, N * 3);
#else
    calcF_IJpair<class MDpairPotential<potentialLJ>><<<MPnum, THnum1D>>>(P2, r, F, N);
#endif
    std::cerr << constTemp() << std::endl;
    propagateVelocityVerlet<<<MPnum, THnum2D * THnum2D / 2>>>(r, dt, v, F, Fold, minv, N);
    //	std::cerr << scaleTemp(MPnum, THnum1D, T) << std::endl;
    applyPeriodicCondition<<<MPnum, THnum1D>>>(r, cell[0], cell[1], N);
    applyPeriodicCondition<<<MPnum, THnum1D>>>(&(r[N]), cell[2], cell[3], N);
    applyPeriodicCondition<<<MPnum, THnum1D>>>(&(r[N * 2]), cell[4], cell[5], N);
  }

  std::cerr << "temperature scaling: " << scaleTemp(T) << std::endl;
}
