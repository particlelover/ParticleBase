#include "kernelfuncs.h"
#include "kernelinline.hh"
#include <assert.h>
//#include <iostream>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)) || defined(__APPLE_CC__)
#undef assert
#define assert(arg)
#endif

/*
 * common procedures
 */
__global__ void addArray(real *r, real val, uint32_t num)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i = thID; i < num; i += thNum)
    r[i] += val;
}

__global__ void mulArray(real *r, real val, uint32_t num)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i = thID; i < num; i += thNum)
    r[i] *= val;
}

__global__ void calcReciproc(real *r, real *rinv, uint32_t num)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i = thID; i < num; i += thNum)
  {
    assert((r[i] != 0));
    rinv[i] = 1 / r[i];
  }
}

__global__ void multiplies(real *A, real *B, uint32_t num)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i = thID; i < num; i += thNum)
  {
    A[i] *= B[i];
  }
}

__global__ void calcA(real *a, real *minv, real *F, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    const real masinv = minv[i];
    a[i] = F[i] * masinv;
    a[i + N] = F[i + N] * masinv;
    a[i + N2] = F[i + N2] * masinv;
  }
}

/*
 * particleBase
 */
__global__ void applyPeriodicCondition(real *r, real c0, real c1, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const real cell = c1 - c0;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    //while (r[i]<c0) r[i] += cell;
    //while (r[i]>c1) r[i] -= cell;
    const real d0 = c0 - r[i];
    const real d1 = r[i] - c1;
    if (d0 > 0)
    {
      const signed int l0 = static_cast<signed int>(d0 / cell) + 1;
      r[i] += l0 * cell;
    }
    if (d1 > 0)
    {
      const signed int l1 = static_cast<signed int>(d1 / cell) + 1;
      r[i] -= l1 * cell;
    }
  }
}

__global__ void treatAbsoluteBoundary(real *r, real c0, real c1, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    if (r[i] < c0)
      r[i] = c0;
    if (r[i] > c1)
      r[i] = c1;
  }
}

__global__ void propagateEuler(real *r, real dt, real *v, real *a, char *move, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    const real dt2 = dt * move[i]; // move[i]: 0(fixed) or 1(other)
    r[i] += v[i] * dt2;
    r[i + N] += v[i + N] * dt2;
    r[i + N2] += v[i + N2] * dt2;

    v[i] += a[i] * dt2;
    v[i + N] += a[i + N] * dt2;
    v[i + N2] += a[i + N2] * dt2;

    assert(!isnan(r[i]));
    assert(!isnan(r[i + N]));
    assert(!isnan(r[i + N2]));
  }
}

__global__ void inspectV(real *v, uint32_t N, uint32_t vlim, real *tmp, real lim_u, real lim_l, bool debug)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    real _v[3] = {0.0, 0.0, 0.0};
    _v[0] = v[i];
    _v[1] = v[i + N];
    _v[2] = v[i + N2];

    _v[0] *= _v[0];
    _v[1] *= _v[1];
    _v[2] *= _v[2];

    const real vratio = sqrt(_v[0] + _v[1] + _v[2]) / vlim;
    //assert(vratio < 1.0);

    if (debug)
      tmp[i] = vratio;

    if (vratio >= lim_u)
    {
      tmp[N] = 1.0;
    }
    else if (vratio >= lim_l)
    {
      tmp[N + 1] = 1.0;
    }
  }
}

/*
 * particle LF
 */
__global__ void calcLFVinit(real *v, real dt_2, real *F, real *minv, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    const real masinv = minv[i] * dt_2;
    v[i] -= F[i] * masinv;
    v[i + N] -= F[i + N] * masinv;
    v[i + N2] -= F[i + N2] * masinv;
  }
}

__global__ void propagateLeapFrog(real *r, real dt, real *v, real *a, real *minv, char *move, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    const real dt2 = dt * move[i]; // move[i]: 0(fixed) or 1(other)
    v[i] += a[i] * dt2;
    v[i + N] += a[i + N] * dt2;
    v[i + N2] += a[i + N2] * dt2;
    // a(t)=F(t)/m => v(t+dt/2)

    r[i] += v[i] * dt2;
    r[i + N] += v[i + N] * dt2;
    r[i + N2] += v[i + N2] * dt2;
    // v(t+dt/2) => r(t+dt)
    assert(!isnan(r[i]));
    assert(!isnan(r[i + N]));
    assert(!isnan(r[i + N2]));
  }
}

__global__ void rollbackLeapFrog(real *r, real dt, real *v, real *a, real *minv, char *move, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    const real dt2 = dt * move[i]; // move[i]: 0(fixed) or 1(other)
    real _v[3] = {0.0, 0.0, 0.0};
    _v[0] = v[i];
    _v[1] = v[i + N];
    _v[2] = v[i + N2];

    real _r[3] = {0.0, 0.0, 0.0};
    // r(t+dt) => r(t)
    _r[0] = r[i] - _v[0] * dt2;
    _r[1] = r[i + N] - _v[1] * dt2;
    _r[2] = r[i + N2] - _v[2] * dt2;

    // v(t+dt/2) => v(t-dt/2)
    _v[0] -= a[i] * dt2;
    _v[1] -= a[i + N] * dt2;
    _v[2] -= a[i + N2] * dt2;

    v[i] = _v[0];
    v[i + N] = _v[1];
    v[i + N2] = _v[2];

    // r(t) => r(t-dt)
    r[i] = _r[0] - _v[0] * dt2;
    r[i + N] = _r[1] - _v[1] * dt2;
    r[i + N2] = _r[2] - _v[2] * dt2;

    assert(!isnan(r[i]));
    assert(!isnan(r[i + N]));
    assert(!isnan(r[i + N2]));
  }
}

/*
 * particleVV
 */
__global__ void propagateVelocityVerlet(real *r, real dt, real *v, real *F, real *Fold, real *minv, uint32_t N)
{
  /**
   * Velocity Verlet scheme of this code
   *
   * \f{eqnarray*}{
   * v(t)    &=& v(t-dt) + dt/m * (F(t)+F(t-dt))/2 \\
   * r(t+dt) &=& r(t) + dt * v(t) + dt^2/m * F(t)/2
   * \f}
   *
   */
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  const real dt2 = dt * 0.5;
  for (uint32_t i = thID; i < N; i += thNum)
  {
    const real dt_mass_2 = dt2 * minv[i]; // dt/mass/2
    v[i] += (F[i] + Fold[i]) * dt_mass_2;
    v[i + N] += (F[i + N] + Fold[i + N]) * dt_mass_2;
    v[i + N2] += (F[i + N2] + Fold[i + N2]) * dt_mass_2;

    const real dt22 = dt_mass_2 * dt;
    r[i] += v[i] * dt + F[i] * dt22;
    r[i + N] += v[i + N] * dt + F[i + N] * dt22;
    r[i + N2] += v[i + N2] * dt + F[i + N2] * dt22;

    Fold[i] = F[i];
    Fold[i + N] = F[i + N];
    Fold[i + N2] = F[i + N2];

    assert(!isnan(r[i]));
    assert(!isnan(r[i + N]));
    assert(!isnan(r[i + N2]));
  }
}

/*
 * GaussianThermo
 */
__global__ void calcGaussianThermoA1(real *A, real dt, real *v, real *F, real *Fold, real *minv, uint32_t N, real xi)
{
  // A(t) = v(t-dt)+(F(t)+F(t-dt))dt/2m - v(t-dt)xi(t-dt)dt/2
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  const real dt2 = dt * 0.5;
  for (uint32_t i = thID; i < N; i += thNum)
  {
    const real dt_mass_2 = dt2 * minv[i]; // dt/mass/2

    const real vx = v[i] + (F[i] + Fold[i]) * dt_mass_2;
    const real vy = v[i + N] + (F[i + N] + Fold[i + N]) * dt_mass_2;
    const real vz = v[i + N2] + (F[i + N2] + Fold[i + N2]) * dt_mass_2;

    const real xi2 = xi * dt * 0.5;

    A[i] = vx - v[i] * xi2;
    A[i + N] = vy - v[i + N] * xi2;
    A[i + N2] = vz - v[i + N2] * xi2;

    // initial term v0
    v[i] = vx;
    v[i + N] = vy;
    v[i + N2] = vz;
  }
}

__global__ void calcGaussianThermoFoverF(real *A, real dt, real *v, real *F, real *m, real *tmp3N, uint32_t N, real xi, real mv2inv)
{
  /**
   *
   * \f{eqnarray*}{
   * f(v)  &=& A - dt/2 * v * \xi - v \\
   * f'(v) &=& -dt/2[\xi + (F-2mv \xi)v / \sum mv^2] -1 \\
   * v &=& v - f(v)/f'(v)
   * \f}
   */
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  real dt2 = dt * 0.5;
  for (uint32_t i = thID; i < N * 3; i += thNum)
  {
    const uint32_t j = i % N;
    const real f1 = A[i] - (dt2 * xi + 1) * v[i];
    const real f2 = -dt2 * (xi + (F[i] - 2 * xi * m[j] * v[i]) * v[i] * mv2inv) - 1;

    tmp3N[i] = f1 / f2;
    v[i] -= f1 / f2;
  }
}

__global__ void propagateVelocityVerletGaussianThermo(real *r, real dt, real *v, real *F, real *Fold, real *minv, uint32_t N, real xi)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  const real dt2_2 = dt * dt * 0.5; // dt^2/2
  const real xidt2 = xi * dt2_2;
  for (uint32_t i = thID; i < N; i += thNum)
  {
    const real dt2_mass_2 = dt2_2 * minv[i]; // dt^2/mass/2

    r[i] += v[i] * (dt - xidt2) + F[i] * dt2_mass_2;
    r[i + N] += v[i + N] * (dt - xidt2) + F[i + N] * dt2_mass_2;
    r[i + N2] += v[i + N2] * (dt - xidt2) + F[i + N2] * dt2_mass_2;

    Fold[i] = F[i];
    Fold[i + N] = F[i + N];
    Fold[i + N2] = F[i + N2];

    assert(!isnan(r[i]));
    assert(!isnan(r[i + N]));
    assert(!isnan(r[i + N2]));
  }
}

/*
 * particleMD
 */
__global__ void calcV2(real *v, real *tmp3N, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    real _tmp = v[i] * v[i];
    _tmp += v[i + N] * v[i + N];
    _tmp += v[i + N2] * v[i + N2];
    tmp3N[i] = _tmp;
  }
}

__global__ void calcV20(real *v, real *tmp3N, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    tmp3N[i] = v[i] * v[i];
    tmp3N[i + N] = v[i + N] * v[i + N];
    tmp3N[i + N2] = v[i + N2] * v[i + N2];
  }
}
__global__ void correctConstTemp(real *v, real *F, real *m, real lambda, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    F[i] -= lambda * m[i] * v[i];
    F[i + N] -= lambda * m[i] * v[i + N];
    F[i + N2] -= lambda * m[i] * v[i + N2];
  }
}
/*
 * particleSPH
 */

/*
 * particleSPH_NS
 */
__global__ void calcF_SPH_NS(real *r, real *a, unsigned short *typeID, uint32_t N, real *dW2D, real *rhoinv, real *m, real *mu, real *v, const real K)
{
  /**
   * calculates acceleration a from r as 
   * \f[
   * a_i[x] -= -p1 * (r_j[x] - r_i[x]) + v1 * (v_j[x] - v_i[x])
   * \f]
   * where
   * \f{eqnarray*}{
   * p1 &=& m_j*(p_i/rho_i/rho_i + p_j/rho_j/rho_j) / r * dW/dR \\
   * v1 &=& m_j*(mu_i + mu_j) / rho_i / rho_j   / r * dW/dR
   * \f}
   * (p is calculated from number density n as p = Kn, so
   *	\f$ p/rho/rho = Krho/rho/rho = K/rho; p1 = m_j K (1/rho_i + 1/rho_j) * dW/dR /r \f$
   *
   * dW/dR, and n are calculated by calcKernels() and calcDensity()
   */

  /* i-j pair are divided into (blockDim.x*4) x (blockDim.y*4) tiling
   * in each tile, threads checks all grid
   * cuda block X and Y corresponds to this tile
   */
  uint32_t istart = (blockDim.x * 4) * blockIdx.x;
  uint32_t jstart = (blockDim.y * 4) * blockIdx.y;
  uint32_t iend = istart + (blockDim.x * 4);
  uint32_t jend = jstart + (blockDim.y * 4);
  if (iend > N)
    iend = N;
  if (jend > N)
    jend = N;
  uint32_t N2 = N * 2;

  for (uint32_t i = istart + threadIdx.x; i < iend; i += blockDim.x)
  { // 4times loop
    real _r[3];
    _r[0] = r[i];
    _r[1] = r[i + N];
    _r[2] = r[i + N2];
    for (uint32_t j = jstart + threadIdx.y; j < jend; j += blockDim.y)
    { // 4times loop
      if (i != j)
      {

        // p1 = K (1/n_i + 1/n_j) * dW/dR
        // v1 = (mu_i + mu_j) / n_i / n_j   * dW/dR
        //const real p1 = K * (ninv[i] + ninv[j]) * dW2D[i+j*N];
        //const real v1 = (mu[i]+mu[j]) * ninv[i] * ninv[j] * dW2D[i+j*N];
        const real p1 = m[j] * K * ((1 + typeID[i] * 4) * rhoinv[i] + (1 + typeID[j] * 4) * rhoinv[j]) * dW2D[i + j * N];
        const real v1 = m[j] * (mu[i] + mu[j]) * rhoinv[i] * rhoinv[j] * dW2D[i + j * N];

        // F_i[x] += -p1*r[x] + v1 * (v_i[x] - v_j[x])

        real _f[3];
        _f[0] = r[j] - _r[0];
        _f[1] = r[j + N] - _r[1];
        _f[2] = r[j + N2] - _r[2];

        a[i] -= -_f[0] * p1 + (v[j] - v[i]) * v1;
        a[i + N] -= -_f[1] * p1 + (v[j + N] - v[i + N]) * v1;
        a[i + N2] -= -_f[2] * p1 + (v[j + N2] - v[i + N2]) * v1;
      }
    }
  }
}

__global__ void inspectDense(real *n, char *move, uint32_t N, real *R)
{
  const uint32_t thID = threadIdx.x;
  const uint32_t thNum = blockDim.x;

  extern __shared__ real tmp_r[];
  uint32_t *tmp_n = (uint32_t *)(&tmp_r[thNum]);
  tmp_r[thID] = 0.0;
  tmp_n[thID] = 0;
  for (uint32_t i = thID; i < N; i += thNum)
  {
    if (move[i] > 0)
    {
      tmp_n[thID] += 1;
      tmp_r[thID] += n[i];
    }
  }
  __syncthreads();

  uint32_t k = 1;
  do
  {
    const uint32_t J = thID * k * 2;
    if (J + k < thNum)
    {
      tmp_n[J] += tmp_n[J + k];
      tmp_r[J] += tmp_r[J + k];
    }
    k *= 2;
  } while (k < thNum);

  if (thID == 0)
    R[0] = tmp_r[thID] / tmp_n[thID];

  return;
}

/*
 * used from cudaCutoffBlock
 */
__global__ void calcBID(real *r, uint32_t *bid, uint32_t *pid, uint32_t N, real b0, real b1, real b2,
                        real cxmin, real cymin, real czmin,
                        uint32_t CX, uint32_t CY)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t _i = thID; _i < N; _i += thNum)
  {
    const uint32_t i = pid[_i];
    const uint32_t d0 = static_cast<uint32_t>((r[i] - cxmin) / b0);
    const uint32_t d1 = static_cast<uint32_t>((r[i + N] - cymin) / b1);
    const uint32_t d2 = static_cast<uint32_t>((r[i + N2] - czmin) / b2);

    bid[_i] = d2 + d1 * CX + d0 * CY;
    //    pid[i] = i;
  }
}

__global__ void sortByBID(uint32_t *pid, uint32_t *bid, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  __shared__ uint32_t d; // should be global

  do
  {
    d = 0;
    for (uint32_t i = thID * 2; i < N - 1; i += thNum * 2)
    {
      if (bid[i] > bid[i + 1])
      {
        const uint32_t b = bid[i + 1];
        const uint32_t c = pid[i + 1];
        bid[i + 1] = bid[i];
        pid[i + 1] = pid[i];
        bid[i] = b;
        pid[i] = c;
        //	bid[i] = atomicExch(&bid[i+1], bid[i]);
        //	pid[i] = atomicExch(&pid[i+1], pid[i]);
        d = 1;
      }
    }

    for (uint32_t i = thID * 2 + 1; i < N - 1; i += thNum * 2)
    {
      if (bid[i] > bid[i + 1])
      {
        const uint32_t b = bid[i + 1];
        const uint32_t c = pid[i + 1];
        bid[i + 1] = bid[i];
        pid[i + 1] = pid[i];
        bid[i] = b;
        pid[i] = c;
        //	bid[i] = atomicExch(&bid[i+1], bid[i]);
        //	pid[i] = atomicExch(&pid[i+1], pid[i]);
        d = 1;
      }
    }
    __syncthreads();
  } while (d > 0);
}

__global__ void makeBindex(uint32_t *bid, uint32_t *bindex, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i = thID; i < N - 1; i += thNum)
  {
    const uint32_t c1 = bid[i];
    const uint32_t c2 = bid[i + 1];
    if (c1 != c2)
    {
      // block boundary in i|i+1; block c2 starts from i+1
      bindex[c2] = i + 1;
    }
    //    bindex[end] = N;
  }
  if (thID == 0)
    bindex[0] = 0;
}

__global__ void sortByBID_M1(uint32_t *pid, uint32_t *bid, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t startI = thID + thNum; startI < N; startI += thNum)
  {

    /** gnome sort
     *
    uint32_t I = startI;
    while ((I>N)&&(bid[I]<bid[I-thNum])) {
      const uint32_t  b = bid[I];
      const uint32_t c = pid[I];
      bid[I] = bid[I-thNum];
      pid[I] = pid[I-thNum];
      bid[I-thNum] = b;
      pid[I-thNum] = c;
      I-=thNum;
    }
    */

    // insertion sort
    while (bid[startI - thNum] > bid[startI])
    {
      uint32_t I = startI - thNum;
      const uint32_t tmpbid = bid[startI];
      const uint32_t tmppid = pid[startI];

      while ((I > thNum) && (bid[I - thNum] > tmpbid))
        I -= thNum;
      bid[startI] = bid[I];
      pid[startI] = pid[I];
      bid[I] = tmpbid;
      pid[I] = tmppid;
    }
  }
}

__global__ void reduce27(real *A, real *A27, uint32_t num, uint32_t blocksize, uint32_t offset)
{
  if (blocksize == 0)
    blocksize = num;
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i = thID; i < num; i += thNum)
  {
    real T = A[i];
    for (int b = 0; b < 27; ++b)
    {
      const real *const V = &(A27[blocksize * b + offset]);
      T += V[i];
    }
    A[i] = T;
  }
}

__global__ void RestoreByPid(real *dst, real *src, uint32_t N, uint32_t *pid)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    dst[pid[i]] = src[i];
    dst[pid[i] + N] = src[i + N];
    dst[pid[i] + N2] = src[i + N2];
  }
}

__global__ void addArray(real *A, real *tmp3N, uint32_t N)
{
  const uint32_t N2 = N * 2;

  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i = thID; i < N; i += thNum)
  {
    A[i] += tmp3N[i];
    A[i + N] += tmp3N[i + N];
    A[i + N2] += tmp3N[i + N2];
  }
}

/*
 * cudaParticleRotation
 */
__global__ void propagateEulerianRotation(real *w, real dt, real *L, real *T, real *m0inv, char *move, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    real dt2 = dt * move[i]; // move[i]: 0(fixed) or 1(other)
    real m0 = m0inv[i] * move[i];
    L[i] += T[i] * dt2;
    L[i + N] += T[i + N] * dt2;
    L[i + N2] += T[i + N2] * dt2;

    w[i] = L[i] * m0;
    w[i + N] = L[i + N] * m0;
    w[i + N2] = L[i + N2] * m0;
  }
}

/*
 * cudaSelectedBlock
 */
__global__ void checkBlocks(uint32_t *selected, uint32_t *pid, char *move,
                            uint32_t *bindex, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    if (bindex[i] != UINT_MAX)
    {
      long __I = i + 1;
      while (bindex[__I] == UINT_MAX)
        ++__I;
      const uint32_t bendI = bindex[__I];
      for (uint32_t i0 = bindex[i] + threadIdx.y; i0 < bendI; i0 += blockDim.y)
      {
        const uint32_t p = pid[i0];
        if (move[p] == 1)
        {
          selected[i] = 1;
        }
      }
    }
  }
}

__global__ void accumulate(uint32_t *selected, uint32_t N, real *Res)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  extern __shared__ uint32_t tmp_s[];
  tmp_s[thID] = 0;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    tmp_s[thID] += selected[i];
  }

  __syncthreads();
  uint32_t k = 1;
  do
  {
    const uint32_t J = thID * k * 2;
    if (J + k < thNum)
    {
      tmp_s[J] += tmp_s[J + k];
    }
    k *= 2;
  } while (k < thNum);

  if (thID == 0)
    Res[0] = static_cast<real>(tmp_s[0]);
}

__global__ void writeBlockID(uint32_t *selected, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    selected[i] *= i;
  }
}

/*
 * cudaSparseMat
 */
__global__ void real2ulong(real *r, uint32_t *l, uint32_t N, uint32_t *pid)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  if (pid != NULL)
  {
    for (uint32_t i = thID; i < N; i += thNum)
    {
      l[pid[i]] = static_cast<uint32_t>(r[i]);
    }
  }
  else
  {
    for (uint32_t i = thID; i < N; i += thNum)
    {
      l[i] = static_cast<uint32_t>(r[i]);
    }
  }
}
__global__ void partialsum(uint32_t *rowPtr, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  extern __shared__ uint32_t subarray[];

  subarray[thID] = 0;

  uint32_t d = static_cast<uint32_t>(N / thNum);
  //  if (d*thNum<N) ++d;
  uint32_t *const _row = &(rowPtr[d * thID]);
  if (thID == thNum - 1)
    d = N - (d * (thNum - 1));

  //  uint32_t _N=_row[0];
  for (uint32_t i = 1; i < d; ++i)
  {
    //    _N += _row[i];
    //    _row[i] = N;
    _row[i] += _row[i - 1];
  }
  subarray[thID + 1] = _row[d - 1];
  __syncthreads();

  if (thID == 0)
    for (uint32_t i = 1; i < thNum; ++i)
      subarray[i] += subarray[i - 1];
  __syncthreads();

  const uint32_t D = subarray[thID];
  for (uint32_t i = 0; i < d; ++i)
  {
    _row[i] += D;
  }
}

__global__ void makeJlist_WithBlock(uint32_t *rowPtr, uint32_t *colIdx,
                                    real *r0,
                                    real *r,
                                    uint32_t BlockOffset,
                                    uint32_t *blockNeighbor, uint32_t *pid, uint32_t *bindex,
                                    uint32_t N,
                                    uint32_t *selectedBlock)
{
  /*
   * cuda block ID for X and Y indicates ID of cutoff block I and J.
   * threads i, j sweeps particles in I and J.
   *
   */
  const uint32_t I =
      (selectedBlock != NULL) ? selectedBlock[blockIdx.x + BlockOffset] : blockIdx.x + BlockOffset;

  if (bindex[I] == UINT_MAX)
    return;

  const uint32_t N2 = N * 2;

  int __I = I + 1;
  while (bindex[__I] == UINT_MAX)
    ++__I;
  const uint32_t bendI = bindex[__I];

  for (uint32_t i0 = bindex[I] + threadIdx.x; i0 < bendI; i0 += blockDim.x)
  {
    const uint32_t i = pid[i0];

    const real _r[3] = {r[i], r[i + N], r[i + N2]};
    const real r0i = r0[i];

    if (rowPtr[i + 1] != rowPtr[i])
    {
      uint32_t *col = &(colIdx[rowPtr[i]]);

      // loop for all j particle in 27Blocks
      for (int ___J = 0; ___J < 27; ___J++)
      {
        const uint32_t J = blockNeighbor[I * 27 + ___J];
        if (J == UINT_MAX)
          continue;
        if (bindex[J] == UINT_MAX)
          continue;

        int __J = J + 1;
        while (bindex[__J] == UINT_MAX)
          ++__J;
        const uint32_t bendJ = bindex[__J];

        for (uint32_t j0 = bindex[J]; j0 < bendJ; ++j0)
        {
          const uint32_t j = pid[j0];

          if (i != j)
          {
            // register if i--j is enough close
            const real R2 = distance2(_r[0], _r[1], _r[2], r[j], r[j + N], r[j + N2]);
            real rc2 = r0i + r0[j];
            if (R2 <= rc2 * rc2)
            {
              *col++ = j;
            }
          }
        }
      }
    }
  }
}

__global__ void makeJlist_WithBlock2(const uint32_t *rowPtr, uint32_t *colIdx,
                                     const real *r0,
                                     const real *r,
                                     const uint32_t *blockNeighbor, const uint32_t *pid, const uint32_t *bindex,
                                     const uint32_t *bid_by_pid, const char *move,
                                     const uint32_t N, const uint32_t Nstart, const uint32_t Nend)
{
  /*
   * i-particles are selected in 1D thread parallel,
   * block ID of I-block was calculated from the position of i-particle and then
   * neighboring J-blocks are treated in serial
   */
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x + Nstart;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;
  const uint32_t Nmax = N - Nend;

  for (uint32_t i = thID; i < Nmax; i += thNum)
  {
    const int I = bid_by_pid[i];
    if (move[i] > 0)
    {
      const real _r[3] = {r[i], r[i + N], r[i + N2]};
      const real r0i = r0[i];

      if (rowPtr[i + 1] != rowPtr[i])
      {
        uint32_t *col = &(colIdx[rowPtr[i]]);

        for (int J27 = 0; J27 < 27; ++J27)
        {
          const int J = blockNeighbor[I * 27 + J27];
          if (J != UINT_MAX)
          {
            if (bindex[J] != UINT_MAX)
            {
              int __J = J + 1;
              while (bindex[__J] == UINT_MAX)
                ++__J;
              const uint32_t bendJ = bindex[__J];
              for (uint32_t j0 = bindex[J]; j0 < bendJ; ++j0)
              {
                const uint32_t j = pid[j0];

                if (i != j)
                {
                  // register if i--j is enough close
                  const real R2 = distance2(_r[0], _r[1], _r[2], r[j], r[j + N], r[j + N2]);
                  const real rc2 = r0i + r0[j];
                  if (R2 <= rc2 * rc2)
                  {
                    *col++ = j;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

__global__ void makeJlist_WithBlock4(const uint32_t *rowPtr, uint32_t *colIdx,
                                     const real *r0,
                                     const real *r,
                                     const uint32_t *blockNeighbor, const uint32_t *bindex,
                                     const uint32_t *bid, const char *move,
                                     const uint32_t N,
                                     const uint32_t Nstart, const uint32_t Nend)
{
  /*
   * i-particles are selected in 1D thread parallel,
   * block ID of I-block was calculated from the position of i-particle and then
   * neighboring J-blocks are treated in serial
   */
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x + Nstart;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;
  const uint32_t Nmax = N - Nend;

  for (uint32_t i = thID; i < Nmax; i += thNum)
  {
    const long I = bid[i];
    assert(bindex[I] == i);
    if (move[i] > 0)
    {
      const real _r[3] = {r[i], r[i + N], r[i + N2]};
      const real r0i = r0[i];

      if (rowPtr[i + 1] != rowPtr[i])
      {
        uint32_t *col = &(colIdx[rowPtr[i]]);

        for (int J27 = 0; J27 < 125; ++J27)
        {
          const long J = blockNeighbor[I * 125 + J27];
          if (J != UINT_MAX)
          {
            if (bindex[J] != UINT_MAX)
            {
              const uint32_t j = bindex[J];
              if (i != j)
              {
                // register if i--j is enough close
                const real R2 = distance2(_r[0], _r[1], _r[2], r[j], r[j + N], r[j + N2]);
                const real rc2 = r0i + r0[j];
                if (R2 <= rc2 * rc2)
                {
                  *col++ = j;
                }
              }
            }
          }
        }
      }
    }
  }
}

__global__ void calcBID_direct(const real *r, uint32_t *bid, uint32_t *bindex,
                               const uint32_t N, const real b0, const real b1, const real b2,
                               const real cxmin, const real cymin, const real czmin,
                               const uint32_t blen0, const uint32_t blen1, const uint32_t blen2,
                               uint32_t CX, uint32_t CY)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    signed long d0 = static_cast<signed long>((r[i] - cxmin) / b0);
    signed long d1 = static_cast<signed long>((r[i + N] - cymin) / b1);
    signed long d2 = static_cast<signed long>((r[i + N2] - czmin) / b2);
    assert(d0 >= -1);
    assert(d1 >= -1);
    assert(d2 >= -1);
    assert(d0 <= blen0);
    assert(d1 <= blen1);
    assert(d2 <= blen2);
    if (d0 == blen0)
      --d0;
    if (d1 == blen1)
      --d1;
    if (d2 == blen2)
      --d2;
    if (d0 == -1)
      ++d0;
    if (d1 == -1)
      ++d1;
    if (d2 == -1)
      ++d2;

    const uint32_t BID = d2 + d1 * CX + d0 * CY;
    assert(bindex[BID] == UINT_MAX);

    bid[i] = BID;
    bindex[BID] = i;
  }
}

__global__ void makeBIDbyPID(const uint32_t *pid, uint32_t *bid, uint32_t *bid_by_pid,
                             const uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t _i = thID; _i < N; _i += thNum)
  {
    bid_by_pid[pid[_i]] = bid[_i];
  }
}

__global__ void sortColIdx(const uint32_t *rowPtr, uint32_t *colIdx, uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    // sort between rowPtr[i]..rowPtr[i+1]
    const uint32_t jbegin = rowPtr[i];
    const uint32_t jend = rowPtr[i + 1];
    const uint32_t gap = jend - jbegin;

    if (gap < 2)
      return;
    else if (gap == 2)
    {
      if (colIdx[jbegin] > colIdx[jbegin + 1])
      {
        const uint32_t _j = colIdx[jbegin];
        colIdx[jbegin] = colIdx[jbegin + 1];
        colIdx[jbegin + 1] = _j;
      }
    }
    else
    {
      for (int __I = jbegin + 1; __I < jend; ++__I)
      {
        while ((__I > jbegin) && (colIdx[__I - 1] > colIdx[__I]))
        {
          const uint32_t __J = colIdx[__I - 1];
          colIdx[__I - 1] = colIdx[__I];
          colIdx[__I] = __J;
          --__I;
        }
      }
    }
  }
}

__global__ void succeedPrevState(uint32_t *prevRow, uint32_t *prevCol, real *prevVal,
                                 uint32_t *curRow, uint32_t *curCol, real *curVal,
                                 uint32_t N)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i = thID; i < N; i += thNum)
  {
    uint32_t jx = curRow[i];
    const uint32_t _j = curRow[i + 1];
    for (uint32_t j = prevRow[i]; j < prevRow[i + 1]; ++j)
    {
      const uint32_t j1 = prevCol[j];

      uint32_t j2 = curCol[jx];
      while ((j2 < j1) && (jx < _j))
        j2 = curCol[++jx];
      if (j2 == j1)
        curVal[jx] = prevVal[j];
    }
  }
}
