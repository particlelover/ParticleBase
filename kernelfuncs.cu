#include "kernelfuncs.h"
#include "kernelinline.hh"
#include <assert.h>
//#include <iostream>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)) || defined(__APPLE_CC__)
# undef   assert
# define  assert(arg)
#endif


/*
 * common procedures
 */
__global__ void clearArray_F4(float4 *r, uint32_t num, float4 val) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i=thID;i<num;i+=thNum) {
    float4 _r = r[i];
    _r.x = val.x;
    _r.y = val.y;
    _r.z = val.z;
    _r.w = val.w;
    r[i] = _r;
  }
}

__global__ void addArray_F4(float4 *r, float4 val, uint32_t num) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i=thID;i<num;i+=thNum) {
    float4 _r = r[i];
    _r.x += val.x;
    _r.y += val.y;
    _r.z += val.z;
    _r.w += val.w;
    r[i] = _r;
  }
}

__global__ void addArray_F4(float4 *r, float4 *val, uint32_t num) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i=thID;i<num;i+=thNum) {
    float4 _r = r[i];
    _r.x += val[i].x;
    _r.y += val[i].y;
    _r.z += val[i].z;
    _r.w += val[i].w;
    r[i] = _r;
  }
}

__global__ void calcReciproc(real *r, real *rinv, uint32_t num) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i=thID;i<num;i+=thNum) {
    assert((r[i]!=0));
    rinv[i] = 1/r[i];
  }
}

__global__ void multiplies(real *A, real *B, uint32_t num) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i=thID;i<num;i+=thNum) {
    A[i] *= B[i];
  }
}

__global__ void calcA_F4(float4 *a, real *minv, float4 *F, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    const real masinv = minv[i];
    float4 A = a[i];
    A.x = F[i].x * masinv;
    A.y = F[i].y * masinv;
    A.z = F[i].z * masinv;
    a[i] = A;
  }
}

/*
 * particleBase
 */
__global__ void applyPeriodicCondition_F4(float4 *r, float3 c0, float3 c1, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  float3 cell;
  cell.x = c1.x - c0.x;
  cell.y = c1.y - c0.y;
  cell.z = c1.z - c0.z;

  for (uint32_t i=thID;i<N;i+=thNum) {
    //while (r[i]<c0) r[i] += cell;
    //while (r[i]>c1) r[i] -= cell;
    float4 _r = r[i];
    float d0, d1;
    d0 = c0.x - _r.x;
    if (d0>0) {
      const signed int l0 = static_cast<signed int>(d0 / cell.x) + 1;
      _r.x += l0 * cell.x;
    }
    d1 = _r.x - c1.x;
    if (d1>=0) {
      const signed int l1 = static_cast<signed int>(d1 / cell.x) + 1;
      _r.x -= l1 * cell.x;
    }
    d0 = c0.y - _r.y;
    if (d0>0) {
      const signed int l0 = static_cast<signed int>(d0 / cell.y) + 1;
      _r.y += l0 * cell.y;
    }
    d1 = _r.y - c1.y;
    if (d1>=0) {
      const signed int l1 = static_cast<signed int>(d1 / cell.y) + 1;
      _r.y -= l1 * cell.y;
    }
    d0 = c0.z - _r.z;
    if (d0>0) {
      const signed int l0 = static_cast<signed int>(d0 / cell.z) + 1;
      _r.z += l0 * cell.z;
    }
    d1 = _r.z - c1.z;
    if (d1>=0) {
      const signed int l1 = static_cast<signed int>(d1 / cell.z) + 1;
      _r.z -= l1 * cell.z;
    }
    assert(c0.x <= _r.x);
    assert(_r.x < c1.x);
    assert(c0.y <= _r.y);
    assert(_r.y < c1.y);
    assert(c0.z <= _r.z);
    assert(_r.z < c1.z);

    r[i] = _r;
  }
}

__global__ void treatAbsoluteBoundary_F4(float4 *r, float4 c0, float4 c1, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    float4 _r = r[i];
    if (_r.x < c0.x) _r.x = c0.x;
    if (_r.x > c1.x) _r.x = c1.x;
    if (_r.y < c0.y) _r.y = c0.y;
    if (_r.y > c1.y) _r.y = c1.y;
    if (_r.z < c0.z) _r.z = c0.z;
    if (_r.z > c1.z) _r.z = c1.z;
    r[i] = _r;
  }
}

__global__ void treatRefrectBoundary_F4(float4 *r, float4 *v, float4 c0, float4 c1, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    float4 _r = r[i];
    float4 _v = v[i];
    if (_r.x < c0.x) _v.x =  abs(_v.x);
    if (_r.x > c1.x) _v.x = -abs(_v.x);
    if (_r.y < c0.y) _v.y =  abs(_v.y);
    if (_r.y > c1.y) _v.y = -abs(_v.y);
    if (_r.z < c0.z) _v.z =  abs(_v.z);
    if (_r.z > c1.z) _v.z = -abs(_v.z);
    v[i] = _v;
  }
}

__global__ void propagateEuler_F4(float4 *r, real dt, float4 *v, float4 *a, char *move, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    const real dt2 = dt * move[i];  // move[i]: 0(fixed) or 1(other)
    float4 _v = v[i];
    float4 _r = r[i];
    _r.x += _v.x * dt2;
    _r.y += _v.y * dt2;
    _r.z += _v.z * dt2;

    _v.x += a[i].x * dt2;
    _v.y += a[i].y * dt2;
    _v.z += a[i].z * dt2;
    v[i] = _v;
    r[i] = _r;
  }
}

__global__ void inspectV_F4(float4 *v, uint32_t N, uint32_t vlim, float4 *tmp,
    float2 thresh, bool debug) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID*4;i<N;i+=thNum*4) {
    for (uint32_t j=0;j<4;++j) {
      if (i+j < N) {
        real _v[3] = {0.0, 0.0, 0.0};
        _v[0] = v[i+j].x;
        _v[1] = v[i+j].y;
        _v[2] = v[i+j].z;

        _v[0] *= _v[0];
        _v[1] *= _v[1];
        _v[2] *= _v[2];

        const real vratio = sqrt(_v[0]+_v[1]+_v[2]) / vlim;
        //assert(vratio < 1.0);

        if (debug) {
          switch(j) {
            case 0:
            tmp[i/4+1].x = vratio;
            break;
            case 1:
            tmp[i/4+1].y = vratio;
            break;
            case 2:
            tmp[i/4+1].z = vratio;
            break;
            case 3:
            tmp[i/4+1].w = vratio;
            break;
          }
        }
        if (vratio >= thresh.x) {
          // too large
          tmp[0].x = 1.0;
        } else if (vratio >= thresh.y) {
          // too small
          tmp[0].y = 1.0;
        }
      }
    }
  }
}


/*
 * particle LF
 */
__global__ void calcLFVinit_F4(float4 *v, real dt_2, float4 *F, real *minv, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    const real masinv = minv[i] * dt_2;
    float4 V = v[i];
    V.x -= F[i].x * masinv;
    V.y -= F[i].y * masinv;
    V.z -= F[i].z * masinv;
    v[i] = V;
  }
}

__global__ void propagateLeapFrog_F4(float4 *r, real dt, float4 *v, float4 *a,
    real *minv, char *move, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    const real dt2 = dt * move[i];  // move[i]: 0(fixed) or 1(other)
    float4 V = v[i];
    V.x += a[i].x * dt2;
    V.y += a[i].y * dt2;
    V.z += a[i].z * dt2;
    v[i] = V;
    // a(t)=F(t)/m => v(t+dt/2)

    float4 R = r[i];
    R.x += V.x * dt2;
    R.y += V.y * dt2;
    R.z += V.z * dt2;
    r[i] = R;
    // v(t+dt/2) => r(t+dt)
    assert(!isnan(r[i].x));
    assert(!isnan(r[i].y));
    assert(!isnan(r[i].z));
  }
}

__global__ void rollbackLeapFrog_F4(float4 *r, real dt, float4 *v, float4 *a,
    real *minv, char *move, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    const real dt2 = dt * move[i];  // move[i]: 0(fixed) or 1(other)
    real _v[3] = {0.0, 0.0, 0.0};
    _v[0] = v[i].x;
    _v[1] = v[i].y;
    _v[2] = v[i].z;

    real _r[3] = {0.0, 0.0, 0.0};
    // r(t+dt) => r(t)
    _r[0] = r[i].x - _v[0] * dt2;
    _r[1] = r[i].y - _v[1] * dt2;
    _r[2] = r[i].z - _v[2] * dt2;

    // v(t+dt/2) => v(t-dt/2)
    _v[0] -= a[i].x * dt2;
    _v[1] -= a[i].y * dt2;
    _v[2] -= a[i].z * dt2;

    float4 tmp4;
    tmp4.x = _v[0];
    tmp4.y = _v[1];
    tmp4.z = _v[2];
    tmp4.w = v[i].w;
    v[i] = tmp4;

    // r(t) => r(t-dt)
    tmp4.x = _r[0] - _v[0] * dt2;
    tmp4.y = _r[1] - _v[1] * dt2;
    tmp4.z = _r[2] - _v[2] * dt2;
    tmp4.w = r[i].w;
    r[i] = tmp4;

    assert(!isnan(r[i].x));
    assert(!isnan(r[i].y));
    assert(!isnan(r[i].z));
  }
}


/*
 * particleVV
 */
__global__ void propagateVelocityVerlet_F4(float4 *r, real dt, float4 *v, float4 *F,
  float4 *Fold, real *minv, uint32_t N) {
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

  const real dt2 = dt * 0.5;
  for (uint32_t i=thID;i<N;i+=thNum) {
    const real dt_mass_2 = dt2 * minv[i]; // dt/mass/2
    float4 _F = F[i];
    float4 _v = v[i];
    _v.x += (_F.x + Fold[i].x) * dt_mass_2;
    _v.y += (_F.y + Fold[i].y) * dt_mass_2;
    _v.z += (_F.z + Fold[i].z) * dt_mass_2;
    _v.w = 0;

    const real dt22 = dt_mass_2 * dt;
    float4 _r = r[i];
    _r.x += _v.x * dt + _F.x * dt22;
    _r.y += _v.y * dt + _F.y * dt22;
    _r.z += _v.z * dt + _F.z * dt22;

    Fold[i] = _F;
    v[i] = _v;
    r[i] = _r;
  }
}

/*
 * GaussianThermo
 */
__global__ void calcGaussianThermoA1_F4(double *A, real dt, double *vd, float4 *F,
  float4 *Fold, real *minv, uint32_t N, double xi) {
  // A(t) = v(t-dt)+(F(t)+F(t-dt))dt/2m - v(t-dt)xi(t-dt)dt/2
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  const real dt2 = dt * 0.5;
  const uint32_t N2 = N*2;
  for (uint32_t i=thID;i<N;i+=thNum) {
    const real dt_mass_2 = dt2 * minv[i]; // dt/mass/2

    double _vx, _vy, _vz;
    _vx = vd[i]    + (F[i].x+ Fold[i].x) * dt_mass_2;
    _vy = vd[i+N]  + (F[i].y+ Fold[i].y) * dt_mass_2;
    _vz = vd[i+N2] + (F[i].z+ Fold[i].z) * dt_mass_2;

    const real xi2 = xi * dt * 0.5;

    double _Ax, _Ay, _Az;
    _Ax = _vx - vd[i]    * xi2;
    _Ay = _vy - vd[i+N]  * xi2;
    _Az = _vz - vd[i+N2] * xi2;

    // initial term v0
    A[i]    = _Ax;
    A[i+N]  = _Ay;
    A[i+N2] = _Az;
    vd[i]    = _vx;
    vd[i+N]  = _vy;
    vd[i+N2] = _vz;
  }
}

__global__ void calcGaussianThermoFoverF_F4(double *A, real dt, double *vd, float4 *F,
    real *m, double *tmp3N, uint32_t N, double xi, real mv2inv) {
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

  const real dt2 = dt * 0.5;
  const uint32_t N2 = N*2;
  for (uint32_t i=thID;i<N;i+=thNum) {
    double _vx, _vy, _vz;
    _vx = vd[i];
    _vy = vd[i+N];
    _vz = vd[i+N2];
    double f1x, f1y, f1z;
    f1x = A[i];
    f1y = A[i+N];
    f1z = A[i+N2];
    f1x -= (dt2 * xi + 1) * _vx;
    f1y -= (dt2 * xi + 1) * _vy;
    f1z -= (dt2 * xi + 1) * _vz;
    double f2x, f2y, f2z;
    f2x = - dt2 * (xi + (F[i].x - 2 * xi * m[i] * _vx)*_vx * mv2inv) -1;
    f2y = - dt2 * (xi + (F[i].y - 2 * xi * m[i] * _vy)*_vy * mv2inv) -1;
    f2z = - dt2 * (xi + (F[i].z - 2 * xi * m[i] * _vz)*_vz * mv2inv) -1;

    f1x /= f2x;
    f1y /= f2y;
    f1z /= f2z;
    _vx -= f1x;
    _vy -= f1y;
    _vz -= f1z;
    // needed for following accumulation
    tmp3N[i] = f1x*f1x + f1y*f1y + f1z*f1z;
    vd[i]    = _vx;
    vd[i+N]  = _vy;
    vd[i+N2] = _vz;
  }
}

__global__ void propagateVelocityVerletGaussianThermo_F4(float4 *r, real dt, double *vd, float4 *F,
  float4 *Fold, real *minv, uint32_t N, double xi) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  const real dt2_2 = dt * dt * 0.5; // dt^2/2
  const real xidt2 = xi * dt2_2;
  const uint32_t N2 = N*2;
  for (uint32_t i=thID;i<N;i+=thNum) {
    const real dt2_mass_2 = dt2_2 * minv[i];  // dt^2/mass/2

    float4 _r = r[i];
    float4 _F = F[i];
    _r.x += vd[i]    * (dt - xidt2) + _F.x * dt2_mass_2;
    _r.y += vd[i+N]  * (dt - xidt2) + _F.y * dt2_mass_2;
    _r.z += vd[i+N2] * (dt - xidt2) + _F.z * dt2_mass_2;

    r[i] = _r;
    Fold[i] = _F;
  }
}

__global__ void innerProduct_F4(float4 *v1, double *vd, uint32_t N, double *Res) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  extern __shared__ double tmp_d[];

  const uint32_t N2 = N*2;
  tmp_d[thID] = 0.0;

  for (uint32_t i=thID;i<N;i+=thNum) {
    double d = v1[i].x * vd[i];
    d += v1[i].y * vd[i+N];
    d += v1[i].z * vd[i+N2];
    tmp_d[thID] = d;
  }
  __syncthreads();

  uint32_t k=1;
  do {
    const uint32_t J = thID * k * 2;
    if (J+k<thNum) {
      tmp_d[J] += tmp_d[J+k];
    }
    k *= 2;
    __syncthreads();
  } while (k<thNum);

  if (thID==0) Res[0] = tmp_d[0];
}

__global__ void calcMV2_F4(double *vd, float* m, double *tmp3N, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  const uint32_t N2 = N*2;
  for (uint32_t i=thID;i<N;i+=thNum) {
    double _tmp = vd[i]*vd[i];
    _tmp += vd[i+N]*vd[i+N];
    _tmp += vd[i+N2]*vd[i+N2];
    _tmp *= m[i];
    tmp3N[i] = _tmp;
  }
}


/*
 * particleMD
 */
__global__ void calcV2_F4(float4 *v, float *tmp3N, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    double _tmp = v[i].x*v[i].x;
    _tmp += v[i].y*v[i].y;
    _tmp += v[i].z*v[i].z;
    tmp3N[i] = (float)_tmp;
  }
}


__global__ void correctConstTemp_F4(float4 *v, float4 *F, real *m, real lambda, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    float4 _F = F[i];
    _F.x -= lambda * m[i] * v[i].x;
    _F.y -= lambda * m[i] * v[i].y;
    _F.z -= lambda * m[i] * v[i].z;
    F[i] = _F;
  }
}

__global__ void adjustVelocity_F4(float4 *v, uint32_t st, uint32_t N, real v1, float *debug) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  extern __shared__ double tmp_v[];
  
  for (int i=0;i<6;++i) {
    tmp_v[thID + i*st] = 0.0;
  }

  for (uint32_t i=thID;i<N;i+=thNum) {
    tmp_v[thID]        += v[i].x;
    tmp_v[thID + st]   += v[i].y;
    tmp_v[thID + st*2] += v[i].z;
    tmp_v[thID + st*3] += v[i].x * v[i].x;
    tmp_v[thID + st*4] += v[i].y * v[i].y;
    tmp_v[thID + st*5] += v[i].z * v[i].z;
  }
  __syncthreads();
  
  // reduction 1024*6 => 6
  uint32_t k=1;
  do {
    const uint32_t J = thID * k * 2;
    if (J+k<thNum) {
      tmp_v[J]        += tmp_v[J+k];
      tmp_v[J + st]   += tmp_v[J+k + st];
      tmp_v[J + st*2] += tmp_v[J+k + st*2];
      tmp_v[J + st*3] += tmp_v[J+k + st*3];
      tmp_v[J + st*4] += tmp_v[J+k + st*4];
      tmp_v[J + st*5] += tmp_v[J+k + st*5];
    }
    k *= 2;
    __syncthreads();
  } while (k<thNum);

  k = 512;
  do {
    const uint32_t J = thID * k * 2;
    if (J+k<thNum) {
      tmp_v[J+k]        = tmp_v[J];
      tmp_v[J+k + st]   = tmp_v[J + st];
      tmp_v[J+k + st*2] = tmp_v[J + st*2];
      tmp_v[J+k + st*3] = tmp_v[J + st*3];
      tmp_v[J+k + st*4] = tmp_v[J + st*4];
      tmp_v[J+k + st*5] = tmp_v[J + st*5];
    }
    k /= 2;
    __syncthreads();
  } while (k>0);

  tmp_v[thID]        /= N;
  tmp_v[thID + st]   /= N;
  tmp_v[thID + st*2] /= N;
  tmp_v[thID + st*3] /= N;
  tmp_v[thID + st*4] /= N;
  tmp_v[thID + st*5] /= N;
  tmp_v[thID + st*3] -= tmp_v[thID]        * tmp_v[thID];
  tmp_v[thID + st*4] -= tmp_v[thID + st]   * tmp_v[thID + st];
  tmp_v[thID + st*5] -= tmp_v[thID + st*2] * tmp_v[thID + st*2];
  tmp_v[thID + st*3] = v1 * rsqrt(tmp_v[thID + st*3]);
  tmp_v[thID + st*4] = v1 * rsqrt(tmp_v[thID + st*4]);
  tmp_v[thID + st*5] = v1 * rsqrt(tmp_v[thID + st*5]);

  for (uint32_t i=thID;i<N;i+=thNum) {
    float4 _v = v[i];
    _v.x = (_v.x - tmp_v[thID])        * tmp_v[thID + st*3];
    _v.y = (_v.y - tmp_v[thID + st])   * tmp_v[thID + st*4];
    _v.z = (_v.z - tmp_v[thID + st*2]) * tmp_v[thID + st*5];
    v[i] = _v;
  }
  
  if (debug != NULL) {
    if (thID < 6) {
      debug[thID] = tmp_v[thID * st];
    }
  }
}

__global__ void adjustVelocity_VD(double *vd, uint32_t st, uint32_t N, real v1, float *debug) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  extern __shared__ double tmp_v[];
  uint32_t N2 = N * 2;

  for (int i=0;i<6;++i) {
    tmp_v[thID + i*st] = 0.0;
  }

  for (uint32_t i=thID;i<N;i+=thNum) {
    tmp_v[thID]        += vd[i];
    tmp_v[thID + st]   += vd[i+N];
    tmp_v[thID + st*2] += vd[i+N2];
    tmp_v[thID + st*3] += vd[i] * vd[i];
    tmp_v[thID + st*4] += vd[i+N] * vd[i+N];
    tmp_v[thID + st*5] += vd[i+N2] * vd[i+N2];
  }
  __syncthreads();
  
  // reduction 1024*6 => 6
  uint32_t k=1;
  do {
    const uint32_t J = thID * k * 2;
    if (J+k<thNum) {
      tmp_v[J]        += tmp_v[J+k];
      tmp_v[J + st]   += tmp_v[J+k + st];
      tmp_v[J + st*2] += tmp_v[J+k + st*2];
      tmp_v[J + st*3] += tmp_v[J+k + st*3];
      tmp_v[J + st*4] += tmp_v[J+k + st*4];
      tmp_v[J + st*5] += tmp_v[J+k + st*5];
    }
    k *= 2;
    __syncthreads();
  } while (k<thNum);

  k = 512;
  do {
    const uint32_t J = thID * k * 2;
    if (J+k<thNum) {
      tmp_v[J+k]        = tmp_v[J];
      tmp_v[J+k + st]   = tmp_v[J + st];
      tmp_v[J+k + st*2] = tmp_v[J + st*2];
      tmp_v[J+k + st*3] = tmp_v[J + st*3];
      tmp_v[J+k + st*4] = tmp_v[J + st*4];
      tmp_v[J+k + st*5] = tmp_v[J + st*5];
    }
    k /= 2;
    __syncthreads();
  } while (k>0);

  tmp_v[thID]        /= N;
  tmp_v[thID + st]   /= N;
  tmp_v[thID + st*2] /= N;
  tmp_v[thID + st*3] /= N;
  tmp_v[thID + st*4] /= N;
  tmp_v[thID + st*5] /= N;
  tmp_v[thID + st*3] -= tmp_v[thID]        * tmp_v[thID];
  tmp_v[thID + st*4] -= tmp_v[thID + st]   * tmp_v[thID + st];
  tmp_v[thID + st*5] -= tmp_v[thID + st*2] * tmp_v[thID + st*2];
  tmp_v[thID + st*3] = v1 * rsqrt(tmp_v[thID + st*3]);
  tmp_v[thID + st*4] = v1 * rsqrt(tmp_v[thID + st*4]);
  tmp_v[thID + st*5] = v1 * rsqrt(tmp_v[thID + st*5]);

  for (uint32_t i=thID;i<N;i+=thNum) {
    vd[i]    = (vd[i]    - tmp_v[thID])        * tmp_v[thID + st*3];
    vd[i+N]  = (vd[i+N]  - tmp_v[thID + st])   * tmp_v[thID + st*4];
    vd[i+N2] = (vd[i+N2] - tmp_v[thID + st*2]) * tmp_v[thID + st*5];
  }

  if (debug != NULL) {
    if (thID < 6) {
      debug[thID] = tmp_v[thID * st];
    }
  }
}

/*
 * particleSPH
 */



/*
 * particleSPH_NS
 */
__global__ void calcF_SPH_NS(real *r, real *a, unsigned short *typeID, uint32_t N,
    real *dW2D, real *rhoinv, real *m, real *mu, real *v, const real K) {
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
   *  \f$ p/rho/rho = Krho/rho/rho = K/rho; p1 = m_j K (1/rho_i + 1/rho_j) * dW/dR /r \f$
   *
   * dW/dR, and n are calculated by calcKernels() and calcDensity()
   */


  /* i-j pair are divided into (blockDim.x*4) x (blockDim.y*4) tiling
   * in each tile, threads checks all grid
   * cuda block X and Y corresponds to this tile
   */
  uint32_t istart = (blockDim.x*4) * blockIdx.x;
  uint32_t jstart = (blockDim.y*4) * blockIdx.y;
  uint32_t iend = istart + (blockDim.x*4);
  uint32_t jend = jstart + (blockDim.y*4);
  if (iend>N) iend = N;
  if (jend>N) jend = N;
  uint32_t N2 = N * 2;

  for (uint32_t i=istart+threadIdx.x;i<iend;i+=blockDim.x) {
    real _r[3];
    _r[0] = r[i];
    _r[1] = r[i+N];
    _r[2] = r[i+N2];
    for (uint32_t j=jstart+threadIdx.y;j<jend;j+=blockDim.y) {
      if (i!=j) {

        // p1 = K (1/n_i + 1/n_j) * dW/dR
        // v1 = (mu_i + mu_j) / n_i / n_j   * dW/dR
        //const real p1 = K * (ninv[i] + ninv[j]) * dW2D[i+j*N];
        //const real v1 = (mu[i]+mu[j]) * ninv[i] * ninv[j] * dW2D[i+j*N];
        const real p1 = m[j] * K * ((1+typeID[i]*4)*rhoinv[i] + (1+typeID[j]*4)*rhoinv[j]) * dW2D[i+j*N];
        const real v1 = m[j] * (mu[i]+mu[j]) * rhoinv[i] * rhoinv[j] * dW2D[i+j*N];

        // F_i[x] += -p1*r[x] + v1 * (v_i[x] - v_j[x])


        real _f[3];
        _f[0] = r[j]    - _r[0];
        _f[1] = r[j+N]  - _r[1];
        _f[2] = r[j+N2] - _r[2];

        a[i]    -= - _f[0] * p1 + (v[j]    - v[i])    * v1;
        a[i+N]  -= - _f[1] * p1 + (v[j+N]  - v[i+N])  * v1;
        a[i+N2] -= - _f[2] * p1 + (v[j+N2] - v[i+N2]) * v1;
      }
    }
  }
}

__global__ void inspectDense_x(float4 *n, char *move, uint32_t N, float4 *R) {
  const uint32_t thID  = threadIdx.x;
  const uint32_t thNum = blockDim.x;

  extern __shared__ real tmp_r[];
  uint32_t *tmp_n = (uint32_t *)(&tmp_r[thNum]);
  tmp_r[thID] = 0.0;
  tmp_n[thID] = 0;
  for (uint32_t i=thID;i<N;i+=thNum) {
    if (move[i]>0) {
      tmp_n[thID] += 1;
      tmp_r[thID] += n[i].x;
    }
  }
  __syncthreads();

  uint32_t k=1;
  do {
    const uint32_t J = thID * k * 2;
    if (J+k<thNum) {
      tmp_n[J] += tmp_n[J+k];
      tmp_r[J] += tmp_r[J+k];
    }
    k *= 2;
  } while (k<thNum);

  if (thID==0) R[0].x = tmp_r[thID] / tmp_n[thID];

  return;
}


/*
 * used from cudaCutoffBlock
 */
__global__ void calcBID_F4(float4 *r, uint32_t *bid , uint32_t *pid, uint32_t N, uint32_t totalNumBlock,
    real b0, real b1, real b2,
    real cxmin, real cymin, real czmin,
    uint32_t c0, uint32_t c1, uint32_t c2) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t CX = c2;
  const uint32_t CY = c1 * c2;

  for (uint32_t _i=thID;_i<N;_i+=thNum) {
    const uint32_t i=pid[_i];
    uint32_t d0 = static_cast<uint32_t>((r[i].x -cxmin) / b0);
    uint32_t d1 = static_cast<uint32_t>((r[i].y -cymin) / b1);
    uint32_t d2 = static_cast<uint32_t>((r[i].z -czmin) / b2);
    if (d0 == c0) {
      d0 = c0-1;
    } else if (d0 > c0) {
      d0 = 0;
    }
    if (d1 == c1) {
      d1 = c1-1;
    } else if (d1 > c1) {
      d1 = 0;
    }
    if (d2 == c2) {
      d2 = c2-1;
    } else if (d2 > c2) {
      d2 = 0;
    }

    bid[_i] = d2 + d1 * CX + d0 * CY;
    assert(bid[_i] < totalNumBlock);
    //pid[i] = i;
  }
}


__global__ void sortByBID(uint32_t *pid, uint32_t *bid, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  __shared__ uint32_t d;  // should be global


  do {
    d=0;
    for (uint32_t i=thID*2;i<N-1;i+=thNum*2) {
      if (bid[i]>bid[i+1]) {
        const uint32_t b = bid[i+1];
        const uint32_t c = pid[i+1];
        bid[i+1] = bid[i];
        pid[i+1] = pid[i];
        bid[i] = b;
        pid[i] = c;
        //bid[i] = atomicExch(&bid[i+1], bid[i]);
        //pid[i] = atomicExch(&pid[i+1], pid[i]);
        d=1;
      }
    }

    for (uint32_t i=thID*2+1;i<N-1;i+=thNum*2) {
      if (bid[i]>bid[i+1]) {
        const uint32_t b = bid[i+1];
        const uint32_t c = pid[i+1];
        bid[i+1] = bid[i];
        pid[i+1] = pid[i];
        bid[i] = b;
        pid[i] = c;
        //bid[i] = atomicExch(&bid[i+1], bid[i]);
        //pid[i] = atomicExch(&pid[i+1], pid[i]);
        d=1;
      }
    }
    __syncthreads();
  } while (d>0);
}

__global__ void makeBindex(uint32_t *bid , uint32_t *bindex, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N-1;i+=thNum) {
    const uint32_t c1 = bid[i];
    const uint32_t c2 = bid[i+1];
    if (c1!=c2) {
      // block boundary in i|i+1; block c2 starts from i+1
      bindex[c2] = i+1;
    }
    //bindex[end] = N;
  }
  if (thID==0) bindex[0] = 0;
}

__global__ void sortByBID_M1(uint32_t *pid, uint32_t *bid, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t startI=thID+thNum;startI<N;startI+=thNum) {

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
    while (bid[startI-thNum]>bid[startI]) {
      uint32_t I = startI - thNum;
      const uint32_t tmpbid = bid[startI];
      const uint32_t tmppid = pid[startI];

      while ((I>thNum)&&(bid[I-thNum]>tmpbid)) I -= thNum;
      bid[startI] = bid[I];
      pid[startI] = pid[I];
      bid[I] = tmpbid;
      pid[I] = tmppid;
    }
  }
}


__global__ void reduce27_F4(float4 *A, real *A27, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i=thID;i<N;i+=thNum) {
    float4 T = A[i];
    for (int b=0;b<27;++b) {
      const real * const V = &(A27[N*b]);
      T.x += V[i];
      T.y += V[i+N*27];
      T.z += V[i+N*54];
      T.w = 0; // used for DOT in gaussianThermo
    }
    A[i] = T;
  }
}

__global__ void RestoreByPid_F4(float4 *dst, float4 *src, uint32_t N, uint32_t *pid) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    dst[pid[i]].x = src[i].x;
    dst[pid[i]].y = src[i].y;
    dst[pid[i]].z = src[i].z;
    dst[pid[i]].w = src[i].w;
  }
}

/*
 * cudaParticleRotation
 */
__global__ void propagateEulerianRotation_F4(float4 *w, real dt, float4 *L, float4 *T,
    real *m0inv, char *move, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    real dt2 = dt * move[i];  // move[i]: 0(fixed) or 1(other)
    real m0  = m0inv[i] * move[i];
    float4 l = L[i];
    l.x += T[i].x * dt2;
    l.y += T[i].y * dt2;
    l.z += T[i].z * dt2;
    L[i] = l;

    float4 W = w[i];
    W.x = l.x * m0;
    W.y = l.y * m0;
    W.z = l.z * m0;
    w[i] = W;
  }
}



/*
 * cudaSelectedBlock
 */
__global__ void checkBlocks(uint32_t *selected, uint32_t *pid, char *move,
    uint32_t *bindex, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    if (bindex[i]!=UINT_MAX) {
      long __I = i+1;
      while (bindex[__I]==UINT_MAX) ++__I;
      const uint32_t bendI = bindex[__I];
      for (uint32_t i0=bindex[i]+threadIdx.y;i0<bendI;i0+=blockDim.y) {
        const uint32_t p = pid[i0];
        if (move[p]==1) {
          selected[i] = 1;
        }
      }
    }
  }
}

__global__ void accumulate_F4(uint32_t *selected, uint32_t N, float4 *Res) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  extern __shared__ uint32_t tmp_s[];
  tmp_s[thID] = 0;

  for (uint32_t i=thID;i<N;i+=thNum) {
    tmp_s[thID] += selected[i];
  }

  __syncthreads();
  uint32_t k=1;
  do {
    const uint32_t J = thID * k * 2;
    if (J+k<thNum) {
      tmp_s[J] += tmp_s[J+k];
    }
    k *= 2;
  } while (k<thNum);

  if (thID==0) Res[0].x = static_cast<real>(tmp_s[0]);
}

__global__ void writeBlockID(uint32_t *selected, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    selected[i] *= i;
  }
}

__global__ void accumulate(float4 *selected, uint32_t N, float4 *Res) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  extern __shared__ double tmp_s4[];
  tmp_s4[thID] = 0.0;

  for (uint32_t i=thID;i<N;i+=thNum) {
    tmp_s4[thID] += selected[i].x;
  }

  __syncthreads();
  uint32_t k=1;
  do {
    const uint32_t J = thID * k * 2;
    if (J+k<thNum) {
      tmp_s4[J] += tmp_s4[J+k];
    }
    k *= 2;
  __syncthreads();
  } while (k<thNum);

  if (thID==0) Res[0].x = static_cast<float>(tmp_s4[0]);
}

/*
 * cudaSparseMat
 */
__global__ void real2ulong_F4(float4 *r, uint32_t *l, uint32_t N, uint32_t *pid) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  if (pid!=NULL) {
    for (uint32_t i=thID;i<N;i+=thNum) {
      l[pid[i]] = static_cast<uint32_t>(r[i].x);
    }
  } else {
    for (uint32_t i=thID;i<N;i+=thNum) {
      l[i] = static_cast<uint32_t>(r[i].x);
    }
  }
}
__global__ void partialsum(uint32_t *rowPtr, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  extern __shared__ uint32_t subarray[];

  subarray[thID] = 0;

  uint32_t d = static_cast<uint32_t>(N / thNum);
  //if (d*thNum<N) ++d;
  uint32_t * const _row = &(rowPtr[d*thID]);
  if (thID==thNum-1) d = N - (d*(thNum-1));

  //uint32_t _N=_row[0];
  for (uint32_t i=1;i<d;++i) {
    //_N += _row[i];
    //_row[i] = N;
    _row[i] += _row[i-1];
  }
  subarray[thID+1] = _row[d-1];
  __syncthreads();

  if (thID==0)
    for (uint32_t i=1;i<thNum;++i)
      subarray[i] += subarray[i-1];
  __syncthreads();

  const uint32_t D=subarray[thID];
  for (uint32_t i=0;i<d;++i) {
    _row[i] += D;
  }
}

__global__ void makeJlist_WithBlock2_F4(const uint32_t *rowPtr, uint32_t *colIdx,
    const float4 *r,
    const uint32_t *blockNeighbor, const uint32_t *pid, const uint32_t *bindex,
    const uint32_t *bid_by_pid, const char *move,
    const uint32_t N,  const uint32_t Nstart, const uint32_t Nend
  ) {
  /*
   * i-particles are selected in 1D thread parallel,
   * block ID of I-block was calculated from the position of i-particle and then
   * neighboring J-blocks are treated in serial
   */
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x + Nstart;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t Nmax = N - Nend;

  for (uint32_t i=thID;i<Nmax;i+=thNum) {
    const int I=bid_by_pid[i];
    if (move[i]>0) {
      const real _r[3] = {r[i].x, r[i].y, r[i].z};
      const double r0i = r[i].w;

      if (rowPtr[i+1]!=rowPtr[i]) {
        uint32_t *col = &(colIdx[rowPtr[i]]);

        for (int J27=0;J27<27;++J27) {
          const int J=blockNeighbor[I*27+J27];
          if (J!=UINT_MAX) {
            if (bindex[J]!=UINT_MAX) {
              int __J = J+1;
              while (bindex[__J]==UINT_MAX) ++__J;
              const uint32_t bendJ = bindex[__J];
              for (uint32_t j0=bindex[J];j0<bendJ;++j0) {
                const uint32_t j=pid[j0];

                if (i!=j) {
                  // register if i--j is enough close
                  const double R2 = distance2d(_r[0], _r[1], _r[2], r[j].x, r[j].y, r[j].z);
                  const double rc2 = r0i + r[j].w;
                  if (R2<=rc2*rc2) {
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

__global__ void makeJlist_WithBlock4_F4(const uint32_t *rowPtr, uint32_t *colIdx,
    const float4 *r,
    const uint32_t *blockNeighbor, const uint32_t *bindex,
    const uint32_t *bid, const char *move,
    const uint32_t N,
    const uint32_t Nstart, const uint32_t Nend
  ) {
  /*
   * i-particles are selected in 1D thread parallel,
   * block ID of I-block was calculated from the position of i-particle and then
   * neighboring J-blocks are treated in serial
   */
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x + Nstart;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t Nmax = N - Nend;

  for (uint32_t i=thID;i<Nmax;i+=thNum) {
    if (move[i]>0) {
      if (rowPtr[i+1]!=rowPtr[i]) {
        const long I=bid[i];
        assert(bindex[I]==i);
        const real _r[3] = {r[i].x, r[i].y, r[i].z};
        const real r0i = r[i].w;

        uint32_t *col = &(colIdx[rowPtr[i]]);

        for (int J27=0;J27<125;++J27) {
          const long J=blockNeighbor[I*125+J27];
          if (J!=UINT_MAX) {
            if (bindex[J]!=UINT_MAX) {
              const uint32_t j=bindex[J];
              if (i!=j) {
                // register if i--j is enough close
                const double R2 = distance2d(_r[0], _r[1], _r[2], r[j].x, r[j].y, r[j].z);
                const real rc2 = r0i + r[j].w;
                if (R2<=rc2*rc2) {
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


__global__ void calcBID_direct_F4(const float4 *r, uint32_t *bid, uint32_t *bindex,
    const uint32_t N, const real b0, const real b1, const real b2,
    const real cxmin, const real cymin, const real czmin,
    const uint32_t blen0, const uint32_t blen1, const uint32_t blen2,
    const uint32_t CX, const uint32_t CY) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    signed long d0 = static_cast<signed long>((r[i].x -cxmin) / b0);
    signed long d1 = static_cast<signed long>((r[i].y -cymin) / b1);
    signed long d2 = static_cast<signed long>((r[i].z -czmin) / b2);
    assert(d0>=-1);
    assert(d1>=-1);
    assert(d2>=-1);
    assert(d0<=blen0);
    assert(d1<=blen1);
    assert(d2<=blen2);
    if (d0==blen0) --d0;
    if (d1==blen1) --d1;
    if (d2==blen2) --d2;
    if (d0==-1) ++d0;
    if (d1==-1) ++d1;
    if (d2==-1) ++d2;

    const uint32_t BID = d2 + d1 * CX + d0 * CY;
    assert(bindex[BID]==UINT_MAX);

    bid[i] = BID;
    bindex[BID] = i;
  }
}

__global__ void makeBIDbyPID(const uint32_t *pid, uint32_t *bid, uint32_t *bid_by_pid,
    const uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t _i=thID;_i<N;_i+=thNum) {
    bid_by_pid[pid[_i]] = bid[_i];
  }
}

__global__ void sortColIdx(const uint32_t *rowPtr, uint32_t *colIdx, uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    // sort between rowPtr[i]..rowPtr[i+1]
    const uint32_t jbegin=rowPtr[i];
    const uint32_t jend=rowPtr[i+1];
    const uint32_t gap = jend - jbegin;

    if (gap<2) return;
    else if (gap==2) {
      if (colIdx[jbegin]>colIdx[jbegin+1]) {
        const uint32_t _j = colIdx[jbegin];
        colIdx[jbegin] = colIdx[jbegin+1];
        colIdx[jbegin+1] = _j;
      }
    } else {
      for (int __I=jbegin+1;__I<jend;++__I) {
        while ((__I>jbegin)&&(colIdx[__I-1]>colIdx[__I])) {
          const uint32_t __J = colIdx[__I-1];
          colIdx[__I-1] = colIdx[__I];
          colIdx[__I] = __J;
          --__I;
        }
      }
    }
  }
}

__global__ void succeedPrevState(uint32_t *prevRow, uint32_t *prevCol, real *prevVal,
    uint32_t *curRow, uint32_t *curCol, real *curVal,
    uint32_t N) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  for (uint32_t i=thID;i<N;i+=thNum) {
    uint32_t jx = curRow[i];
    const uint32_t _j = curRow[i+1];
    for (uint32_t j=prevRow[i];j<prevRow[i+1];++j) {
      const uint32_t j1 = prevCol[j];

      uint32_t j2 = curCol[jx];
      while ((j2<j1)&&(jx<_j)) j2 = curCol[++jx];
      if (j2==j1) curVal[jx] = prevVal[j];
    }
  }
}
