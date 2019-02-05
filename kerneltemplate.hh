#if !defined(KERNELTEMPLATE)
#define KERNELTEMPLATE

#include <stdio.h>
#include "kernelinline.hh"

/*
 * common template functions
 */
/** clear array on GPU with a size num
 *
 * @param r	array on GPU
 * @param num size of array
 */
template <typename T>
__global__ void clearArray(T *r, const uint32_t num, const T val = 0)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i = thID; i < num; i += thNum)
    r[i] = val;
}

/** calculate forces for all particles with function object
 *
 * @param op	function object to calculate the force from the distance between i and j particles
 * @param r	array for position (size 3N)
 * @param F	results array for calculated forces (size 3N)
 * @param N	number of particles
 */
template <typename Core>
__global__ void calcF_IJpair(const Core op, const real *r, real *F, const uint32_t N, real *OPT = NULL)
{
  /* i-j pair are divided into (blockDim.x*4) x (blockDim.y*4) tiling
   * in each tile, threads checks all grid
   * cuda block X and Y corresponds to this tile
   */
  const uint32_t istart = (blockDim.x * 4) * blockIdx.x;
  uint32_t iend = istart + (blockDim.x * 4);
  if (iend > N)
    iend = N;
  const uint32_t N2 = N * 2;

  for (uint32_t i = istart + threadIdx.x; i < iend; i += blockDim.x)
  { // 4times loop
    const real _r[3] = {r[i], r[i + N], r[i + N2]};
    real _f0[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (uint32_t j = 0; j < N; ++j)
    {
      op(_r, r, N, N2, i, j, _f0);
    }
    F[i] += _f0[0];
    F[i + N] += _f0[1];
    F[i + N2] += _f0[2];

    if (OPT != NULL)
    {
      OPT[i] += _f0[3];
      OPT[i + N] += _f0[4];
      OPT[i + N2] += _f0[5];
    }
  }
}

/** calculate forces for all particles with function object
 *
 * @param op	function object to calculate the force from the distance between i and j particles
 * @param rowPtr pointer to colIdx[] for each i particles
 * @param colIdx array which stores particle ID for j
 * @param r	array for position (size 3N)
 * @param F	results array for calculated forces (size 3N)
 * @param N	number of particles
 * @param OPT	optinal args for Torque
 * @param Nstart	offset for pid range
 * @param Nend		determin the range of particle ID to calculates, Nmax=N-Nend
 */
template <typename Core>
__global__ void calcF_IJpairWithList(const Core op,
                                     const uint32_t *rowPtr, const uint32_t *colIdx,
                                     const real *r, real *F, const uint32_t N, real *OPT = NULL,
                                     const uint32_t Nstart = 0, const uint32_t Nend = 0)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x + Nstart;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t N2 = N * 2;
  const uint32_t Nmax = N - Nend;

  for (uint32_t i = thID; i < Nmax; i += thNum)
  {
    const real _r[3] = {r[i], r[i + N], r[i + N2]};
    real _f0[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (uint32_t col = rowPtr[i]; col < rowPtr[i + 1]; ++col)
    {
      op(_r, r, N, N2, i, colIdx[col], _f0);
    }
    F[i] += _f0[0];
    F[i + N] += _f0[1];
    F[i + N2] += _f0[2];

    if (OPT != NULL)
    {
      OPT[i] += _f0[3];
      OPT[i + N] += _f0[4];
      OPT[i + N2] += _f0[5];
    }
  }
}

/** calculate forces for all particles with function object with Cut Off Block defined in class cudaCutoffBlock
 *
 * @param op	function object to calculate the force from the i-j particle pair
 * @param r	array for position (size 3N)
 * @param F	results array for calculated forces (size 3N)
 * @param blockNeighbor	neighbor judgement for I-J block pair
 * @param totalNumBlock	total number of cut off blocks
 * @param pid	particle ID in sorted order
 * @param bindex	start particle num of this block
 * @param N	number of particles
 * @param OPT	optinal args for Torque
 * @param sorted	default false; if true, sorted array r_s[] and typeid_s[] are used insetead of r[] and typeid[]
 * @param sortedF	default false; if true, write to F[] with i0 instead of pid[i0], following calculation should have conversion
 */
template <typename Core>
__global__ void calcF_IJpairWithBlock(const Core op, real *r,
                                      real *tmp81N,
                                      uint32_t BlockOffset,
                                      uint32_t *blockNeighbor, uint32_t *pid, uint32_t *bindex,
                                      uint32_t N, real *OPT = NULL,
                                      uint32_t *selectedBlock = NULL, const bool sorted = false, const bool sortedF = false)
{
  /*
   * cuda block ID for X and Y indicates ID of cutoff block I and J.
   * threads i, j sweeps particles in I and J.
   *
   */
  const uint32_t I =
      (selectedBlock != NULL) ? selectedBlock[blockIdx.x + BlockOffset] : blockIdx.x + BlockOffset;
  const uint32_t J27 = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t J = blockNeighbor[I * 27 + J27];
  if (J == UINT_MAX)
    return;

  if (bindex[I] == UINT_MAX)
    return;
  if (bindex[J] == UINT_MAX)
    return;

  const uint32_t N2 = N * 2;

  real *_F = &(tmp81N[3 * N * J27]);

  int __I = I + 1;
  while (bindex[__I] == UINT_MAX)
    ++__I;
  const uint32_t bendI = bindex[__I];
  int __J = J + 1;
  while (bindex[__J] == UINT_MAX)
    ++__J;
  const uint32_t bendJ = bindex[__J];
  // bindex[totalNumBlock]==N

  for (uint32_t i0 = bindex[I] + threadIdx.x; i0 < bendI; i0 += blockDim.x)
  {
    const uint32_t i = pid[i0];
    const uint32_t i2 = (!sorted) ? i : i0;
    const uint32_t i3 = (!sortedF) ? i : i0;

    const real _r[3] = {r[i2], r[i2 + N], r[i2 + N2]};
    real _f0[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const uint32_t Jnum = bendJ - bindex[J];
    for (uint32_t j0 = 0; j0 < Jnum; ++j0)
    {
      const uint32_t j2 = (j0 + threadIdx.x) % Jnum;
      const uint32_t j = (!sorted) ? pid[j2 + bindex[J]] : j2 + bindex[J];

      // call function object to calculate the CORE of the calculation
      op(_r, r, N, N2, i2, j, _f0);
    }
    _F[i3] = _f0[0];
    _F[i3 + N] = _f0[1];
    _F[i3 + N2] = _f0[2];

    if (OPT != NULL)
    {
      real *_T = &(OPT[3 * N * J27]);
      _T[i3] = _f0[3];
      _T[i3 + N] = _f0[4];
      _T[i3 + N2] = _f0[5];
    }
  }
}
/** calculate forces for all particles with function object with Cut Off Block defined in class cudaCutoffBlock
 *
 * @param op	function object to calculate the force from the i-j particle pair
 * @param r	array for position (size 3N)
 * @param _F	results array for calculated forces (size 3N)
 * @param blockNeighbor	neighbor judgement for I-J block pair
 * @param pid	particle ID in sorted order
 * @param bindex	start particle num of this block
 * @param bid	array for the Block ID
 * @param move  move/fixed particle table
 * @param N	number of particles
 * @param _T	optinal args for Torque
 * @param Nstart	offset for pid range
 * @param Nend		determin the range of particle ID to calculates, Nmax=N-Nend
 */
template <typename Core>
__global__ void calcF_IJpairWithBlock2(const Core op, const real *r,
                                       real *_F,
                                       const uint32_t *blockNeighbor, const uint32_t *pid, const uint32_t *bindex,
                                       const uint32_t *bid_by_pid, const char *move,
                                       const uint32_t N, real *_T = NULL,
                                       const uint32_t Nstart = 0, const uint32_t Nend = 0)
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
      real _f0[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      for (int J27 = 0; J27 < 27; ++J27)
      {
        const uint32_t J = blockNeighbor[I * 27 + J27];
        if (J != UINT_MAX)
        {
          if (bindex[J] != UINT_MAX)
          {
            uint32_t __J = J + 1;
            while (bindex[__J] == UINT_MAX)
              ++__J;
            const uint32_t bendJ = bindex[__J];
            for (int j0 = bindex[J]; j0 < bendJ; ++j0)
            {
              const uint32_t j = pid[j0];

              // call function object to calculate the CORE of the calculation
              op(_r, r, N, N2, i, j, _f0);
            }
          }
        }
      }
      _F[i] = _f0[0];
      _F[i + N] = _f0[1];
      _F[i + N2] = _f0[2];

      /*
      // never used
      if (_T!=NULL) {
	_T[i]    = _f0[3];
	_T[i+N]  = _f0[4];
	_T[i+N2] = _f0[5];
      }
*/
    }
  }
}

template <typename Core>
__global__ void calcF_IJpairWithBlock4(const Core op, const real *r,
                                       real *_F,
                                       const uint32_t *blockNeighbor, const uint32_t *bindex,
                                       const uint32_t *bid, const char *move,
                                       const uint32_t N, real *_T = NULL,
                                       const uint32_t Nstart = 0, const uint32_t Nend = 0)
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
    const int I = bid[i];
    if (move[i] > 0)
    {
      const real _r[3] = {r[i], r[i + N], r[i + N2]};
      real _f0[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      for (int J27 = 0; J27 < 125; ++J27)
      {
        const uint32_t J = blockNeighbor[I * 125 + J27];
        if (J != UINT_MAX)
        {
          if (bindex[J] != UINT_MAX)
          {
            const uint32_t j = bindex[J];
            // call function object to calculate the CORE of the calculation
            op(_r, r, N, N2, i, j, _f0);
          }
        }
      }
      _F[i] = _f0[0];
      _F[i + N] = _f0[1];
      _F[i + N2] = _f0[2];

      /*
      // never used
      if (_T!=NULL) {
	_T[i]    = _f0[3];
	_T[i+N]  = _f0[4];
	_T[i+N2] = _f0[5];
      }
*/
    }
  }
}

/*
 * particleMD
 */
/** the core for the calculation of forces from the pair potential used in MD
 *
 * implementation of the pair potential such as LJ, softcore... are defined by the template parameter
 *
 */
template <typename Potential>
class MDpairPotential
{
public:
  // cell length
  real cx, cy, cz, c0x, c0y, c0z, rmax2;
  unsigned short *typeID;
  Potential op;
  __device__ void operator()(const real *_r, const real *r, uint32_t N, uint32_t N2, uint32_t i, uint32_t j, real *_f0) const
  {
    if (i != j)
    {
      const real R2 = distance2p(_r[0], _r[1], _r[2], r[j], r[j + N], r[j + N2], cx, cy, cz);
      if (R2 < rmax2)
      {
        //	real f = (R2<rmax2) ? op(R2, typeID_i, typeID[j]) : 0.0 ;
        real f = op(R2, typeID[i], typeID[j]);

        real _f[3];
        _f[0] = _r[0] - r[j];
        _f[1] = _r[1] - r[j + N];
        _f[2] = _r[2] - r[j + N2];
        if ((_f[0] * _f[0]) > cx)
          _f[0] += (_f[0] < 0) ? c0x : -c0x;
        if ((_f[1] * _f[1]) > cy)
          _f[1] += (_f[1] < 0) ? c0y : -c0y;
        if ((_f[2] * _f[2]) > cz)
          _f[2] += (_f[2] < 0) ? c0z : -c0z;

        const real _ff = rsqrt(R2);
        f *= _ff;
        _f[0] *= f;
        _f[1] *= f;
        _f[2] *= f;
#if defined(DUMPCOLLISION)
        if (((_f[0] * _f[0]) > 100) || ((_f[1] * _f[1]) > 100) || ((_f[2] * _f[2]) > 100))
        {
          printf("XXX: %ld:%ld\t%g, %g, %g\t%g, %g, %g\t%g, %g, %g\t%g, %g,  %hd %hd\nXX2: \t%d:%d %d:%d %ld,%d,%ld,%d\n",
                 i, j, r[i], r[i + N], r[i + N2], r[j], r[j + N], r[j + N2],
                 _f[0], _f[1], _f[2],
                 sqrt(R2), f, typeID[i], typeID[j],
                 I, J, i0, j0, bindex[I], bendI, bindex[J], bendJ);
        }
#endif
        _f0[0] += _f[0];
        _f0[1] += _f[1];
        _f0[2] += _f[2];
      }
    }
  }
};

/*
 * particle SPH
 */
/** calculate SPH kernel field as a 2D array for all i-j pair,
 * dW/dr was also calculated
 *
 * @param r	array for position (size 3N)
 * @param w2D	results array for kernel field
 * @param dW2D	results array for gradient of kernel dW/dr
 * @param N	number of particles
 * @param h	SPH kernel radius
 * @param w0	SPH kernel coefficient
 * @param w1	SPH kernel coefficient
 */
template <typename Kernel, typename KernelDW>
__global__ void calcSPHKernel(real *r, real *w2D, real *dW2D, uint32_t N, real h, real w0, real w1)
{
  /* i-j pair are divided into (blockDim.x*4) x (blockDim.y*4) tiling
   * in each tile, threads checks all grid
   * cuda block X and Y corresponds to this tile
   */
  const uint32_t istart = (blockDim.x * 4) * blockIdx.x;
  uint32_t iend = istart + (blockDim.x * 4);
  if (iend > N)
    iend = N;
  const uint32_t N2 = N * 2;

  Kernel K1;
  KernelDW K2;

  for (uint32_t i = istart + threadIdx.x; i < iend; i += blockDim.x)
  { // 4times loop
    const real _r[3] = {r[i], r[i + N], r[i + N2]};
    for (uint32_t j = 0; j < N; ++j)
    {
      //      if (i!=j) {
      const real R = sqrt(distance2(_r[0], _r[1], _r[2], r[j], r[j + N], r[j + N2]));

      const real W = K1(R, h, w0);
      w2D[i + j * N] = W;
      const real dW = K2(R, h, w1);
      dW2D[i + j * N] = dW;

      //      }
    }
  }
}

/** calculate mass and number density (without periodic boundary)
 *
 */
template <typename Kernel>
class SPHcalcDensity
{
public:
  real *m;
  real h, w0;
  real *opt;

  Kernel K1;

  __device__ void operator()(const real *_r, const real *r, uint32_t N, uint32_t N2, uint32_t i, uint32_t j, real *_f0) const
  {
    //      if (i!=j) {
    const real R = sqrt(distance2(_r[0], _r[1], _r[2], r[j], r[j + N], r[j + N2]));
    const real W = K1(R, h, w0);

    _f0[0] += W;
    _f0[1] += m[j] * W;
    if (opt != NULL)
      _f0[2] += opt[j] * m[j] * W;
    //      }
  }
};

/*
 * particleSPH_NS
 */
/** calculate accelerations by Naiver-Stokes equations
 *
 */
template <typename KernelDW>
class SPHNavierStokes
{
public:
  real *v;
  real *m;
  real *rhoinv;
  real *mu;
  real *c2;
  real h, w1;
  real rho0;

  KernelDW K2;

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
  __device__ void operator()(const real *_r, const real *r, uint32_t N, uint32_t N2, uint32_t i, uint32_t j, real *_f0) const
  {
    if (i != j)
    {
      const real R = sqrt(distance2(_r[0], _r[1], _r[2], r[j], r[j + N], r[j + N2]));
      // m_j * dw/dr /r

      const real m_j = m[j] * K2(R, h, w1);
      const real p1 = (c2[i] * rhoinv[i] * (1 - rho0 * rhoinv[i]) + c2[j] * rhoinv[j] * (1 - rho0 * rhoinv[j]));
      const real v1 = (mu[i] + mu[j]) * rhoinv[i] * rhoinv[j];

      _f0[0] += m_j * ((r[j] - _r[0]) * p1 - (v[j] - v[i]) * v1);
      _f0[1] += m_j * ((r[j + N] - _r[1]) * p1 - (v[j + N] - v[i + N]) * v1);
      _f0[2] += m_j * ((r[j + N2] - _r[2]) * p1 - (v[j + N2] - v[i + N2]) * v1);
    }
  }
};
#endif
