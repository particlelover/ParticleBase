#if !defined(KERNELTEMPLATE)
#define KERNELTEMPLATE

#include <stdio.h>
#include "kernelinline.hh"

/*
 * common template functions
 */
/** clear array on GPU with a size num
 *
 * @param r array on GPU
 * @param num size of array
 */
template <typename T>
__global__ void clearArray(T *r, const uint32_t num, const T val=0) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i=thID;i<num;i+=thNum)
    r[i] = val;
}

/** multiply array by value
 *
 * @param r array on GPU
 * @param num size of array
 * @param val value to multiply
 */
template <typename T>
__global__ void mulArray(T *r, T val, uint32_t num) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i=thID;i<num;i+=thNum)
    r[i] *= val;
}


/** calculates on GPU with device lambda function
 *
 *
 *
 */
template <typename V1, typename V2, typename T>
__global__ void calcBinaryFunc(V1 *src, V2 *dst, uint32_t num, T F) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i=thID;i<num;i+=thNum) {
    F(src[i], dst[i]);
  }
}

/** accumulate 1D array with shared memory and reduction
 * (MPnum must be 1)
 *
 * @param selected  1D table for the block selection
 * @param N total number of the cutoff blocks
 * @param Res results transfered from GPU's device memory to host memory
 */
template <typename T>
__global__ void accumulate(T *selected, uint32_t N, T *Res) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;

  extern __shared__ T tmp_s[];
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
  __syncthreads();
  } while (k<thNum);

  if (thID==0) Res[0] = tmp_s[0];
}


/** calculate forces for all particles with function object
 *
 * @param op  function object to calculate the force from the distance between i and j particles
 * @param r array for position (size 3N)
 * @param F results array for calculated forces (size 3N)
 * @param N number of particles
 */
template <typename Core>
__global__ void calcF_IJpair(const Core op, const real *r, real *F, const uint32_t N, real *OPT=NULL) {
  /* i-j pair are divided into (blockDim.x*4) x (blockDim.y*4) tiling
   * in each tile, threads checks all grid
   * cuda block X and Y corresponds to this tile
   */
  const uint32_t istart = (blockDim.x*4) * blockIdx.x;
  uint32_t iend = istart + (blockDim.x*4);
  if (iend>N) iend = N;
  const uint32_t N2 = N * 2;

  for (uint32_t i=istart+threadIdx.x;i<iend;i+=blockDim.x) {
    const real _r[3] = {r[i], r[i+N], r[i+N2]};
    real _f0[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (uint32_t j=0;j<N;++j) {
      op(_r, r, N, N2, i, j, _f0);
    }
    F[i]    += _f0[0];
    F[i+N]  += _f0[1];
    F[i+N2] += _f0[2];

    if (OPT!=NULL) {
      OPT[i]    += _f0[3];
      OPT[i+N]  += _f0[4];
      OPT[i+N2] += _f0[5];
    }
  }
}


/** calculate forces for all particles with function object
 *
 * @param op  function object to calculate the force from the distance between i and j particles
 * @param rowPtr pointer to colIdx[] for each i particles
 * @param colIdx array which stores particle ID for j
 * @param r array for position (float4 x size N)
 * @param F results array for calculated forces (float4 x size N)
 * @param N number of particles
 * @param OPT optinal args for Torque
 * @param Nstart  offset for pid range
 * @param Nend    determin the range of particle ID to calculates, Nmax=N-Nend
 */
template <typename Core>
__global__ void calcF_IJpairWithList_F4(const Core op,
    const uint32_t *rowPtr, const uint32_t *colIdx,
    const float4 *r, float4 *F, const uint32_t N, float4 *OPT=NULL,
    const uint32_t Nstart=0, const uint32_t Nend=0
  ) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x + Nstart;
  const uint32_t thNum = gridDim.x * blockDim.x;
  const uint32_t Nmax = N - Nend;

  for (uint32_t i=thID;i<Nmax;i+=thNum) {
    float _f0[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (uint32_t col = rowPtr[i];col<rowPtr[i+1];++col) {
      uint32_t j = colIdx[col];
      op(r[i], r[j], i, j, _f0);
    }
    float4 __f = F[i];
    __f.x += _f0[0];
    __f.y += _f0[1];
    __f.z += _f0[2];
    //__f.w = 0;
    F[i] = __f;

    if (OPT!=NULL) {
      float4 __ff = OPT[i];
      __ff.x    += _f0[3];
      __ff.y    += _f0[4];
      __ff.z    += _f0[5];
      //__ff.w    = 0;
      OPT[i] = __ff;
    }
  }
}

/** calculate forces for all particles with function object with Cut Off Block defined in class cudaCutoffBlock
 *
 * @param op function object to calculate the force from the i-j particle pair
 * @param r array for position (size N (float4))
 * @param tmp81N results array for calculated forces (size 3N x 27)
 * @param BlockOffset calculation range when multi-GPU mode
 * @param blockNeighbor neighboring Block ID Table for I-J block pair
 * @param pid particle ID sorted by block ID
 * @param bindex point the start point in pid for each block
 * @param N number of particles
 * @param OPT optinal args for the calculation of Torque
 * @param selectedBlock Block ID table which contains moving particles
 * @param sorted  default false; if true, sorted array r_s[] and typeid_s[] are used insetead of r[] and typeid[]
 * @param sortedF default false; if true, write to F[] with i0 instead of pid[i0], following calculation should have conversion
 *
 * How it works
 *
 * 1. choose block ID I from CUDA block index (*1)
 * 2. choose block ID J from the blockNeighbor[] table
 * 3. choose particle ID i in I-th block from pid[] table
 *   whose index i0 is in between bindex[I] and bindex[I+1] (*2)
 * 4. choose particle ID j in J-th block from pid[] table
 *   whose index j0 is in between bindex[J] and bindex[J+1] (*3)
 * 5. calculates the distance in between particle i and j, and then calculates the force
 *   by function object op
 * 6. caculated forces are stored and summed in thread local variable and write back to
 *   tmp81N[] array (*4)
 *
 * (*1) If the pointer to selectedBlock[] is given, the range of block I is restricted in blocks those
 *   who has moving particles, otherwise, all blocks are the target.
 *   In multi GPU mode, each GPU has different block ranges(controld by myBlockOffset and gridDim.x).
 * (*2) In particleMD, pre-sort for the position r[] and typeID[] are processed and then converting i0 to i
 *   is not needed. It is indicated by sorted=true.
 * (*3) The particle ID j has same treatment with i (sorted or not).
 * (*4) If sortedF set true, calculated force of i-th particle are write to F[i0] instead of F[pid[i0]].
 *   It is used for particleSPH_NS with exchangeForce() or exchangeAcceleration().
 *
 * There are two modes to choose the I-th block,
 *  a) from all blocks in the simulation cell
 *  b) from blocks which contains moving particles.
 *
 * To use them,
 *  a) set selectedBlock=NULL and sortedF=false
 *  b) set selectedBlock and sortedF=true.
 * The Range of blocks are
 *
 * ParticleMD uses mode (a) with sorted=true.
 *
 * using multi-gpu, the range [p1, p2) or [p3, p4) is used to indicates the target particles
 * to calculates the force. Exchange of calculated force after the calcF process is needed.
 *
 *
 * ParticleDEM uses this calcF_IJpairWithBlock to calculates the coordination number, originary, but
 * calcF_IJpairWithBlock2 or calcF_IJpairWithBlock4 are developed and now used instead of this.
 * ParticleDEM uses calcF_IJpairWithList for the calculation of forces.
 * makeJlist_WithBlock, makeJlist_WithBlock2 and makeJlist_WithBlock4 are also used from particleDEM,
 * as a succeeding process of the calculation of the coordinatio number.
 *
 */
template <typename Core>
__global__ void calcF_IJpairWithBlock_F4(const Core op, float4 *r,
    real *tmp81N,
    uint32_t BlockOffset,
    uint32_t *blockNeighbor, uint32_t *pid, uint32_t *bindex,
    uint32_t N, real *OPT=NULL,
    uint32_t *selectedBlock=NULL,
    const bool sorted=false,
    const bool sortedF=false
  ) {
  /*
   * cuda block ID for X and Y indicates ID of cutoff block I and J.
   * threads i, j sweeps particles in I and J.
   *
   */
  const uint32_t I =
    (selectedBlock!=NULL) ? selectedBlock[blockIdx.x+BlockOffset] :
    blockIdx.x + BlockOffset;
  const uint32_t J27 = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t J = blockNeighbor[I*27+J27];
  if (J==UINT_MAX) return;

  if (bindex[I]==UINT_MAX) return;
  if (bindex[J]==UINT_MAX) return;

  int __I = I+1;
  while (bindex[__I]==UINT_MAX) ++__I;
  const uint32_t bendI = bindex[__I];
  int __J = J+1;
  while (bindex[__J]==UINT_MAX) ++__J;
  const uint32_t bendJ = bindex[__J];
  // bindex[totalNumBlock]==N

  for (uint32_t i0=bindex[I]+threadIdx.x;i0<bendI;i0+=blockDim.x) {
    const uint32_t i = pid[i0];
    const uint32_t i2 = (!sorted) ? i : i0;
    const uint32_t i3 = (!sortedF) ? i : i0;

    real _f0[3] = {0.0, 0.0, 0.0};

    const uint32_t Jnum = bendJ - bindex[J];
    for (uint32_t j0=0;j0<Jnum;++j0) {
      const uint32_t j2 = (j0 + threadIdx.x) % Jnum;
      const uint32_t j = (!sorted) ? pid[j2 + bindex[J]] : j2 + bindex[J];

      // call function object to calculate the CORE of the calculation
      op(r[i2], r[j], i2, j, _f0);
    }

    // tmp81N[] array has a nested structure like [[[N] x27] x3]
    real *_F = &(tmp81N[N*J27]);
    _F[i3]      = _f0[0];
    _F[i3+N*27] = _f0[1];
    _F[i3+N*54] = _f0[2];

/*
      // never used
    if (OPT!=NULL) {
      real *_T = &(OPT[N*J27]);
      _T[i3]      = _f0[3];
      _T[i3+N*27] = _f0[4];
      _T[i3+N*54] = _f0[5];
    }
*/
  }
}

/** calculate forces for all particles with function object with Cut Off Block defined in class cudaCutoffBlock
 *
 * @param op  function object to calculate the force from the i-j particle pair
 * @param r array for position (float4 x size N)
 * @param _F  results array for calculated forces (float4 x size N)
 * @param blockNeighbor neighbor judgement for I-J block pair
 * @param pid particle ID in sorted order
 * @param bindex  start particle num of this block
 * @param bid array for the Block ID
 * @param move  move/fixed particle table
 * @param N number of particles
 * @param _T  optinal args for Torque
 * @param Nstart  offset for pid range
 * @param Nend  determin the range of particle ID to calculates, Nmax=N-Nend
 */
template <typename Core>
__global__ void calcF_IJpairWithBlock2_F4(const Core op, const float4 *r,
    float4 *_F,
    const uint32_t *blockNeighbor, const uint32_t *pid, const uint32_t *bindex,
    const uint32_t *bid_by_pid, const char *move,
    const uint32_t N, real *_T=NULL,
    const uint32_t Nstart=0, const uint32_t Nend=0
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
      real _f0[3] = {0.0, 0.0, 0.0};

      for (int J27=0;J27<27;++J27) {
        const uint32_t J=blockNeighbor[I*27+J27];
        if (J!=UINT_MAX) {
          if (bindex[J]!=UINT_MAX) {
            uint32_t __J = J+1;
            while (bindex[__J]==UINT_MAX) ++__J;
            const uint32_t bendJ = bindex[__J];
            for (int j0=bindex[J];j0<bendJ;++j0) {
              const uint32_t j=pid[j0];

              // call function object to calculate the CORE of the calculation
              op(r[i], r[j], i, j, _f0);
            }
          }
        }
      }
      float4 __f;
      __f.x = _f0[0];
      __f.y = _f0[1];
      __f.z = _f0[2];
      __f.w = 0;
      _F[i] = __f;

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
__global__ void calcF_IJpairWithBlock4_F4(const Core op, const float4 *r,
    float4 *_F,
    const uint32_t *blockNeighbor, const uint32_t *bindex,
    const uint32_t *bid, const char *move,
    const uint32_t N, real *_T=NULL,
    const uint32_t Nstart=0, const uint32_t Nend=0
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
    const int I=bid[i];
    if (move[i]>0) {
      real _f0[3] = {0.0, 0.0, 0.0};

      for (int J27=0;J27<125;++J27) {
        const uint32_t J=blockNeighbor[I*125+J27];
        if (J!=UINT_MAX) {
          if (bindex[J]!=UINT_MAX) {
            const uint32_t j=bindex[J];
            // call function object to calculate the CORE of the calculation
            op(r[i], r[j], i, j, _f0);
          }
        }
      }
      float4 __f;
      __f.x = _f0[0];
      __f.y = _f0[1];
      __f.z = _f0[2];
      __f.w = 0;
      _F[i] = __f;

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
 * calculate g(r)
 */
template <typename Core>
__global__ void calcF_IJpairWithBlock5_F4(const Core op, const float4 *r,
    const uint4 *pairInfo, uint32_t pairNum,
    uint32_t *igr, uint32_t rnum
  ) {
  extern __shared__ uint2 red[]; // size: uint2 * 64*64

  for (long p=blockIdx.x;p<pairNum;p+=gridDim.x) {
    for (uint32_t i=threadIdx.y;i<64;i+=blockDim.y) {
      for (uint32_t j=threadIdx.x;j<64;j+=blockDim.x) {
        red[i*64+j] = make_uint2(0, 0);
      }
    }

    const uint32_t bstartI = pairInfo[p].x;
    const uint32_t bstartJ = pairInfo[p].z;
    const uint32_t bendI = pairInfo[p].y;
    const uint32_t bendJ = pairInfo[p].w;
    assert(bendI - bstartI <= 64);
    assert(bendJ - bstartJ <= 64);
    const bool sameblock = bstartI == bstartJ;
    // i-particle in I-block, j-particle in J-block pair scan using 32x32 thread block
    for (uint32_t i=bstartI+threadIdx.y;i<bendI;i+=blockDim.y) {
      for (uint32_t j=bstartJ+threadIdx.x;j<bendJ;j+=blockDim.x) {
        const uint32_t _x = i - bstartI;
        const uint32_t _y = j - bstartJ;
        assert(_x < 64);
        assert(_y < 64);
        if (!sameblock || (i<j))
          op(r[i], r[j], i, j, &red[_x*64+_y]);
      }
    }
    __syncthreads();

    // reduction
    const uint32_t ofs = (threadIdx.y * 32 + threadIdx.x) * 64;
    for (uint32_t i=ofs+1;i<ofs+64;++i) {
      if ((threadIdx.y < 2) && (red[i].x != 0)) {
        for (uint32_t j=ofs;j<i;++j) {
          if (red[j].x == red[i].x) {
            red[j] = make_uint2(red[j].x, red[j].y + red[i].y);
            red[i] = make_uint2(0, 0);
            break;
          } else if (red[j].x == 0) {
            red[j] = make_uint2(red[i].x, red[i].y);
            red[i] = make_uint2(0, 0);
            break;
          }
        }
      }
    }
    __syncthreads();
    /*
    const uint32_t idx = threadIdx.y * 32 + threadIdx.x;
    for (uint32_t I=1;I<64;++I) {
      const uint32_t i = I * 64 + idx;
      if ((i<64*64) && (red[i].x != 0)) {
        for (uint32_t J=0;J<I;++J) {
          const uint32_t j = J * 64 + idx;
          if (red[j].x == red[i].x) {
            red[j].y += red[i].y;
            red[i].x = 0;
            break;
          } else if (red[j].x == 0) {
            red[j].y = red[i].y;
            red[i].x = 0;
            break;
          }
        }
      }
    }
    __syncthreads();
    */

    // write to memory
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
      for (uint32_t i=0;i<64;++i) {
        for (uint32_t j=0;j<64;++j) {
          const uint32_t p = i*64+j;
          const uint32_t r = red[p].x;
          if (r!=0) {
            igr[blockIdx.x * rnum + r] += red[p].y;
          } else {
            continue;
          }
        }
      }
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
class MDpairForce {
public:
  // cell length
  real cx, cy, cz, c0x, c0y, c0z, rmax2;
  unsigned short *typeID;
  Potential op;
  __device__ void operator()(const float4 _r, const float4 r, uint32_t i, uint32_t j, real *_f0) const {
    if (i!=j) {
      const real R2 = distance2p(_r.x, _r.y, _r.z, r.x, r.y, r.z, cx, cy, cz);
      if (R2<rmax2) {
        //real f = (R2<rmax2) ? op(R2, typeID_i, typeID[j]) : 0.0 ;
        real f = op.F(R2, typeID[i], typeID[j]);

        real _f[3];
        _f[0] = _r.x - r.x;
        _f[1] = _r.y - r.y;
        _f[2] = _r.z - r.z;
        if ((_f[0]*_f[0])>cx) _f[0] += (_f[0]<0) ? c0x : - c0x;
        if ((_f[1]*_f[1])>cy) _f[1] += (_f[1]<0) ? c0y : - c0y;
        if ((_f[2]*_f[2])>cz) _f[2] += (_f[2]<0) ? c0z : - c0z;

        const real _ff = rsqrt(R2);
        f *= _ff;
        _f[0] *= f;
        _f[1] *= f;
        _f[2] *= f;
#if defined(DUMPCOLLISION)
        if (((_f[0]*_f[0])>100)|| ((_f[1]*_f[1])>100)|| ((_f[2]*_f[2])>100)) {
          printf("XXX: %ld:%ld\t%g, %g, %g\t%g, %g, %g\t%g, %g, %g\t%g, %g,  %hd %hd\nXX2: \t%d:%d %d:%d %ld,%d,%ld,%d\n",
            i, j, r.x, r.y, r.z, _r.x, _r.y, _r.z,
            _f[0], _f[1], _f[2],
            sqrt(R2), f, typeID[i], typeID[j],
            I,J, i0, j0, bindex[I], bendI, bindex[J], bendJ);
        }
#endif
        _f0[0] += _f[0];
        _f0[1] += _f[1];
        _f0[2] += _f[2];
      }
    }
  }
};

/** calculates LJ potential
 * 
 */
template <typename Potential>
class MDpairPotential {
public:
  // cell length
  real cx, cy, cz, c0x, c0y, c0z, rmax2;
  unsigned short *typeID;
  Potential op;
  __device__ void operator()(const float4 _r, const float4 r, uint32_t i, uint32_t j, real *_f0) const {
    if (i!=j) {
      const real R2 = distance2p(_r.x, _r.y, _r.z, r.x, r.y, r.z, cx, cy, cz);
      if (R2<rmax2) {
        //real f = (R2<rmax2) ? op(R2, typeID_i, typeID[j]) : 0.0 ;
        _f0[0] = op(R2, typeID[i], typeID[j]);
        _f0[1] = _f0[2] = 0;
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
 * @param r array for position (size 3N)
 * @param w2D results array for kernel field
 * @param dW2D  results array for gradient of kernel dW/dr
 * @param N number of particles
 * @param h SPH kernel radius
 * @param w0  SPH kernel coefficient
 * @param w1  SPH kernel coefficient
 */
template <typename Kernel, typename KernelDW>
__global__ void calcSPHKernel(real *r, real *w2D, real *dW2D, uint32_t N, real h, real w0, real w1) {
  /* i-j pair are divided into (blockDim.x*4) x (blockDim.y*4) tiling
   * in each tile, threads checks all grid
   * cuda block X and Y corresponds to this tile
   */
  const uint32_t istart = (blockDim.x*4) * blockIdx.x;
  uint32_t iend = istart + (blockDim.x*4);
  if (iend>N) iend = N;
  const uint32_t N2 = N * 2;

  Kernel K1;
  KernelDW K2;

  for (uint32_t i=istart+threadIdx.x;i<iend;i+=blockDim.x) {
    const real _r[3] = {r[i], r[i+N], r[i+N2]};
    for (uint32_t j=0;j<N;++j) {
//      if (i!=j) {
        const real R = sqrt(distance2(_r[0], _r[1], _r[2], r[j], r[j+N], r[j+N2]));

        const real W  = K1(R, h, w0);
        w2D[i+j*N]  = W;
        const real dW = K2(R, h, w1);
        dW2D[i+j*N] = dW;

//      }
    }
  }
}

/** calculate mass and number density (without periodic boundary)
 *
 */
template <typename Kernel>
class SPHcalcDensity {
public:
  real *m;
  real h, w0;
  real *opt;

  Kernel K1;

  __device__ void operator()(const float4 ri, const float4 rj,
      uint32_t i, uint32_t j, real *_f0) const {
//  if (i!=j) {
    const real R = sqrt(distance2(ri.x, ri.y, ri.z, rj.x, rj.y, rj.z));
    const real W  = K1(R, h, w0);

    _f0[0] +=        W; // number for num[]
    _f0[1] += m[j] * W; // mass for rho[]
    if (opt!=NULL) _f0[2] += opt[j] * m[j] * W;
//  }
  }
};



/*
 * particleSPH_NS
 */
/** calculate accelerations by Naiver-Stokes equations
 *
 */
template <typename KernelDW>
class SPHNavierStokes {
public:
  float4 *v;
  real *m;
  real *rhoinv;
  float4 *num;
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
   *  \f$ p/rho/rho = Krho/rho/rho = K/rho; p1 = m_j K (1/rho_i + 1/rho_j) * dW/dR /r \f$
   *
   * dW/dR, and n are calculated by calcKernels() and calcDensity()
   */
  __device__ void operator()(const float4 ri, const float4 rj,
      uint32_t i, uint32_t j, real *_f0) const {
    if (i!=j) {
      const real R = sqrt(distance2(ri.x, ri.y, ri.z, rj.x, rj.y, rj.z));
      // m_j * dw/dr /r

      const real m_j = m[j] * K2(R, h, w1);
      const real p1 = (c2[i]*rhoinv[i]*(1-rho0*rhoinv[i]) + c2[j]*rhoinv[j]*(1-rho0*rhoinv[j]));
      const real v1 = (num[i].z+num[j].z) * rhoinv[i] * rhoinv[j];

      _f0[0] +=  m_j * ((rj.x - ri.x) * p1 - (v[j].x - v[i].x) * v1);
      _f0[1] +=  m_j * ((rj.y - ri.y) * p1 - (v[j].y - v[i].y) * v1);
      _f0[2] +=  m_j * ((rj.z - ri.z) * p1 - (v[j].z - v[i].z) * v1);
    }
  }
};

/** make list of particle j around i as colIdx[] by using Cutoff Block
 *
 * @param op  function object to judge the distance to register
 * @param rowPrt row pointer for each i particle
 * @param colIdx  results array, colIdx[rowPtr[i]..rowPtr[i+1]] stores ID of particle j around i
 * @param r array for position
 * @param blockOffset offset for the block ID table
 * @param blockNeighbor neighbor judgement for I-J block pair
 * @param pid particle ID in sorted order
 * @param bindex  start particle num of this block
 * @param N number of particles
 * @param selectedBlock table for the selected blocks
 */
template <typename Core>
__global__ void makeJlist_WithBlock_F4(const Core op, uint32_t *rowPtr, uint32_t *colIdx,
    float4 *r,
    uint32_t BlockOffset,
    uint32_t *blockNeighbor, uint32_t *pid, uint32_t *bindex,
    uint32_t N,
    uint32_t *selectedBlock
  ) {
  const uint32_t I =
    (selectedBlock!=NULL) ? selectedBlock[blockIdx.x+BlockOffset] :
    blockIdx.x + BlockOffset;

  if (bindex[I]==UINT_MAX) return;

  int __I = I+1;
  while (bindex[__I]==UINT_MAX) ++__I;
  const uint32_t bendI = bindex[__I];


  for (uint32_t i0=bindex[I]+threadIdx.x;i0<bendI;i0+=blockDim.x) {
    const uint32_t i = pid[i0];

    if (rowPtr[i+1]!=rowPtr[i]) {
      uint32_t *col = &(colIdx[rowPtr[i]]);

      // loop for all j particle in 27Blocks
      for (int ___J=0;___J<27;___J++) {
        const uint32_t J = blockNeighbor[I*27+___J];
        if (J==UINT_MAX) continue;
        if (bindex[J]==UINT_MAX) continue;

        int __J = J+1;
        while (bindex[__J]==UINT_MAX) ++__J;
        const uint32_t bendJ = bindex[__J];

        for (uint32_t j0=bindex[J];j0<bendJ;++j0) {
          const uint32_t j = pid[j0];

          if (i!=j) {
            // register if i--j is enough close
            if (op(r[i], r[j])) {
              *col++ = j;
            }
          }
        }
      }
    }
  }
}
#endif
