#include "cudaCutoffBlock.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"
#include <iostream>

/** calculates the blockNeighbor, array for the neighboring block id
 *
 * @param blockIJtable results array on GPU
 * @param w 2 for single particle block or 1 (plus minus 2 or plus minus 1)
 * @param periodic boundary condition
 * @param J27 number of neighboring block 27=3^3 or 125 = 5^3
 * @param blocknum_x number of block in the simulation cell in X direction
 * @param blocknum_x number of block in the simulation cell in Y direction
 * @param blocknum_x number of block in the simulation cell in Z direction
 */

__global__ void calc_BlockNeighbor(uint32_t *blockIJtable,
    int w, bool periodic, int J27,
    uint32_t blocknum_x, uint32_t blocknum_y, uint32_t blocknum_z) {
  // (1, Y, Z) thin 3D thread block is placed in direction to X
  for (uint32_t i=blockIdx.x;i<blocknum_x;i+=gridDim.x) {
    for (uint32_t j=threadIdx.y;j<blocknum_y;j+=blockDim.y) {
      for (uint32_t k=threadIdx.z;k<blocknum_z;k+=blockDim.z) {
        const uint32_t I = k  + j *blocknum_z + i *blocknum_z*blocknum_y;

        uint32_t L=0;
        for (int i1=-w;i1<w+1;++i1) {
          for (int j1=-w;j1<w+1;++j1) {
            for (int k1=-w;k1<w+1;++k1) {
              signed long i2=(int32_t)i+i1;
              signed long j2=(int32_t)j+j1;
              signed long k2=(int32_t)k+k1;
              if (i2==blocknum_x) i2=0;
              if (j2==blocknum_y) j2=0;
              if (k2==blocknum_z) k2=0;
              if (i2==-1) i2=blocknum_x-1;
              if (j2==-1) j2=blocknum_y-1;
              if (k2==-1) k2=blocknum_z-1;
              if (i2==blocknum_x+1) i2=1;
              if (j2==blocknum_y+1) j2=1;
              if (k2==blocknum_z+1) k2=1;
              if (i2==-2) i2=blocknum_x-2;
              if (j2==-2) j2=blocknum_y-2;
              if (k2==-2) k2=blocknum_z-2;

              uint32_t J=UINT_MAX;
              if (periodic ||
                  ((((signed long)i+i1)>=0)&&((i+i1)<blocknum_x)&&
                  (((signed long)j+j1)>=0)&&((j+j1)<blocknum_y)&&
                  (((signed long)k+k1)>=0)&&((k+k1)<blocknum_z))) {
                uint32_t _J = k2 + j2*blocknum_z + i2*blocknum_z*blocknum_y;
                assert(_J<UINT_MAX);
                J=_J;
              }
              assert(L<J27);
              blockIJtable[I*J27+L] = J;
              ++L;
            }
          }
        }

      }
    }
  }
}

cudaCutoffBlock::~cudaCutoffBlock() {
  if (bid!=NULL)    cudaFree(bid);
  if (pid!=NULL)    cudaFree(pid);
  if (bindex!=NULL) cudaFree(bindex);
  if (blockNeighbor!=NULL)  cudaFree(blockNeighbor);
  if (tmp81N!=NULL) cudaFree(tmp81N);
}

void cudaCutoffBlock::setupCutoffBlock(real rmax, bool periodic) {
  std::cerr << "CutoffBlock::setup" << std::endl;

  // alloc
  cudaMalloc((void **)&bid, sizeof(uint32_t)*N);
  if (withInfo) ErrorInfo("malloc bid[] on GPU");

  cudaMalloc((void **)&pid, sizeof(uint32_t)*N);
  if (withInfo) ErrorInfo("malloc pid[] on GPU");

  std::vector<uint32_t> _tmp(N);
  for (uint32_t i=0;i<N;++i) _tmp[i] = i;

  cudaMemcpy(pid, &(_tmp[0]), sizeof(uint32_t)*N, cudaMemcpyHostToDevice);
  if (withInfo) ErrorInfo("copy pid to GPU");

  real c[3];
  std::cerr << "rmax: " << rmax << std::endl
            << "cell size is" << std::endl;
  for (int i=0;i<3;++i) {
    std::cerr << cell[i*2] << ":" << cell[i*2+1] << std::endl;
  }
  for (int i=0;i<3;++i) {
    c[i] = cell[i*2+1] - cell[i*2];
    if (periodic) {
      blocknum[i] = static_cast<uint32_t>(c[i] / rmax);
      blocklen[i] = c[i] / static_cast<real>(blocknum[i]);
    } else {
      // add a block outside of cell boundary
      blocklen[i] = rmax;
      blocknum[i] = static_cast<uint32_t>(c[i] / rmax) + 1;
    }

    std::cerr << c[i] << "  " << blocklen[i] << " x " << blocknum[i] << std::endl;
    assert(blocknum[i]>2);

    // cell min for X,Y,Z
    blocklen[i+3] = cell[i*2];
  }

  if (SingleParticleBlock) {
    // in SingleParticleBlock Rmax(=blocklen) should be r0(radius of particle)*2/sqrt(3)
    const double r0 = rmax * sqrt(3) / 2.0;
    std::cerr << "r0: " << r0 << std::endl;
    const double R02 = (r0*2.0)*(r0*2.0);
    while (blocklen[0]*blocklen[0]+blocklen[1]*blocklen[1]+blocklen[2]*blocklen[2] > R02) {
      if ((blocklen[0] > blocklen[1]) && (blocklen[0] > blocklen[2])) {
        ++blocknum[0];
        blocklen[0] = c[0] / blocknum[0];
      } else if (blocklen[1] > blocklen[2]) {
        ++blocknum[1];
        blocklen[1] = c[1] / blocknum[1];
      } else {
        ++blocknum[2];
        blocklen[2] = c[2] / blocknum[2];
      }
    }
    std::cerr << "(SingleParticleBlock) blocknum changed: ";
    for (int i=0;i<3;++i) {
      std::cerr << blocknum[i] << "(" << blocklen[i] << ") ";
      cell[i*2+1] = cell[i*2] + blocklen[i]*blocknum[i];
    }
    std::cerr << std::endl;
  }


  totalNumBlock = static_cast<uint32_t>(blocknum[0]) * blocknum[1] * blocknum[2];
  assert(totalNumBlock < UINT_MAX);

  cudaMalloc((void **)&bindex, sizeof(uint32_t)*(totalNumBlock+1));
  cudaMemcpy(&(bindex[totalNumBlock]), &(N), sizeof(uint32_t), cudaMemcpyHostToDevice);
  if (withInfo) ErrorInfo("malloc bindex[] on GPU");

  const int J27 = (SingleParticleBlock) ? 125 : 27;

  cudaMalloc((void **)&blockNeighbor, sizeof(uint32_t)*totalNumBlock*J27);
  if (withInfo) ErrorInfo("malloc blockNeighbor[] on GPU");

  const int w = (SingleParticleBlock) ? 2 : 1;
  dim3 _thnum;
  _thnum.x = 1; _thnum.y = THnum2D; _thnum.z = THnum2D;
  calc_BlockNeighbor<<<MPnum, _thnum>>>(blockNeighbor, w, periodic, J27, blocknum[0], blocknum[1], blocknum[2]);
/*
  std::vector<uint32_t> blockIJtable(totalNumBlock*J27, 0);

#pragma omp parallel for
  for (uint32_t i=0;i<blocknum[0];++i) {
    for (uint32_t j=0;j<blocknum[1];++j) {
      for (uint32_t k=0;k<blocknum[2];++k) {
        const uint32_t I = k  + j *blocknum[2] + i *blocknum[2]*blocknum[1];

        uint32_t L=0;
        for (int i1=-w;i1<w+1;++i1) {
          for (int j1=-w;j1<w+1;++j1) {
            for (int k1=-w;k1<w+1;++k1) {
              signed long i2=(int32_t)i+i1;
              signed long j2=(int32_t)j+j1;
              signed long k2=(int32_t)k+k1;
              if (i2==blocknum[0]) i2=0;
              if (j2==blocknum[1]) j2=0;
              if (k2==blocknum[2]) k2=0;
              if (i2==-1) i2=blocknum[0]-1;
              if (j2==-1) j2=blocknum[1]-1;
              if (k2==-1) k2=blocknum[2]-1;
              if (i2==blocknum[0]+1) i2=1;
              if (j2==blocknum[1]+1) j2=1;
              if (k2==blocknum[2]+1) k2=1;
              if (i2==-2) i2=blocknum[0]-2;
              if (j2==-2) j2=blocknum[1]-2;
              if (k2==-2) k2=blocknum[2]-2;

              uint32_t J=UINT_MAX;
              if (periodic ||
                  ((((signed long)i+i1)>=0)&&((i+i1)<blocknum[0])&&
                  (((signed long)j+j1)>=0)&&((j+j1)<blocknum[1])&&
                  (((signed long)k+k1)>=0)&&((k+k1)<blocknum[2]))) {
                uint32_t _J = k2 + j2*blocknum[2] + i2*blocknum[2]*blocknum[1];
                assert(_J<UINT_MAX);
                J=_J;
              }
              assert(L<J27);
              blockIJtable[I*J27+L] = J;
              ++L;
            }
          }
        }
      }
    }
  }

  cudaMemcpy(blockNeighbor, &(blockIJtable[0]), sizeof(uint32_t)*totalNumBlock*J27, cudaMemcpyHostToDevice);
*/

  cudaMalloc((void **)&tmp81N, sizeof(real)*3*N*J27);
  if (withInfo) ErrorInfo("malloc tmp81N[] on GPU");

  std::cerr
    << "Total Number of Particles " << N
    << " / total number of blocks " << totalNumBlock
    << " mean "  << (real)(N) / totalNumBlock
    << std::endl
    << "cuda grid for i-j pair table: " << totalNumBlock << " x " << J27
//    << "\t with threads " << th << " x " << th
    << std::endl;

  myBlockOffset = 0;
  myBlockNum = totalNumBlock;

  THnum2D2 = THnum2D;
  {
    int _N = (real)(N) / totalNumBlock;
    while (THnum2D2<_N) THnum2D2 *= 2;
  }
  std::cerr << "THnum2D2: " << THnum2D2 << std::endl;

  uninitialized = false;
}

void cudaCutoffBlock::calcBlockID(void) {
  calcBID_F4<<<MPnum, THnum1D>>>(r, bid, pid, N, totalNumBlock,
    blocklen[0], blocklen[1], blocklen[2],
    blocklen[3], blocklen[4], blocklen[5],
    blocknum[0], blocknum[1], blocknum[2]);

  if (withInfo) ErrorInfo("calc BlockID");

  // sortByBlock(void)
  clearArray<<<MPnum, THnum1D>>>(bindex, totalNumBlock, UINT_MAX);

  int M1 = MPnum;
  int T1 = THnum2D;
  while ((M1*T1 > 32) && (M1 > 1) && (T1 > 1)) {
    sortByBID_M1<<<M1, T1>>>(pid, bid, N);
    if ((T1 * 2 )> M1) {
      T1 /= 2;
    } else {
      M1 /= 2;
    }
  }

  sortByBID<<<1, threadsMax>>>(pid, bid, N);
  makeBindex<<<MPnum, THnum1D>>>(bid, bindex, N);
  if (withInfo) ErrorInfo("sort By Block");
}

void cudaCutoffBlock::importAcceleration(const cudaCutoffBlock &A,
  bool directAccess, int idMe, int idPeer) {
  const size_t sizeN = sizeof(float4) * N;
  if (directAccess) {
    cudaMemcpyPeer(tmp3N, idMe, A.a, idPeer, sizeN);
    cudaDeviceSynchronize();
  } else {
    cudaMemcpy(tmp3N, &(A.TMP[0]), sizeN, cudaMemcpyHostToDevice);
  }

  addArray_F4<<<MPnum, THnum1D>>>(a, tmp3N, N);
  if (withInfo) ErrorInfo("merge the acclerations");
}

void cudaCutoffBlock::importForce(const cudaCutoffBlock &A,
  bool directAccess, int idMe, int idPeer) {
  const size_t sizeN = sizeof(float4) * N;
  if (directAccess) {
    cudaMemcpyPeer(tmp3N, idMe, A.F, idPeer, sizeN);
    cudaDeviceSynchronize();
  } else {
    cudaMemcpy(tmp3N, &(A.TMP[0]), sizeN, cudaMemcpyHostToDevice);
  }

  addArray_F4<<<MPnum, THnum1D>>>(F, tmp3N, N);
  if (withInfo) ErrorInfo("merge the forces");
}

void cudaCutoffBlock::switchBlockAlgorithm(const ParticleBlockType t) {
  if (uninitialized) {
    if (t==ParticleBlockType::many) {
      SingleParticleBlock = false;
    } else if (t==ParticleBlockType::single) {
      SingleParticleBlock = true;
    } else {
      std::cerr << "cudaCutoffBlock::switchBlockAlgorithm(): unsupported t=" << (int)t << std::endl;
    }
  } else {
    std::cerr << "cudaCutoffBlock::switchBlockAlgorithm() rejects" << std::endl;
  }
}

void cudaCutoffBlock::setBlockRange(uint32_t blockNum, uint32_t N, uint32_t myID) {
  calcBlockRange(blockNum, N, myID, [&](uint32_t offset, uint32_t num) {
    myBlockOffset = offset;
    myBlockNum    = num;
  });
}
