#include "cudaCutoffBlock.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"
#include <iostream>

cudaCutoffBlock::~cudaCutoffBlock()
{
  if (bid != NULL)
    cudaFree(bid);
  if (pid != NULL)
    cudaFree(pid);
  if (bindex != NULL)
    cudaFree(bindex);
  if (blockNeighbor != NULL)
    cudaFree(blockNeighbor);
  if (tmp81N != NULL)
    cudaFree(tmp81N);
}

void cudaCutoffBlock::setupCutoffBlock(real rmax, bool periodic)
{
  std::cerr << "CutoffBlock::setup" << std::endl;

  // alloc
  cudaMalloc((void **)&bid, sizeof(uint32_t) * N);
  if (withInfo)
    ErrorInfo("malloc bid[] on GPU");

  cudaMalloc((void **)&pid, sizeof(uint32_t) * N);
  if (withInfo)
    ErrorInfo("malloc pid[] on GPU");

  std::vector<uint32_t> _tmp(N);
  for (uint32_t i = 0; i < N; ++i)
    _tmp[i] = i;

  cudaMemcpy(pid, &(_tmp[0]), sizeof(uint32_t) * N, cudaMemcpyHostToDevice);
  if (withInfo)
    ErrorInfo("copy pid to GPU");

  real c[3];
  std::cerr << "rmax: " << rmax << std::endl
            << "cell size is" << std::endl;
  for (int i = 0; i < 3; ++i)
  {
    std::cerr << cell[i * 2] << ":" << cell[i * 2 + 1] << std::endl;
  }
  for (int i = 0; i < 3; ++i)
  {
    c[i] = cell[i * 2 + 1] - cell[i * 2];
    if (periodic)
    {
      blocknum[i] = static_cast<uint32_t>(c[i] / rmax);
      blocklen[i] = c[i] / static_cast<real>(blocknum[i]);
    }
    else
    {
      // add a block outside of cell boundary
      blocklen[i] = rmax;
      blocknum[i] = static_cast<uint32_t>(c[i] / rmax) + 1;
    }

    std::cerr << c[i] << "  " << blocklen[i] << " x " << blocknum[i] << std::endl;
    assert(blocknum[i] > 2);

    // cell min for X,Y,Z
    blocklen[i + 3] = cell[i * 2];
  }

  if (SingleParticleBlock)
  {
    // in SingleParticleBlock Rmax(=blocklen) should be r0(radius of particle)*2/sqrt(3)
    const double r0 = rmax * sqrt(3) / 2.0;
    std::cerr << "r0: " << r0 << std::endl;
    const double R02 = (r0 * 2.0) * (r0 * 2.0);
    while (blocklen[0] * blocklen[0] + blocklen[1] * blocklen[1] + blocklen[2] * blocklen[2] > R02)
    {
      if ((blocklen[0] > blocklen[1]) && (blocklen[0] > blocklen[2]))
      {
        ++blocknum[0];
        blocklen[0] = c[0] / blocknum[0];
      }
      else if (blocklen[1] > blocklen[2])
      {
        ++blocknum[1];
        blocklen[1] = c[1] / blocknum[1];
      }
      else
      {
        ++blocknum[2];
        blocklen[2] = c[2] / blocknum[2];
      }
    }
    std::cerr << "(SingleParticleBlock) blocknum changed: ";
    for (int i = 0; i < 3; ++i)
    {
      std::cerr << blocknum[i] << "(" << blocklen[i] << ") ";
      cell[i * 2 + 1] = cell[i * 2] + blocklen[i] * blocknum[i];
    }
    std::cerr << std::endl;
  }

  totalNumBlock = static_cast<uint32_t>(blocknum[0]) * blocknum[1] * blocknum[2];
  assert(totalNumBlock < UINT_MAX);

  cudaMalloc((void **)&bindex, sizeof(uint32_t) * (totalNumBlock + 1));
  cudaMemcpy(&(bindex[totalNumBlock]), &(N), sizeof(uint32_t), cudaMemcpyHostToDevice);
  if (withInfo)
    ErrorInfo("malloc bindex[] on GPU");

  const int J27 = (SingleParticleBlock) ? 125 : 27;
  std::vector<uint32_t> blockIJtable(totalNumBlock * J27, 0);

  const int w = (SingleParticleBlock) ? 2 : 1;
#pragma omp parallel for
  for (uint32_t i = 0; i < blocknum[0]; ++i)
  {
    for (uint32_t j = 0; j < blocknum[1]; ++j)
    {
      for (uint32_t k = 0; k < blocknum[2]; ++k)
      {
        const uint32_t I = k + j * blocknum[2] + i * blocknum[2] * blocknum[1];

        uint32_t L = 0;
        for (int i1 = -w; i1 < w + 1; ++i1)
        {
          for (int j1 = -w; j1 < w + 1; ++j1)
          {
            for (int k1 = -w; k1 < w + 1; ++k1)
            {
              signed long i2 = i + i1;
              signed long j2 = j + j1;
              signed long k2 = k + k1;
              if (i2 == blocknum[0])
                i2 = 0;
              if (j2 == blocknum[1])
                j2 = 0;
              if (k2 == blocknum[2])
                k2 = 0;
              if (i2 == -1)
                i2 = blocknum[0] - 1;
              if (j2 == -1)
                j2 = blocknum[1] - 1;
              if (k2 == -1)
                k2 = blocknum[2] - 1;
              if (i2 == blocknum[0] + 1)
                i2 = 1;
              if (j2 == blocknum[1] + 1)
                j2 = 1;
              if (k2 == blocknum[2] + 1)
                k2 = 1;
              if (i2 == -2)
                i2 = blocknum[0] - 2;
              if (j2 == -2)
                j2 = blocknum[1] - 2;
              if (k2 == -2)
                k2 = blocknum[2] - 2;

              uint32_t J = UINT_MAX;
              if (periodic ||
                  ((((signed long)i + i1) >= 0) && ((i + i1) < blocknum[0]) &&
                   (((signed long)j + j1) >= 0) && ((j + j1) < blocknum[1]) &&
                   (((signed long)k + k1) >= 0) && ((k + k1) < blocknum[2])))
              {
                uint32_t _J = k2 + j2 * blocknum[2] + i2 * blocknum[2] * blocknum[1];
                assert(_J < UINT_MAX);
                J = _J;
              }
              assert(L < J27);
              blockIJtable[I * J27 + L] = J;
              ++L;
            }
          }
        }
      }
    }
  }

  cudaMalloc((void **)&blockNeighbor, sizeof(uint32_t) * totalNumBlock * J27);
  if (withInfo)
    ErrorInfo("malloc blockNeighbor[] on GPU");

  cudaMemcpy(blockNeighbor, &(blockIJtable[0]), sizeof(uint32_t) * totalNumBlock * J27, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&tmp81N, sizeof(real) * 3 * N * J27);
  if (withInfo)
    ErrorInfo("malloc tmp81N[] on GPU");

  std::cerr
      << "Total Number of Particles " << N
      << " / total number of blocks " << totalNumBlock
      << " mean " << (real)(N) / totalNumBlock
      << std::endl
      << "cuda grid for i-j pair table: " << totalNumBlock << " x " << J27
      //    << "\t with threads " << th << " x " << th
      << std::endl;

  myBlockOffset = 0;
  myBlockNum = totalNumBlock;

  THnum2D2 = THnum2D;
  {
    int _N = (real)(N) / totalNumBlock;
    while (THnum2D2 < _N)
      THnum2D2 *= 2;
  }
  std::cerr << "THnum2D2: " << THnum2D2 << std::endl;
}

void cudaCutoffBlock::calcBlockID(void)
{
  calcBID<<<MPnum, THnum1D>>>(r, bid, pid, N, blocklen[0], blocklen[1], blocklen[2],
                              blocklen[3], blocklen[4], blocklen[5],
                              blocknum[2], blocknum[1] * blocknum[2]);

  if (withInfo)
    ErrorInfo("calc BlockID");

  // sortByBlock(void)
  clearArray<<<MPnum, THnum1D>>>(bindex, totalNumBlock, UINT_MAX);

  sortByBID_M1<<<MPnum, THnum2D>>>(pid, bid, N);

  sortByBID<<<1, threadsMax>>>(pid, bid, N);
  makeBindex<<<MPnum, THnum1D>>>(bid, bindex, N);
  if (withInfo)
    ErrorInfo("sort By Block");
}

void cudaCutoffBlock::importAcceleration(const cudaCutoffBlock &A,
                                         bool directAccess, int idMe, int idPeer)
{
  const size_t sizeN = sizeof(real) * N * 3;
  if (directAccess)
  {
    cudaMemcpyPeer(tmp3N, idMe, A.a, idPeer, sizeN);
    cudaThreadSynchronize();
  }
  else
  {
    cudaMemcpy(tmp3N, &(A.TMP[0]), sizeN, cudaMemcpyHostToDevice);
  }

  addArray<<<MPnum, THnum1D>>>(a, tmp3N, N);
  if (withInfo)
    ErrorInfo("merge the acclerations");
}

void cudaCutoffBlock::importForce(const cudaCutoffBlock &A,
                                  bool directAccess, int idMe, int idPeer)
{
  const size_t sizeN = sizeof(real) * N * 3;
  if (directAccess)
  {
    cudaMemcpyPeer(tmp3N, idMe, A.F, idPeer, sizeN);
    cudaThreadSynchronize();
  }
  else
  {
    cudaMemcpy(tmp3N, &(A.TMP[0]), sizeN, cudaMemcpyHostToDevice);
  }

  addArray<<<MPnum, THnum1D>>>(F, tmp3N, N);
  if (withInfo)
    ErrorInfo("merge the forces");
}
