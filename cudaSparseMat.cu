#include "cudaSparseMat.hh"
#include "kernelfuncs.h"

#include <iostream>

#define RESIZE_PITCH 512

cudaSparseMat::~cudaSparseMat()
{
  if (rowPtr != NULL)
    cudaFree(rowPtr);
  if (colIdx != NULL)
    cudaFree(colIdx);
  if (val != NULL)
    cudaFree(val);
}

void cudaSparseMat::setup(uint32_t N, uint32_t threadsMax)
{
  cudaMalloc((void **)&rowPtr, sizeof(uint32_t) * (N + 1));

  _N = N;
  TH = std::min(static_cast<uint32_t>(sqrt(N)) + 1, threadsMax);
}

uint32_t cudaSparseMat::makeRowPtr(void)
{
  partialsum<<<1, TH, sizeof(uint32_t) * (TH + 1)>>>(rowPtr, _N + 1);

  uint32_t res = 0;
  cudaMemcpy(&res, &(rowPtr[_N]), sizeof(uint32_t), cudaMemcpyDeviceToHost);

  return res;
}

void cudaSparseMat::Resize(uint32_t n)
{
  if (NNZ < n)
  {
    //    std::cerr << "Resize: " << NNZ << " to " << n;
    NNZ = n;
    if (nmax < n)
    {
      if (colIdx != NULL)
        cudaFree(colIdx);
      if (val != NULL)
        cudaFree(val);
      size_t NNZ_new = n;
      NNZ_new /= RESIZE_PITCH;
      NNZ_new += 1;
      NNZ_new *= RESIZE_PITCH;

      size_t pitch = 0, p2 = 0;
      cudaMallocPitch(&colIdx, &pitch, sizeof(uint32_t) * NNZ_new, 1);
      cudaMallocPitch(&val, &p2, sizeof(real) * NNZ_new, 1);
      //cudaThreadSynchronize();
      cudaError_t t = cudaGetLastError();
      if (t != 0)
      {
        std::cerr << "SparseMat Resize"
                  << ": "
                  << cudaGetErrorString(t) << std::endl;
      }

      uint32_t n2 = pitch / sizeof(uint32_t);
      uint32_t n3 = p2 / sizeof(real);
      std::cerr << "Resize: " << nmax << " to " << n2
                << " (pitch: " << pitch << ") ";
      if (n2 != n3)
        std::cerr << "!= " << n3 << " ";
      nmax = std::min(n2, n3);
      //      std::cerr << std::endl;
    }
  }
}
