#include "cudaSortedPositions.hh"

cudaSortedPositions::~cudaSortedPositions()
{
  if (r_s != NULL)
    cudaFree(r_s);
  if (typeID_s != NULL)
    cudaFree(typeID_s);
}

void cudaSortedPositions::setupSortedPositions(void)
{
  std::cerr << "SortedPositions::setup" << std::endl;

  // alloc
  cudaMalloc((void **)&r_s, sizeof(real) * N * 3);
  if (withInfo)
    ErrorInfo("malloc r_s[] on GPU");

  cudaMalloc((void **)&typeID_s, sizeof(unsigned short) * N);
  if (withInfo)
    ErrorInfo("malloc typeID_s[] on GPU");
}

template <typename T>
__global__ void copyWithIndex(T *r_src, T *r_dst, uint32_t num, uint32_t *id)
{
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t i = thID; i < num; i += thNum)
    r_dst[i] = r_src[id[i]];
}

void cudaSortedPositions::calcBlockID(void)
{
  copyWithIndex<<<MPnum, THnum1D>>>(r, r_s, N, pid);
  copyWithIndex<<<MPnum, THnum1D>>>(&(r[N]), &(r_s[N]), N, pid);
  copyWithIndex<<<MPnum, THnum1D>>>(&(r[N * 2]), &(r_s[N * 2]), N, pid);
  copyWithIndex<<<MPnum, THnum1D>>>(typeID, typeID_s, N, pid);
}
