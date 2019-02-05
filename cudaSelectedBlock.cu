#include "cudaSelectedBlock.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

cudaSelectedBlock::~cudaSelectedBlock()
{
  if (selectedBlock != NULL)
    cudaFree(selectedBlock);
}

void cudaSelectedBlock::setupCutoffBlock(real rmax, bool periodic)
{
  cudaCutoffBlock::setupCutoffBlock(rmax, periodic);

  cudaMalloc((void **)&selectedBlock, sizeof(uint32_t) * totalNumBlock);
  if (withInfo)
    ErrorInfo("malloc selectedBlock[] on GPU");
}

void cudaSelectedBlock::selectBlocks(void)
{
  clearArray<<<MPnum, THnum1D>>>(selectedBlock, totalNumBlock);

  //dim3 _thnum;
  //_thnum.x = THnum2D; _thnum.y = THnum2D; _thnum.z = 1;
  // for SPH, more fine tuning is till needed
  checkBlocks<<<MPnum, THnum1D>>>(selectedBlock, pid, move,
                                  bindex, totalNumBlock);
  if (withInfo)
    ErrorInfo("checkBlocks");

  accumulate<<<1, threadsMax, sizeof(uint32_t) * threadsMax>>>(
      selectedBlock, totalNumBlock, tmp3N);
  real R = 0.0;
  cudaMemcpy(&R, tmp3N, sizeof(real), cudaMemcpyDeviceToHost);
  numSelected = static_cast<long>(R);
  myBlockSelected = static_cast<int>(
      numSelected *
          (static_cast<real>(myBlockNum) / totalNumBlock) +
      0.5);
  if (withInfo)
    ErrorInfo("accumulate");

  writeBlockID<<<MPnum, THnum1D>>>(selectedBlock, totalNumBlock);
  if (withInfo)
    ErrorInfo("writeBlockID");

  // sort by Thrust
  thrust::device_ptr<uint32_t> dev_ptr(selectedBlock);
  thrust::sort(dev_ptr, &(dev_ptr[totalNumBlock]));
  if (withInfo)
    ErrorInfo("thrrust::sort");
}

void cudaSelectedBlock::getForceSelected(const int typeID)
{
  assert(typeID != 2);
  assert(typeID != 4);
  const size_t pstart = (typeID == 3) ? p1 : p3;
  const size_t pend = (typeID == 3) ? p2 : p4;
  const size_t sizeN = sizeof(real) * (pend - pstart);
  real *tgt = (typeID == 1) ? a : F;

  pthread_mutex_lock(&mutTMP);
  cudaMemcpy(&(TMP[pstart]), &(tgt[pstart]), sizeN, cudaMemcpyDeviceToHost);
  if (typeID != 3)
  {
    cudaMemcpy(&(TMP[pstart + N]), &(tgt[pstart + N]), sizeN, cudaMemcpyDeviceToHost);
    cudaMemcpy(&(TMP[pstart + N * 2]), &(tgt[pstart + N * 2]), sizeN, cudaMemcpyDeviceToHost);
  }
  pthread_mutex_unlock(&mutTMP);

  static const std::string reason[] = {"Force", "Acceleration", "Torque", "coordination number"};
  if (withInfo)
    ErrorInfo("do getForceSelected:" + reason[typeID]);
}
void cudaSelectedBlock::importForceSelected(const cudaSelectedBlock &A, const int typeID,
                                            bool directAccess, int idMe, int idPeer)
{
  const size_t pstart = (typeID == 3) ? A.p1 : A.p3;
  const size_t pend = (typeID == 3) ? A.p2 : A.p4;
  const size_t sizeN = sizeof(real) * (pend - pstart);
  real *tgt = (typeID == 1) ? a : F;

  if (directAccess)
  {
    real *src = (typeID == 1) ? A.a : A.F;
    cudaMemcpyPeer(&(tgt[pstart]), idMe, &(src[pstart]), idPeer, sizeN);
    if (typeID != 3)
    {
      cudaMemcpyPeer(&(tgt[pstart + N]), idMe, &(src[pstart + N]), idPeer, sizeN);
      cudaMemcpyPeer(&(tgt[pstart + N * 2]), idMe, &(src[pstart + N * 2]), idPeer, sizeN);
    }
    cudaThreadSynchronize();
  }
  else
  {
    cudaMemcpy(&(tgt[pstart]), &(A.TMP[pstart]), sizeN, cudaMemcpyHostToDevice);
    if (typeID != 3)
    {
      cudaMemcpy(&(tgt[pstart + N]), &(A.TMP[pstart + N]), sizeN, cudaMemcpyHostToDevice);
      cudaMemcpy(&(tgt[pstart + N * 2]), &(A.TMP[pstart + N * 2]), sizeN, cudaMemcpyHostToDevice);
    }
  }

  static const std::string reason[] = {"Force", "Acceleration", "Torque", "coordination number"};
  if (withInfo)
    ErrorInfo("import the forces:" + reason[typeID]);
}
