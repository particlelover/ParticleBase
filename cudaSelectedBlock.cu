#include "cudaSelectedBlock.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

cudaSelectedBlock::~cudaSelectedBlock() {
  if (selectedBlock!=NULL) cudaFree(selectedBlock);
}

void cudaSelectedBlock::setupCutoffBlock(real rmax, bool periodic) {
  cudaCutoffBlock::setupCutoffBlock(rmax, periodic);

  cudaMalloc((void **)&selectedBlock, sizeof(uint32_t)*totalNumBlock);
  if (withInfo) ErrorInfo("malloc selectedBlock[] on GPU");
}

void cudaSelectedBlock::selectBlocks(void) {
  clearArray<<<MPnum, THnum1D>>>(selectedBlock, totalNumBlock);

  //dim3 _thnum;
  //_thnum.x = THnum2D; _thnum.y = THnum2D; _thnum.z = 1;
  checkBlocks<<<MPnum, THnum1D>>>(selectedBlock, pid, move,
    bindex, totalNumBlock);
  if (withInfo) ErrorInfo("checkBlocks");

  accumulate_F4<<<1, threadsMax, sizeof(uint32_t)*threadsMax>>>(
    selectedBlock, totalNumBlock, tmp3N);
  real R=0.0;
  cudaMemcpy(&R, tmp3N, sizeof(real), cudaMemcpyDeviceToHost);
  numSelected = static_cast<long>(R);
  myBlockSelected = static_cast<int>(
    numSelected *
    (static_cast<real>(myBlockNum) / totalNumBlock)+0.5);
  if (withInfo) ErrorInfo("accumulate");

  writeBlockID<<<MPnum, THnum1D>>>(selectedBlock, totalNumBlock);
  if (withInfo) ErrorInfo("writeBlockID");

  // sort by Thrust
  thrust::device_ptr<uint32_t> dev_ptr(selectedBlock);
  thrust::sort(dev_ptr, &(dev_ptr[totalNumBlock]));
  if (withInfo) ErrorInfo("thrrust::sort");
}

void cudaSelectedBlock::getForceSelected(const ExchangeMode typeID) {
  assert(typeID!=ExchangeMode::torque);
  assert(typeID!=ExchangeMode::density);
  const size_t pstart = (typeID==ExchangeMode::coordnumber) ? p1 : p3;
  const size_t pend   = (typeID==ExchangeMode::coordnumber) ? p2 : p4;
  const size_t sizeN = sizeof(float4) * (pend-pstart);
  float4 *tgt = (typeID==ExchangeMode::acceleration) ? a : F;

  pthread_mutex_lock(&mutTMP);
  cudaMemcpy(&(TMP[pstart]), &(tgt[pstart]), sizeN, cudaMemcpyDeviceToHost);
  if (typeID!=ExchangeMode::coordnumber) {
  }
  pthread_mutex_unlock(&mutTMP);

  static const std::string reason[] = {"Force", "Acceleration", "Torque", "coordination number"};
  if (withInfo) ErrorInfo("do getForceSelected:"+reason[static_cast<int>(typeID)]);
}
void cudaSelectedBlock::importForceSelected(const cudaSelectedBlock &A, const ExchangeMode typeID,
  bool directAccess, int idMe, int idPeer) {
  const size_t pstart = (typeID==ExchangeMode::coordnumber) ? A.p1 : A.p3;
  const size_t pend   = (typeID==ExchangeMode::coordnumber) ? A.p2 : A.p4;
  const size_t sizeN = sizeof(float4) * (pend-pstart);
  float4 *tgt = (typeID==ExchangeMode::acceleration) ? a : F;

  if (directAccess) {
    float4 *src = (typeID==ExchangeMode::acceleration) ? A.a : A.F;
    cudaMemcpyPeer(&(tgt[pstart]), idMe, &(src[pstart]), idPeer, sizeN);
    cudaDeviceSynchronize();
  } else {
    cudaMemcpy(&(tgt[pstart]), &(A.TMP[pstart]), sizeN, cudaMemcpyHostToDevice);
  }

  static const std::string reason[] = {"Force", "Acceleration", "Torque", "coordination number"};
  if (withInfo) ErrorInfo("import the forces:"+reason[static_cast<int>(typeID)]);
}

void cudaSelectedBlock::setSelectedRange(uint32_t blockNum, uint32_t N, uint32_t myID) {
  calcBlockRange(blockNum, N, myID, [&](uint32_t offset, uint32_t num) {
    myOffsetSelected = offset;
    myBlockSelected  = num;
  });

  /*
  std::cerr << "[" << myID << ":" << myOffsetSelected << ":" << myBlockSelected << "]";
  std::cerr << " ";
  */
}
