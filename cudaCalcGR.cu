#include "cudaCalcGR.hh"
#include <cmath>

__global__ void reduceigr(uint32_t *igr, const uint32_t rnum, const uint32_t MPnum) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  for (uint32_t j=thID;j<rnum;j+=thNum) {
    for (uint32_t i=1;i<MPnum;++i) {
      igr[j] += igr[j + i*rnum];
    }
  }
}

__global__ void kigr2gr(uint32_t *igr, const uint32_t rnum, real *gr, const real Q) {
  const uint32_t thID = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t thNum = gridDim.x * blockDim.x;
  /*
   * igr[] / N / (r^2 \Delta r) / \rho
   */
  const real onethrid = 1.0 / 3.0;
  for (uint32_t j=thID+1;j<rnum;j+=thNum) {
    const real rinv2 = 1.0 / (4*j*j + onethrid);
    gr[j] = static_cast<real>(igr[j]) * Q * rinv2;
  }
  if (threadIdx.x == 0) {
    gr[0] = 0;
  }
}

void cudaCalcGR::makePairInfo(const cudaParticleMD &P) {
  std::vector<uint32_t> lbindex;
  lbindex.resize(P.totalNumBlock + 1);
  cudaMemcpy(&(lbindex[0]), P.bindex, sizeof(uint32_t)*(P.totalNumBlock + 1), cudaMemcpyDeviceToHost);
  std::cerr << "totalNumBlock: " << P.totalNumBlock << std::endl;
  if (withInfo) ErrorInfo("memcpy bindex failed");

  long N = P.totalNumBlock * (P.totalNumBlock+1) / 2;
  pairInfo.reserve(N);
  pairInfo.resize(0);
  for (uint32_t I=0;I<P.totalNumBlock;++I) {
    if (lbindex[I]==UINT_MAX) continue;
    const uint32_t bstartI = lbindex[I];
    int __I = I+1;
    while (lbindex[__I]==UINT_MAX) ++__I;
    const uint32_t bendI = lbindex[__I];
    for (uint32_t J=I;J<P.totalNumBlock;++J) {
      if (lbindex[J]==UINT_MAX) continue;
      const uint32_t bstartJ = lbindex[J];
      int __J = J+1;
      while (lbindex[__J]==UINT_MAX) ++__J;
      const uint32_t bendJ = lbindex[__J];
      //std::cerr << bstartJ << ":" << bendJ << "  " << std::flush;

      // candidates I-J block pair
      uint32_t i = bstartI;
      do {
        uint32_t j = bstartJ;
        do {
          uint4 p;
          p.x = i; p.y = std::min(i+64, bendI);
          p.z = j; p.w = std::min(j+64, bendJ);
          assert(p.y-p.x <= 64);
          assert(p.w-p.z <= 64);
          pairInfo.push_back(p);

          j+=64;
        } while (j<bendJ);
        i+=64;
      } while (i<bendI);
    }
    //std::cerr << std::endl;
  }
  std::cerr << "N(N+1)/2: " << N << std::endl
            << "pairInfo size: " << pairInfo.size() << std::endl;
  if (pInfo != NULL) {
    cudaFree(pInfo);
  }
  if (withInfo) ErrorInfo("cudafree pInfo failed");
  
  const size_t psize = sizeof(uint4)*bunchsize;
  std::cerr << "pInfo size: " << psize << std::endl;
  cudaMalloc((void **)&pInfo, psize);
  if (withInfo) ErrorInfo("malloc pInfo failed");
}

void cudaCalcGR::calcgr(const cudaParticleMD &part) {
  for (int i=0;i<9;++i) std::cerr << part.cell[i] << " ";
  std::cerr << std::endl;
  class calcGR_F4 P;
  P.cx = part.cell[6];
  P.cy = part.cell[7];
  P.cz = part.cell[8];
  //P.typeID = typeID_s;
  P.rstepinv = rstepinv;
  P.rnum = rnum;

  dim3 _mpnum, _thnum;
  _mpnum.x = MPnum ; _mpnum.y = 1; _mpnum.z = 1;
  _thnum.x = THnum2D; _thnum.y = THnum2D; _thnum.z = 1;

  for (uint32_t i=0;i<pairInfo.size();i+=bunchsize) {
    std::cerr << i << "\t" << std::flush;
    const uint32_t psize = std::min(bunchsize, (pairInfo.size() - i));
    cudaMemcpy(pInfo, &(pairInfo[i]), sizeof(uint4)*psize, cudaMemcpyHostToDevice);
    if (withInfo) ErrorInfo("makePairInfo() failed");
    calcF_IJpairWithBlock5_F4<<<_mpnum, _thnum, sizeof(uint2)*4096>>>(P, part.r_s,
      pInfo, psize,
      igr, rnum
    );
    if (withInfo) ErrorInfo("calcgr() failed");
  }
  std::cerr << "done" << std::endl;
}

void cudaCalcGR::igr2gr(const cudaParticleMD &part) {
  reduceigr<<<MPnum, THnum1D>>>(igr, rnum, MPnum);
  if (withInfo) ErrorInfo("reduceigr failed");

  const real rhoinv = (part.cell[1]-part.cell[0]) * (part.cell[3]-part.cell[2]) * (part.cell[5]-part.cell[4]) / part.N;
  /*
   * Volume of spherical shell in the range (r-\Delta r/2, r+\Delta r/2) is
   * 4\pi / 3 (r+\Delta r/2)^3 - 4\pi / 3 (r-\Delta r/2)^3
   * = 4pi / 3 (3r^2 \Delta r + 2(\Delta r/2)^3)
   * 4\pi r^2 \Delta r + 1/3 \pi \Delta r^3,
   * 1/V = 1 / (\pi \Delta r^3(4 i^2 + 1/3))
   */
  const real Vinv = M_1_PI * rstepinv * rstepinv * rstepinv;
  const real Q = 2.0 / part.N * Vinv * rhoinv;

  kigr2gr<<<MPnum, THnum1D>>>(igr, rnum, gr, Q);
  if (withInfo) ErrorInfo("kigr2gr failed");
}

void cudaCalcGR::getGr(std::ostream &o) {
  size_t sizeN = sizeof(real) * rnum;

  pthread_mutex_lock(&mutTMP);
  cudaMemcpy(&(TMP[0]), gr, sizeN, cudaMemcpyDeviceToHost);
  for (uint32_t i=0;i<rnum;++i) {
    const real r = i / rstepinv;
    o << r << "  " << TMP[i] << std::endl;
  }
  pthread_mutex_unlock(&mutTMP);

  if (withInfo) ErrorInfo("do getGr");
}