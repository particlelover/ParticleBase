#if !defined(CUDACALCGR)
#define CUDACALCGR

#include <vector>
#include "cudaParticleMD.hh"
#include "cudaFrame.hh"


class cudaCalcGR: public cudaFrame {
  public:
  cudaCalcGR(): pInfo(NULL), igr(NULL), gr(NULL) {};

  virtual ~cudaCalcGR() {
    if (igr != NULL) {
      cudaFree(igr);
    }
    if (gr != NULL) {
      cudaFree(gr);
    }
    if (pInfo != NULL) {
      cudaFree(pInfo);
    }
  }

  void setup(const real rstep, const real rmax) {
    r2max = rmax * rmax;
    rstepinv = 1.0 / rstep;
    rnum = static_cast<uint32_t>(rmax * rstepinv);

    bunchsize = 65536 * MPnum;
    std::cerr << "bunchsize: " << bunchsize << std::endl;

    cudaMalloc((void **)&igr, sizeof(uint32_t)*rnum*MPnum);
    cudaMalloc((void **)&gr, sizeof(real)*rnum);
    std::cerr << "rnum: " << rnum << std::endl;
    if (withInfo) ErrorInfo("setup() failed");
    
    TMP.resize(rnum);
  }

  void copyCell(const cudaParticleMD &P) {
    for (int i=0;i<9;++i) {
      cell[i] = P.cell[i];
    }
  }

  void makePairInfo(const cudaParticleMD &P);

  void calcgr(const cudaParticleMD &part);

  void cleargr(void) {
    clearArray<<<MPnum, THnum1D>>>(igr, rnum*MPnum);
    if (withInfo) ErrorInfo("cleargr() failed");
  }

  void igr2gr(const cudaParticleMD &part);

  void getGr(std::ostream &o=std::cout);

  private:
  real rstepinv;
  real r2max;
  uint32_t rnum;
  uint32_t *igr;
  real *gr;
  size_t bunchsize;

  std::vector<uint4> pairInfo;
  uint4 *pInfo;
};

#endif