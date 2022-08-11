#include <iostream>
#include "CUDAenv.hh"
#include "GaussianThermo.hh"
#include "kernelfuncs.h"
#include "cudaCalcGR.hh"
#include <random>


int main(int argc, char **argv) {
  class GaussianThermo particles;
  {
    CUDAenv<GaussianThermo> c;
    c.setThnum(particles);
  }

  if (argc==2) {
    std::cerr << "reading serialization file " << argv[1] << std::endl;

    std::ifstream ifs(argv[1]);
    boost::archive::binary_iarchive ia(ifs);
    ia >> boost::serialization::make_nvp("cudaParticles", particles);
    ifs.close();

  } else {
      std::cerr << "specify the serialization file" << std::endl;
      return 0;
  }

  std::vector<real> LJpara;
  const int elemnum = 1;
  LJpara.resize(2*elemnum*elemnum);
  LJpara[0] = 3.941; LJpara[1] = 195.2*particles.kB;
  particles.setLJparams(LJpara, elemnum); // not used
  const real Temp = 311;

  particles.setupCutoffBlock(19.5);
  particles.calcBlockID();
  particles.adjustVelocities(Temp);
  particles.setBlockRange(particles.numBlocks(), 1, 0);  // ???

  class cudaCalcGR calcgr;
  {
    CUDAenv<cudaCalcGR> c;
    c.setThnum(calcgr);
    assert(c.getShMem() >= sizeof(uint2)*64*64);
  }
  calcgr.setup(0.1, 150.0);
  calcgr.makePairInfo(particles);
  calcgr.cleargr();
  calcgr.calcgr(particles);
  calcgr.igr2gr(particles);
  calcgr.getGr(std::cout);
  
  return 0;
}
