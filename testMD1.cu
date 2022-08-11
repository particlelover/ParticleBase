#include <iostream>
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "CUDAenv.hh"
#include "GaussianThermo.hh"
#include "kernelfuncs.h"
#include <random>

//#define MVSTAT

typedef std::vector<class ParticleBase> GlobalTable;

void createInitialState(cudaParticleMD &particles) {
  /*
   * gas; N2/O2 at 80C
   *
   * units: \AA, fs, 1/Na g
   */
  std::mt19937 engine;

  GlobalTable G1;
  const real kB = 8.3145e-7;  // kB in \AA, fs, g/NA unit
  real cell[6] = {0.0, 800.0, 0.0, 800.0, 0.0, 800.0};
  const double rho = 0.00002074;
  uint32_t N = static_cast<uint32_t>(((cell[1]-cell[0])*(cell[3]-cell[2])*(cell[5]-cell[4])) * rho);
  const real Temp = 353;

  std::uniform_real_distribution<> s0(0.0, 1.0);
  std::uniform_real_distribution<> s1(0.0, cell[1]-cell[0]);
  std::uniform_real_distribution<> s2(0.0, cell[3]-cell[2]);
  std::uniform_real_distribution<> s3(0.0, cell[5]-cell[4]);

  for (int i=0;i<N;++i) {
    ParticleBase pb;
    pb.r[0] = s1(engine);
    pb.r[1] = s2(engine);
    pb.r[2] = s3(engine);
    if (s0(engine)<0.2) {
      // 20% for O2
      pb.m = 32.0;
      pb.type = 0;
    } else {
      // others, N2
      pb.m = 28.0;
      pb.type = 1;
    }
    // T=90K
    // T=353K
    std::normal_distribution<> t1(0.0, sqrt(kB * Temp / pb.m));
    pb.v[0] = t1(engine);
    pb.v[1] = t1(engine);
    pb.v[2] = t1(engine);
    pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
    G1.push_back(pb);
  }


  //class GaussianThermo particles;
  //particles.setE(1e-13);
  particles.setup(N);
  particles.kB = kB;
  particles.setCell(cell);
  particles.timestep = 0;
  particles.rmax2 = 40*40;

  particles.import(G1);
  particles.setM();
}

// adjustVelocities + initialAnnealing(with constTemp()) + main loop(without constTemp(); scaleTemp() when temp>1.1)
int main(int argc, char **argv) {
  class cudaParticleMD particles;
  {
    CUDAenv<cudaParticleMD> c;
    c.setThnum(particles);
  }

  if (argc==2) {
    std::cerr << "reading serialization file " << argv[1] << std::endl;
    std::ifstream ifs(argv[1]);
    boost::archive::binary_iarchive ia(ifs);
    ia >> boost::serialization::make_nvp("cudaParticles", particles);
    ifs.close();
  } else {
    createInitialState(particles);
  }

  {
    std::vector<real> LJpara;
    const int elemnum = 2;
    LJpara.resize(2*elemnum*elemnum);
    LJpara[0] = 3.1062; LJpara[1] = 3.59044e-5; // for O2-O2 (sigma, epsilon)
    LJpara[2] = 3.21365;LJpara[3] = 4.82095e-5; // for O2-N2 (sigma, epsilon)
    LJpara[4] = 3.21365;LJpara[5] = 4.82095e-5; // for N2-O2 (sigma, epsilon)
    LJpara[6] = 3.3211; LJpara[7] = 4.19355e-5; // for N2-N2 (sigma, epsilon)
    particles.setLJparams(LJpara, elemnum);
  }
  const real Temp = 353;

#if defined(MVSTAT)
  std::ofstream mvstat;
  mvstat.open("mv2stat");
#endif

  if (argc==1) {
  // if use with cutoff block
  particles.setupCutoffBlock(80.0);
  particles.calcBlockID();
#if defined(MVSTAT)
  particles.statMV2(mvstat);
#endif

  particles.adjustVelocities(Temp, true);
  particles.initialAnnealing(1000, 1.0, 5.0, 1e-1, Temp);
  }

  particles.getTypeID();
  particles.getPosition();
  particles.putTMP(std::cout);
#if defined(MVSTAT)
  particles.statMV2(mvstat);
#endif

  const real delta_t = 5.0;
  const uint32_t stepnum = 
    static_cast<uint32_t>(100000 / delta_t);  //100ps
  const uint32_t ointerval =
    static_cast<uint32_t>(800 / delta_t); // 800fs
  const uint32_t initstep = particles.timestep;


  particles.clearForce();
  // MAIN LOOP
  for (uint32_t j=0;j<stepnum;++j) {
    particles.calcBlockID();

    particles.calcForce();
    //real t = particles.constTemp();
    //std::cerr << "constTemp T= " << t << std::endl;
    real t = particles.calcTemp();
    std::cerr << "T= " << t << std::endl;
    if (t>Temp*1.2) {
#if defined(MVSTAT)
      particles.statMV2(mvstat);
#endif
      std::cerr << "abort" << std::endl;
      exit(1);
    }
    if (t>Temp*1.1) 
      std::cerr << "scaling temp by: " << particles.scaleTemp(Temp)
                << std::endl;
    particles.TimeEvolution(delta_t);
    particles.treatPeriodicCondition();

    if ((j+1)%25==0) {
      std::cerr << "K-P\t" << 
      particles.calcKineticE() << "\t" << particles.calcPotentialE() << std::endl;
    }

    if ((j+1)%ointerval==0) {
      particles.timestep = j+1+initstep;
      particles.getPosition();
      particles.putTMP(std::cout);
#if defined(MVSTAT)
      particles.statMV2(mvstat);
#endif
    }
  }

#if defined(MVSTAT)
  mvstat.close();
#endif

  std::ofstream ofs("MD1done");
  boost::archive::binary_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("cudaParticles", particles);
  ofs.close();

  return 0;
}
