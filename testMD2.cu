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

void createInitialState(GaussianThermo &particles) {
  /*
   * liquid argon;
   * sigma = 3.4\AA; epsilon / kB = 120K
   *
   * units: \AA, fs, 1/Na g
   */
  std::mt19937 engine;

  GlobalTable G1;
  const real kB = 8.3145e-7;  // kB in \AA, fs, g/NA unit
  real cell[6] = {0.0, 60.0, 0.0, 60.0, 0.0, 60.0};
  // 1.5114g/cm^3; 39.95g/mol
  const double rho = 0.0228;
  uint32_t N = static_cast<uint32_t>(((cell[1]-cell[0])*(cell[3]-cell[2])*(cell[5]-cell[4])) * rho);
  const real Temp = 85.6;
  // m.p. 83.80K. b.p. 87.30K


  uint32_t n1, n2, n3;
  double l1, l2, l3;
  {
    const double vol = (cell[1]-cell[0])*(cell[3]-cell[2])*(cell[5]-cell[4]);
    const double l0 = pow(vol / N, 1.0/3.0);
    n1 = static_cast<uint32_t>((cell[1]-cell[0])/l0)+1;
    n2 = static_cast<uint32_t>((cell[3]-cell[2])/l0)+1;
    n3 = static_cast<uint32_t>((cell[5]-cell[4])/l0)+1;
    //std::cerr << n1 << "," << n2 << "," << n3 << std::endl;
    assert(n1*n2*n3>N);
    l1 = (cell[1]-cell[0])/n1;
    l2 = (cell[3]-cell[2])/n2;
    l3 = (cell[5]-cell[4])/n3;
  }


  for (int i=0;i<N;++i) {
    ParticleBase pb;

    real m1, m2, m3;
    m1 = l1*(i%n1+0.5);
    m2 = (((i-(i%n1))/n1)%n2+0.5)*l2;
    m3 = ((uint32_t)(i/(n1*n2))+0.5)*l3;
    pb.r[0] = m1;
    pb.r[1] = m2;
    pb.r[2] = m3;
/*
    pb.r[0] = RR.rnd(cell[1]-cell[0]);
    pb.r[1] = RR.rnd(cell[3]-cell[2]);
    pb.r[2] = RR.rnd(cell[5]-cell[4]);
*/
    // Ar
    pb.m = 39.95;
    pb.type = 0;
    std::normal_distribution<> t1(0.0, sqrt(kB * Temp / pb.m));
    pb.v[0] = t1(engine);
    pb.v[1] = t1(engine);
    pb.v[2] = t1(engine);
    pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
    G1.push_back(pb);
  }


  particles.setE(1e-10);
  particles.setup(N);
  particles.kB = kB;
  particles.setCell(cell);
  particles.timestep = 0;
  particles.rmax2 = 15*15;

  particles.import(G1);
  particles.setM();
}

// adjustVelocities + main loop(GaussianThermo; without constTemp(); scaleTemp() when temp>1.1)
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
    createInitialState(particles);
  }

  {
    std::vector<real> LJpara;
    const int elemnum = 1;
    LJpara.resize(2*elemnum*elemnum);
    //LJpara[0] = 3.4; LJpara[1] = 9.969e-5;  // for Ar-Ar (sigma, epsilon)
    LJpara[0] = 3.4; LJpara[1] = 120*particles.kB;
    particles.setLJparams(LJpara, elemnum);
  }
  const real Temp = 85.6;

#if defined(MVSTAT)
  std::ofstream mvstat;
  mvstat.open("mv2stat2");
#endif

  if (argc==1) {
    // if use with cutoff block
    particles.setupCutoffBlock(15.0);
    particles.calcBlockID();
#if defined(MVSTAT)
    particles.statMV2(mvstat);
#endif

    particles.adjustVelocities(Temp, true);
  }

  particles.getTypeID();
  particles.getPosition();
  particles.putTMP(std::cout);
#if defined(MVSTAT)
  particles.statMV2(mvstat);
#endif


  const real delta_t = 2.0;
  const uint32_t stepnum =
    static_cast<uint32_t>(32000 / delta_t); //32.0ps
  const uint32_t ointerval =
    static_cast<uint32_t>(100 / delta_t); // 100fs
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
    if ((t>Temp*1.2) || isnan(t)) {
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

  std::ofstream ofs("MD2done");
  boost::archive::binary_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("cudaParticles", particles);
  ofs.close();

  return 0;
}
