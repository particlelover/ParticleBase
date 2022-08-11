#include <iostream>
#include "CUDAenv.hh"
#include "cudaParticleSPH_NS.hh"
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>


#define SPH_H 1.7

typedef std::vector<class ParticleBase> GlobalTable;

void createInitialState(cudaParticleSPH_NS &particles) {
  /* radius of SPH partciles 1.7
   * => 4/3 pi 1.7^3 rho = 21; rho = 1.0204
   * border particles: 1/(0.980^3)
   * units of this simulation is:
   * length: cm
   * mass: g
   * time: s
   */

  const double lunit = (SPH_H)*0.5938629;

  const int lsize = 18;
  real cell[6] = {-2.0, 20.0, -2.0, 20.0, -2.0, 40.0};

  const double rho_0 = 20.0 / (4.0/3.0*M_PI*(SPH_H)*(SPH_H)*(SPH_H));
  const double m_0 = 1.0 / rho_0;
    // 0.9799 for 20m/(4/3 pi1.7^3)=1.0g/cm^3

  std::cerr << "SPH kernel h = " << (SPH_H)
            << " lunit = " << lunit
            << " mean number density = " << rho_0
            << std::endl;

  GlobalTable G1;
  for (int i=3;i<lsize-3;++i)
    for (int j=3;j<lsize-3;++j)
      for (int k=2;k<lsize-8;++k) {
        ParticleBase pb;
        pb.r[0] = i*lunit+lunit/2;
        pb.r[1] = j*lunit+lunit/2;
        //pb.r[2] = k*lunit+lunit/2;
        pb.r[2] = k*lunit+lunit/2+lunit*lsize*3/4;
        pb.m = m_0;
        pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
        pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
        pb.isFixed = false;
        pb.type = 0;
        G1.push_back(pb);
      }
  uint32_t N1=G1.size();
  std::cerr << "N= " << N1 << std::endl;


  // border
  for (int i=0;i<lsize;++i)
    for (int j=0;j<lsize;++j)
      for (int k=0;k<lsize*2;++k) {
        if ((i<2)||(lsize-3<i) ||
            (j<2)||(lsize-3<j) ||
            (k<2)||(lsize*2-3<k)) {
          ParticleBase pb;
          pb.r[0] = i*lunit+lunit/2;
          pb.r[1] = j*lunit+lunit/2;
          pb.r[2] = k*lunit+lunit/2;
          pb.m = m_0*2.5;   // x2.5 by water
          pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
          pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
          pb.isFixed = true;
          pb.type = 1;
          G1.push_back(pb);
        }
      }
  uint32_t N = G1.size();
  std::cerr << "N= " << N << std::endl;

  particles.setup(N);
  particles.setCell(cell);

  particles.import(G1);
  particles.timestep = 0;

  std::valarray<real> mu(8.9e-3, N);
  mu[std::slice(N1, N-N1, 1)] = 1.0e5;
  std::valarray<real> c1(1.5e3*2, N);     //1500m/s = 1.5e5 cm/s for water
  c1[std::slice(N1, N-N1, 1)] = 5.44e3*2; // 5440m/s for glass
  particles.setSPHProperties(mu, c1, (SPH_H));
  particles.setupCutoffBlock((SPH_H), false);
}

int main(int argc, char **argv) {
  class cudaParticleSPH_NS particles;
  {
    CUDAenv<cudaParticleSPH_NS> c;
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

    // calculate densities for the first output
    particles.calcBlockID();
    particles.calcDensity();
  }

  particles.getTypeID();
  particles.getPosition();
  particles.putTMP(std::cout);

  const real deltaT = 0.00025;
  const uint32_t stepmax = 1.00 / deltaT;
  const uint32_t intaval  = 0.0050 / deltaT;
  const uint32_t initstep = particles.timestep;

  particles.calcVinit(deltaT);

  for (uint32_t j=0;j<stepmax;++j) {
    std::cerr << j << " ";
    particles.calcBlockID();

    particles.calcKernels();  // do nothing
    particles.calcDensity();  // calc mass density field and its reciprocal 1/rho
    particles.calcForce();    // do nothing
    particles.calcAcceleration();
    particles.addAccelerationZ(-9.8e2);
    particles.TimeEvolution(deltaT);
    particles.treatAbsoluteCondition();
    if ((j+1)%intaval==0) {
      particles.timestep = j+1+initstep;
      particles.getPosition();
      particles.putTMP(std::cout);
    }
  }

  std::ofstream ofs("SPH1done");
  boost::archive::binary_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("cudaParticles", particles);
  ofs.close();

  return 0;
}
