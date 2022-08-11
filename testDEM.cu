#include <iostream>
#include "CUDAenv.hh"
#include "cudaParticleDEM.hh"
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <random>

typedef std::vector<class ParticleBase> GlobalTable;

int main(void) {
  /*
   * units of this simulation is:
   * length: cm
   * mass: g
   * time: s
   */
  
  std::mt19937 engine;

  const real R0=1.0;
  GlobalTable G1;
  const int lsize = 19;
  const real lunit = 2 * R0;
  real cell[6] = {0.0, lunit*lsize, 0.0, lunit*lsize, 0.0, lunit*lsize};
  std::normal_distribution<> s1(0.0, 0.1);


  for (int i=3;i<lsize-3;i+=1)
    for (int j=3;j<lsize-3;j+=1)
      for (int k=3;k<lsize-3;k+=1) {
    ParticleBase pb;
	  pb.r[0] = i*lunit+lunit/2 + s1(engine);
	  pb.r[1] = j*lunit+lunit/2 + s1(engine);
	  pb.r[2] = k*lunit*1.5+lunit/2+s1(engine) + lunit*1.25;
    pb.m = 4.19;	// 4/3 pi 1.0^3
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
	if ((i<1)||(lsize-2<i) ||
	    (j<1)||(lsize-2<j) ||
	    (k<1)||(lsize*2-2<k)) {
	  ParticleBase pb;
	  pb.r[0] = i*lunit+lunit/2 + s1(engine);
	  pb.r[1] = j*lunit+lunit/2 + s1(engine);
	  pb.r[2] = k*lunit+lunit/2;
	  if (k<1) pb.r[2] += j * 0.2;
	  pb.m = 10.475;		// x2.5 by water
	  pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
	  pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
	  pb.isFixed = true;
	  pb.type = 1;
	  G1.push_back(pb);
	}
      }
  uint32_t N = G1.size();
  std::cerr << "N= " << N << std::endl;

  class cudaParticleDEM particles;
  {
    CUDAenv<cudaParticleDEM> c;
    c.setThnum(particles);
  }
  particles.setup(N);
  particles.setCell(cell);

  particles.import(G1);
  //particles.setDEMProperties(1.0e11, 0.5, 0.3, 0.9, 0.2,
  particles.setDEMProperties(4.0e4*2.5, 0.10, 0.45, 0.9, 0.01,
			     R0);
  particles.setInertia(R0);
  particles.setupCutoffBlock(R0*2, false);


  // calculate densities for the first output
  particles.calcBlockID();

  particles.getTypeID();
  particles.timestep = 0;
  particles.getPosition();
  particles.putTMP(std::cout);

  particles.checkPidRange();

  const real deltaT = 0.00005;
  const uint32_t stepmax = 0.75 / deltaT;
  const uint32_t intaval  = 0.010 / deltaT;
  const uint32_t initstep = particles.timestep;

  particles.calcVinit(deltaT);

  for (uint32_t j=0;j<stepmax;++j) {
    std::cerr << j << " ";
    particles.calcBlockID();

    particles.calcForce(deltaT);
    particles.calcAcceleration();
    particles.addAccelerationZ(-9.8e2);
    particles.TimeEvolution(deltaT);
//    particles.treatAbsoluteCondition();
    if ((j+1)%intaval==0) {
      particles.timestep = j+1+initstep;
      particles.getPosition();
      particles.putTMP(std::cout);
    }
  }

  std::ofstream ofs("DEM1done");
  boost::archive::binary_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("cudaParticles", particles);
  ofs.close();

  return 0;
}
