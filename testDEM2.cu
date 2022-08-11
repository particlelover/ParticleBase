#include <iostream>
#include "CUDAenv.hh"
#include "cudaParticleDEM.hh"
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <math.h>
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

  const real R0=0.65;
  GlobalTable G1;
  const real L1 = 40.0;
  const real L2 = 100.0;
  const real lunit = 2 * R0;
  const int lsize = L1 / (lunit) - 1;
  const int lsize2 = L2 / (lunit) - 1;
  real cell[6] = {0.0, lunit*lsize2, 0.0, lunit*lsize, 0.0, lunit*2*lsize};

  real WeighFe = 7.874 * 4.0 / 3.0 * M_PI * R0*R0*R0;

/*
  for (int i=2;i<lsize-2;i+=1)
    for (int j=2;j<lsize-2;j+=1)
      for (int k=2+4;k<lsize-2+4;k+=1) {
*/
  std::normal_distribution<> s1(0.0, R0 / 10.0);
  for (int i=0;i<(lsize-4)/1.2;i+=1)
    for (int j=0;j<(lsize-4)/1.2;j+=1)
      for (int k=0;k<(lsize-4)/1.2;k+=1) {
    ParticleBase pb;
	  pb.r[0] = (i*1.2+2)*lunit+lunit/2 + s1(engine) +12.0;
	  pb.r[1] = (j*1.2+2)*lunit+lunit/2 + s1(engine);
	  pb.r[2] = (k*1.2+6)*lunit*1.5+lunit/2 + s1(engine);
    pb.m = WeighFe;
    pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
    pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
    pb.isFixed = false;
    pb.type = 0;
    G1.push_back(pb);
      }
  uint32_t N1=G1.size();
  std::cerr << "N= " << N1 << std::endl;


  // border
  std::normal_distribution<> s2(0.0, R0 / 100.0);
  for (int i=0;i<lsize2;++i)
    for (int j=0;j<lsize;++j)
      for (int k=0;k<lsize*0.6;++k) {
	if (( ((i<1)||(lsize2-2<i) ||
	       (j<1)||(lsize-2<j )) && (k>3*log(1.8-1.6*(real)(i)/(real)(lsize2))+5) )
	    ||
	    ((k==0) && ((0<i)&&(i<lsize2-1)&&(0<j)&&(j<lsize-1)) )
          ) {
	  ParticleBase pb;
	  pb.r[0] = i*lunit+lunit/2 + s2(engine);
	  pb.r[1] = j*lunit+lunit/2 + s2(engine);
	  if (k==0) 
	  pb.r[2] = (3*log(1.8-1.6*(real)(i)/(real)(lsize2))+5)*lunit+lunit/2;
	  else
	  pb.r[2] = k*lunit+lunit/2;
	  if (k<1) pb.r[2] += k*lunit+lunit/2;
	  pb.m = WeighFe;
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
  //particles.setDEMProperties(1.0e5, 0.10, 0.45, 0.9, 0.05,
  // Fe: Young Modulus 2.11GPa, Poisson ratio 0.29, density 7.874gcm-3
  //particles.setDEMProperties(2.11e07, 0.40, 0.29, 0.9, 0.10,
  particles.setDEMProperties(2.11e07, 0.40, 0.29, 0.45, 0.10,
			     R0);
  particles.setInertia(R0);
  particles.setupCutoffBlock(R0*2/sqrt(3.0), false);

  // putTMPselected
  particles.setupSelectedTMP(0, N1, N1, N-N1);
  particles.putUnSelected("DEM2box.dump");



  // calculate densities for the first output
  particles.calcBlockID();

  particles.getSelectedTypeID();
  particles.timestep = 0;
  particles.getSelectedPosition();
  particles.putTMP(std::cout);

  particles.checkPidRange();

  const real deltaT = 0.0000015;
  const uint32_t stepmax = 1.50 / deltaT;
  const uint32_t intaval  = 0.005 / deltaT;
  const uint32_t initstep = particles.timestep;

  particles.selectBlocks();
  particles.calcVinit(deltaT);

  for (uint32_t j=0;j<stepmax;++j) {
    std::cerr << j << " ";
    particles.calcBlockID();


    particles.selectBlocks();

    particles.calcForce(deltaT);
    particles.calcAcceleration();
    particles.addAccelerationZ(-9.8e2);
    particles.TimeEvolution(deltaT);
//    particles.treatAbsoluteCondition();
    if ((j+1)%intaval==0) {
      particles.timestep = j+1+initstep;
      particles.getSelectedPosition();
      particles.putTMP(std::cout);
    }
  }

  std::ofstream ofs("DEM2done");
  boost::archive::binary_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("cudaParticles", particles);
  ofs.close();

  return 0;
}
