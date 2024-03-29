#include <iostream>
#include "CUDAenv.hh"
#include "cudaParticleSPH_NS.hh"
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>


#define SPH_H 0.5

typedef std::vector<class ParticleBase> GlobalTable;


void createInitialState(CUDAenv<cudaParticleSPH_NS> &particles) {
  /* radius of SPH partciles 0.5
   * => 4/3 pi 0.5^3 rho = 21; rho = 40.107
   * 1/l^3 = 40.107; 1/40.107 = l^3
   * border particles: 1/(0.292^3)
   * units of this simulation is:
   * length: cm
   * mass: g
   * time: s
   */

  /* 40x10x40cm box for L-shaped
   * border thickness SPH_H*4 = 0.292*4=1.168
   * plus additional space for 1sphere width
   * 2.168 + inner + 2.168 for each axis
   */
  real cell[6] = {-(SPH_H)*5, 50.0+(SPH_H)*5, -(SPH_H)*5, 8.0+(SPH_H)*5, -(SPH_H)*5, 50.0+(SPH_H)*5};
  const double lunit = (SPH_H)*0.5938629;

  const double rho_0 = 20.0 / (4.0/3.0*M_PI*(SPH_H)*(SPH_H)*(SPH_H));
  const double m_0 = 1.0 / rho_0;
    // 0.9799 for 20m/(4/3 pi1.7^3)=1.0g/cm^3

  std::cerr << "SPH kernel h = " << (SPH_H)
            << " lunit = " << lunit
            << " mean number density = " << rho_0
            << std::endl;

  // 50x8x50 L-shaped

  const double lunit_h = lunit / 2;
  const signed int L1 = 50 / lunit;
  const signed int L2 = 8  / lunit;
  const signed int L3 = 50 / lunit;

  GlobalTable G1;
  for (int i=1;i<L1;++i)
    for (int j=1;j<L2+1;++j)
      for (int k=1;k<L2+1;++k) {
        // i: Z, j: Y, k: X
        ParticleBase pb;
        pb.r[0] = k*lunit+lunit_h;
        pb.r[1] = j*lunit+lunit_h;
        pb.r[2] = i*lunit+lunit_h;
        pb.m = m_0; // 21m/(4/3 pi0.5^3)=1.0g/cm^3
        pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
        pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
        pb.isFixed = false;
        pb.type = 0;
        G1.push_back(pb);
      }
  uint32_t N1=G1.size();
  std::cerr << "N= " << N1 << std::endl;


  // border
  for (int i=-2;i<L1+4;++i)
    for (int j=-2;j<L2+4;++j)
      for (int k=-2;k<L3+4;++k) {
        // i: Z, j: Y, k: X
        if (
          (i<0) ||
          ((L2+1<i)&&(i<L2+4)&&(L2+1<k)) ||
          ((j<0)&&((k<L2+4)||(i<L2+4))) ||
          ((L2+1<j)&&((k<L2+4)||(i<L2+4))) ||
          (k<0) ||
          (((L2+1<k)&&(k<L2+4))&&(L2+1<i)) ||
          ((L3+1<k)&&(i<L2+4))
          ) {
          ParticleBase pb;
          pb.r[0] = k*lunit+lunit_h;
          pb.r[1] = j*lunit+lunit_h;
          pb.r[2] = i*lunit+lunit_h;
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


  std::valarray<real> mu(8.9e-3, N);
  mu[std::slice(N1, N-N1, 1)] = 1.0e5;
  std::valarray<real> c1(1.5e3/2, N);   //1500m/s = 1.5e5 cm/s for water
  c1[std::slice(N1, N-N1, 1)] = 5.44e3/2; // 5440m/s for glass

  const int ndev = particles.nDevices();
  particles.setup();
#pragma omp parallel for
  for (int i=0;i<ndev;++i) {
    particles.setGPU(i);
    particles[i].setup(N);
    particles[i].setCell(cell);
    particles[i].import(G1);
    particles[i].timestep = 0;
    particles[i].setSPHProperties(mu, c1, (SPH_H));
    particles[i].setupCutoffBlock((SPH_H), false);
  }
  std::cerr << "setup done" << std::endl;


  // putTMPselected
  particles[0].setupSelectedTMP(0, N1, N1, N-N1);
  particles[0].putUnSelected("dump.SPH4box");
}


int main(int argc, char **argv) {
  class CUDAenv<cudaParticleSPH_NS> particles;
  const int ndev = particles.nDevices();

  if (argc==2) {
    std::cerr << "reading serialization file " << argv[1] << std::endl;
    const int ndev = particles.nDevices();
    particles.setup();
//#pragma omp parallel for
    for (int i=0;i<ndev;++i) {
      particles.setGPU(i);

      std::ifstream ifs(argv[1]);
      boost::archive::binary_iarchive ia(ifs);
      ia >> boost::serialization::make_nvp("cudaParticles", particles[i]);
      ifs.close();
    }
  } else {
    createInitialState(particles);

    std::cerr << "calc Block ID / calc Density" << std::endl;
#pragma omp parallel for
    for (int i=0;i<ndev;++i) {
      particles.setGPU(i);
      // calculate densities for the first output
      particles[i].calcBlockID();
      particles[i].calcDensity();
    }
    std::cerr << "initialize done" << std::endl;
  }

  for (int i=0;i<ndev;++i) {
    particles[i].setBlockRange(particles[i].numBlocks(), ndev, i);
  }

  particles[0].getSelectedTypeID();
  particles[0].getSelectedPosition();
  particles[0].putTMP(std::cout);

  const real deltaT = 0.000050;
  const uint32_t stepmax = 0.50 / deltaT;
  const uint32_t intaval  = 0.00200 / deltaT;
  const uint32_t initstep = particles[0].timestep;

#pragma omp parallel num_threads(ndev)
{
#pragma omp for
  for (int i=0;i<ndev;++i) {
    particles.setGPU(i);
    particles[i].calcVinit(deltaT);
  }

  for (uint32_t j=0;j<stepmax;++j) {
#pragma omp master
    {
      std::cerr << j << " ";
    }
#pragma omp for
    for (int i=0;i<ndev;++i) {
      //particles.setGPU(i);
      particles[i].calcBlockID();
      particles[i].getExchangePidRange1();  // obtain pid range [p1, p2)

      particles[i].calcKernels();   // do nothing
      particles[i].calcDensity(true); // calc mass density field and its reciprocal 1/rho
    }

    if (ndev>1) {
      particles.exchangeForceSelected(ExchangeMode::density); // exchange density
    }

#pragma omp for
    for (int i=0;i<ndev;++i) {
      particles[i].calcDensityPost(true);
      particles[i].selectBlocks();
      particles[i].setSelectedRange(particles[i].numSelectedBlocks(), ndev, i);
      particles[i].calcForce(); // do nothing
      particles[i].calcAcceleration(true);
      particles[i].getExchangePidRange2();  // obtain pid range [p3, p4)
    }

    if (ndev>1) {
      particles.exchangeForceSelected(ExchangeMode::acceleration); // exchange acceleration
    }

#pragma omp for
    for (int i=0;i<ndev;++i) {
      //particles.setGPU(i);
      particles[i].RestoreAcceleration();
      particles[i].addAccelerationZ(-9.8e2);
      particles[i].TimeEvolution(deltaT);
      particles[i].treatAbsoluteCondition();
    }

#pragma omp master
    if ((j+1)%intaval==0) {
      std::cerr << std::endl << "(" << (j+1)*deltaT << ") ";
      particles[0].timestep = j+1+initstep;
      particles[0].getSelectedPosition();
      particles[0].putTMP(std::cout);
    }
  }
}

  std::ofstream ofs("SPH4done");
  boost::archive::binary_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("cudaParticles", particles[0]);
  ofs.close();

  return 0;
}
