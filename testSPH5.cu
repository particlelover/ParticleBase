#include <iostream>
#include "CUDAenv.hh"
#include "cudaParticleSPH_NS.hh"
#include "AdaptiveTime.hh"
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

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

  const double sph_h = 0.5;
  const double lunit = sph_h * 0.5938629;

  real cell[6] = {0, 20, 0, 20, 0, 10};

  const double rho_0 = 20.0 / (4.0/3.0*M_PI*(sph_h)*(sph_h)*(sph_h));
  const double m_0 = 1.0 / rho_0;
  // 0.9799 for 20m/(4/3 pi1.7^3)=1.0g/cm^3

  std::cerr << "SPH kernel h = " << (sph_h)
            << " lunit = " << lunit
            << " mean number density = " << rho_0
            << std::endl;


  GlobalTable G1;
  const signed int L1 = 20 / lunit - 1;
  const signed int L2 = 20 / lunit - 1;
  const signed int L3 = 10 / lunit - 1;
  const double lunit_h = lunit / 2;

  /*
   * border: 20x20x10 cm box
   */
  for (int i=0;i<L3;++i)
    for (int j=0;j<L2;++j)
      for (int k=0;k<L1;++k) {
        // i: Z, j: Y, k: X
        if (
          (i==0) || // floor
          ((j==0)||(j==L2-1)||(k==0)||(k==L1-1)) || // wall
          ((j==L2/2) && (i< std::min((-2.0*k + L3*3.0), (double)L3))) // wall2
        ) {
          for (int l=0;l<2;++l) { // two border particles set on same position
            ParticleBase pb;
            pb.r[0] = k*lunit+lunit_h;
            pb.r[1] = j*lunit+lunit_h;
            pb.r[2] = i*lunit+lunit_h;
            pb.m = m_0*2.5; // x2.5 by water
            pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
            pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
            pb.isFixed = true;
            pb.type = 1;
            G1.push_back(pb);              
          }
        }
      }
  uint32_t N1=G1.size();
  std::cerr << "N= " << N1 << std::endl;  


  /*
   * fluid particles
   */
  for (int i=2;i<L3;++i)
    for (int j=L2/2+2;j<L2-1;++j)
      for (int k=2;k<L3;++k) {
        // i: Z, j: Y, k: X
        ParticleBase pb;
        pb.r[0] = k*lunit;
        pb.r[1] = j*lunit;
        pb.r[2] = i*lunit;
        pb.m = m_0; // 21m/(4/3 pi0.5^3)=1.0g/cm^3
        pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
        pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
        pb.isFixed = false;
        pb.type = 0;
        G1.push_back(pb);
      }
  uint32_t N = G1.size();
  std::cerr << "N= " << N << std::endl;


  std::valarray<real> mu(8.9e-3, N);
  mu[std::slice(0, N1, 1)] = 1.0e5;
  std::valarray<real> c1(1.5e3/2, N);   //1500m/s = 1.5e5 cm/s for water
  c1[std::slice(0, N1, 1)] = 5.44e3/2;  // 5440m/s for glass

  const int ndev = particles.nDevices();
  particles.setup();
#pragma omp parallel for
  for (int i=0;i<ndev;++i) {
    particles.setGPU(i);
    particles[i].setup(N);
    particles[i].setCell(cell);
    particles[i].import(G1);
    particles[i].timestep = 0;
    particles[i].setSPHProperties(mu, c1, (sph_h));
    particles[i].setupCutoffBlock((sph_h), false);
  }
  std::cerr << "setup done" << std::endl;


  // putTMPselected
  particles[0].setupSelectedTMP(N1, N-N1, 0, N1);
  particles[0].putUnSelected("dump.SPH5box");
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
  particles[0].waitPutTMP();

  const real stepmax = 0.50;
  const real intaval  = 0.00200;
  const uint32_t initstep = particles[0].timestep;
  const real initDeltaT = 0.000050;
  const real ulim = 0.01 * 4;
  const real llim = ulim / 16.0;
  const real R0 = 0.50;
  std::vector<int> res(ndev);


#pragma omp parallel num_threads(ndev)
{
  AdaptiveTime<CUDAenv<cudaParticleSPH_NS>> thistime;
  real nextoutput = intaval;
  thistime.init(initDeltaT);
  thistime.statOutput = true;

#pragma omp master
  {
    thistime.PrintStat();
    std::cerr << "End Time: " << stepmax << std::endl;
  }

#pragma omp for
  for (int i=0;i<ndev;++i) {
    particles.setGPU(i);
    particles[i].calcVinit(initDeltaT);
  }

  uint32_t j = 0;

  while (thistime() < stepmax) {
#pragma omp master
    {
      if (thistime.isRollbacking()) {
        std::cerr << "now rollbacking:" << std::flush;
      }
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
      particles[i].TimeEvolution(thistime.currentDeltaT());

      real _r=0.0;
      uint32_t _r1 = 0;
      res[i] = particles[i].inspectVelocity((2*R0)/thistime.currentDeltaT(), ulim, llim, _r, _r1);
      //std::cerr << "TimeMesh: " << thistime << " " << deltaT << "\t" << _r
      //  << "\t" << _r * ((2*R0)/deltaT) << "\t" << _r1 << std::endl;
    }

    // progress simulation time
    j = thistime.Progress(particles, res, j);

#pragma omp for
    for (int i=0;i<ndev;++i) {
      particles[i].treatAbsoluteCondition();
    }

#pragma omp master
    if (thistime() >= nextoutput) {
      std::cerr << std::endl
            << "(" << thistime() << ") ";
      nextoutput += intaval;
      particles[0].timestep = j + 1 + initstep;
      particles[0].getSelectedPosition();
      particles[0].putTMP(std::cout);
    }
  }
  #pragma omp master
  {
    particles[0].waitPutTMP();
    thistime.PrintStat(j);
  }
}

  std::ofstream ofs("SPH5done");
  boost::archive::binary_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("cudaParticles", particles[0]);
  ofs.close();

  return 0;
}
