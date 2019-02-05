#include <iostream>
#include "CUDAenv.hh"
#include "cudaParticleSPH_NS.hh"
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#define SPH_H 0.5

typedef std::vector<class ParticleBase> GlobalTable;

void createInitialState(CUDAenv<cudaParticleSPH_NS> &particles)
{
  /* radius of SPH partciles 0.5
   * => 4/3 pi 0.5^3 rho = 21; rho = 40.107
   * 1/l^3 = 40.107; 1/40.107 = l^3
   * border particles: 1/(0.292^3)
   * units of this simulation is:
   * length: cm
   * mass: g
   * time: s
   */

  /* 30x10x15cm box for inner
   * border thickness SPH_H*4 = 0.292*4=1.168
   * plus additional space for 1sphere width
   * 2.168 + inner + 2.168 for each axis
   */
  real cell[6] = {-(SPH_H)*5, 30.0 + (SPH_H)*5, -(SPH_H)*5, 10.0 + (SPH_H)*5, -(SPH_H)*5, 15.0 + (SPH_H)*5};
  const double lunit = (SPH_H)*0.5938629;

  const double rho_0 = 20.0 / (4.0 / 3.0 * M_PI * (SPH_H) * (SPH_H) * (SPH_H));
  const double m_0 = 1.0 / rho_0;
  // 0.9799 for 20m/(4/3 pi1.7^3)=1.0g/cm^3

  std::cerr << "SPH kernel h = " << (SPH_H)
            << " lunit = " << lunit
            << " mean number density = " << rho_0
            << std::endl;

  // 30x10x15 => 103x34x51 lattice
  // initial 25x34x45 for dam break

  const double lunit_h = lunit / 2;
  const signed int L1 = 30 / lunit;
  const signed int L2 = 10 / lunit;
  const signed int L3 = 15 / lunit;

  const signed int D1 = 6.5 / lunit;
  const signed int D3 = 13 / lunit;

  GlobalTable G1;
  for (int i = 1; i < D1; ++i)
    for (int j = 1; j < L2 + 1; ++j)
      for (int k = 1; k < D3; ++k)
      {
        ParticleBase pb;
        pb.r[0] = i * lunit + lunit_h;
        pb.r[1] = j * lunit + lunit_h;
        pb.r[2] = k * lunit + lunit_h;
        pb.m = m_0; // 21m/(4/3 pi0.5^3)=1.0g/cm^3
        pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
        pb.a[0] = 0.0;
        pb.a[1] = 0.0;
        pb.a[2] = 0.0;
        pb.isFixed = false;
        pb.type = 0;
        G1.push_back(pb);
      }
  uint32_t N1 = G1.size();
  std::cerr << "N= " << N1 << std::endl;

  // border
  for (int i = -2; i < L1 + 4; ++i)
    for (int j = -2; j < L2 + 4; ++j)
      for (int k = -2; k < L3 + 4; ++k)
      {
        if ((i < 0) || (L1 + 1 < i) ||
            (j < 0) || (L2 + 1 < j) ||
            (k < 0))
        {
          ParticleBase pb;
          pb.r[0] = i * lunit + lunit_h;
          pb.r[1] = j * lunit + lunit_h;
          pb.r[2] = k * lunit + lunit_h;
          pb.m = m_0 * 2.5; // x2.5 by water
          pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
          pb.a[0] = 0.0;
          pb.a[1] = 0.0;
          pb.a[2] = 0.0;
          pb.isFixed = true;
          pb.type = 1;
          G1.push_back(pb);
        }
      }
  uint32_t N = G1.size();
  std::cerr << "N= " << N << std::endl;

  std::valarray<real> mu(8.9e-3, N);
  mu[std::slice(N1, N - N1, 1)] = 1.0e5;
  std::valarray<real> c1(1.5e3 / 2, N);       //1500m/s = 1.5e5 cm/s for water
  c1[std::slice(N1, N - N1, 1)] = 5.44e3 / 2; // 5440m/s for glass

  const int ndev = particles.nDevices();
  particles.setup();
#pragma omp parallel for
  for (int i = 0; i < ndev; ++i)
  {
    particles.setGPU(i);
    particles[i].setup(N);
    particles[i].setCell(cell);
    particles[i].import(G1);
    particles[i].timestep = 0;
    particles[i].setSPHProperties(mu, c1, (SPH_H));
    particles[i].setupCutoffBlock((SPH_H), false);
  }
  std::cerr << "setup done" << std::endl;
}

int main(int argc, char **argv)
{
  class CUDAenv<cudaParticleSPH_NS> particles;
  const int ndev = particles.nDevices();

  if (argc == 2)
  {
    std::cerr << "reading serialization file " << argv[1] << std::endl;
    const int ndev = particles.nDevices();
    particles.setup();
    //#pragma omp parallel for
    for (int i = 0; i < ndev; ++i)
    {
      particles.setGPU(i);

      std::ifstream ifs(argv[1]);
      boost::archive::binary_iarchive ia(ifs);
      ia >> boost::serialization::make_nvp("cudaParticles", particles[i]);
      ifs.close();
    }
  }
  else
  {
    createInitialState(particles);

    for (int i = 0; i < ndev; ++i)
    {
      particles.setGPU(i);
      // calculate densities for the first output
      particles[i].calcBlockID();
      particles[i].calcDensity();
    }
  }

  uint32_t B0 = particles[0].numBlocks();
  uint32_t B1 = B0 / ndev;
  uint32_t B2 = 0;
  for (int i = 0; i < ndev; ++i)
  {
    particles[i].setMyBlock(B2, B1);
    std::cerr << "set block range for GPU " << i << ": " << B2 << " to " << B2 + B1 << std::endl;
    B2 += B1;
  }
  if ((B1 * ndev) != B0)
  {
    particles[ndev - 1].setMyBlock(B2 - B1, B1 + (B0 - (B1 * ndev)));
    std::cerr << "correct block range for GPU " << ndev - 1 << ": " << B2 - B1 << " to " << B2 + (B0 - (B1 * ndev)) << std::endl;
  }

  particles[0].getTypeID();
  particles[0].getPosition();
  particles[0].putTMP(std::cout);

  const real deltaT = 0.000050;
  const uint32_t stepmax = 0.50 / deltaT;
  const uint32_t intaval = 0.00100 / deltaT;
  const uint32_t initstep = particles[0].timestep;

#pragma omp parallel num_threads(ndev)
  {
#pragma omp for
    for (int i = 0; i < ndev; ++i)
    {
      particles.setGPU(i);
      particles[i].calcVinit(deltaT);
    }

    for (uint32_t j = 0; j < stepmax; ++j)
    {
#pragma omp master
      {
        std::cerr << j << " ";
      }
#pragma omp for
      for (int i = 0; i < ndev; ++i)
      {
        particles.setGPU(i);
        particles[i].calcBlockID();

        particles[i].calcKernels(); // do nothing
        particles[i].calcDensity(); // calc mass density field and its reciprocal 1/rho

        particles[i].selectBlocks();
      }

#pragma omp master
      {
        int _k = 0;
        for (int _n = 0; _n < ndev; _n++)
        {
          particles[_n].myOffsetSelected = _k;
          _k += particles[_n].myBlockSelected;
        }
        particles[ndev - 1].myBlockSelected += (particles[ndev - 1].numSelectedBlocks() - _k);
        /*
	  for (int _n=0;_n<ndev;_n++)
	    std::cerr << "[" << _n << ":" << particles[_n].myOffsetSelected << ":" << particles[_n].myBlockSelected << "]";
	  std::cerr << " ";
	  */
      }

#pragma omp for
      for (int i = 0; i < ndev; ++i)
      {
        particles[i].calcForce(); // do nothing
        particles[i].calcAcceleration();
      }

      if (ndev > 1)
      {
        particles.exchangeAccelerations();
      }

#pragma omp for
      for (int i = 0; i < ndev; ++i)
      {
        particles.setGPU(i);
        particles[i].addAccelerationZ(-9.8e2);
        particles[i].TimeEvolution(deltaT);
        particles[i].treatAbsoluteCondition();
      }

#pragma omp master
      if ((j + 1) % intaval == 0)
      {
        particles[0].timestep = j + 1 + initstep;
        particles[0].getPosition();
        particles[0].putTMP(std::cout);
      }
    }
  }

  std::ofstream ofs("SPH2done");
  boost::archive::binary_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("cudaParticles", particles[0]);
  ofs.close();

  return 0;
}
