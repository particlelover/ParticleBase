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

void createInitialState(CUDAenv<GaussianThermo> &particles)
{
  /*
   * liquid argon;
   * sigma = 3.4\AA; epsilon / kB = 120K
   *
   * units: \AA, fs, 1/Na g
   */
  std::mt19937 engine;

  GlobalTable G1;
  const real kB = 8.3145e-7; // kB in \AA, fs, g/NA unit
  real cell[6] = {0.0, 120.0, 0.0, 120.0, 0.0, 120.0};
  // 1.5114g/cm^3; 39.95g/mol
  const double rho = 0.0228;
  uint32_t N = static_cast<uint32_t>(((cell[1] - cell[0]) * (cell[3] - cell[2]) * (cell[5] - cell[4])) * rho);
  const real Temp = 85.6;
  // m.p. 83.80K. b.p. 87.30K

  uint32_t n1, n2, n3;
  double l1, l2, l3;
  {
    const double vol = (cell[1] - cell[0]) * (cell[3] - cell[2]) * (cell[5] - cell[4]);
    const double l0 = pow(vol / N, 1.0 / 3.0);
    n1 = static_cast<uint32_t>((cell[1] - cell[0]) / l0) + 1;
    n2 = static_cast<uint32_t>((cell[3] - cell[2]) / l0) + 1;
    n3 = static_cast<uint32_t>((cell[5] - cell[4]) / l0) + 1;
    //    std::cerr << n1 << "," << n2 << "," << n3 << std::endl;
    assert(n1 * n2 * n3 > N);
    l1 = (cell[1] - cell[0]) / n1;
    l2 = (cell[3] - cell[2]) / n2;
    l3 = (cell[5] - cell[4]) / n3;
  }

  for (int i = 0; i < N; ++i)
  {
    ParticleBase pb;

    real m1, m2, m3;
    m1 = l1 * (i % n1 + 0.5);
    m2 = (((i - (i % n1)) / n1) % n2 + 0.5) * l2;
    m3 = ((uint32_t)(i / (n1 * n2)) + 0.5) * l3;
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
    pb.a[0] = 0.0;
    pb.a[1] = 0.0;
    pb.a[2] = 0.0;
    G1.push_back(pb);
  }

  const int ndev = particles.nDevices();
  particles.setup();
#pragma omp parallel for
  for (int i = 0; i < ndev; ++i)
  {
    particles.setGPU(i);
    particles[i].setE(1e-12);
    particles[i].setup(N);
    particles[i].kB = kB;
    particles[i].setCell(cell);
    particles[i].timestep = 0;
    particles[i].rmax2 = 15 * 15;

    particles[i].import(G1);
    particles[i].setM();
  }
}

int main(int argc, char **argv)
{
  class CUDAenv<GaussianThermo> particles;

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
  }

  const int ndev = particles.nDevices();
#pragma omp parallel for
  for (int i = 0; i < ndev; ++i)
  {
    particles.setGPU(i);
    std::vector<real> LJpara;
    const int elemnum = 1;
    LJpara.resize(2 * elemnum * elemnum);
    //    LJpara[0] = 3.4; LJpara[1] = 9.969e-5;	// for Ar-Ar (sigma, epsilon)
    LJpara[0] = 3.4;
    LJpara[1] = 120 * particles[i].kB;
    particles[i].setLJparams(LJpara, elemnum);
  }
  const real Temp = 85.6;

#if defined(MVSTAT)
  std::ofstream mvstat;
  mvstat.open("mv2stat3");
#endif

  if (argc == 1)
  {
    // if use with cutoff block
#pragma omp parallel for
    for (int i = 0; i < ndev; ++i)
    {
      particles.setGPU(i);
      particles[i].setupCutoffBlock(15.0);
      particles[i].calcBlockID();
#if defined(MVSTAT)
      particles[i].statMV2(mvstat);
#endif

      particles[i].adjustVelocities(Temp);
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
#if defined(MVSTAT)
  particles[0].statMV2(mvstat);
#endif

  const real delta_t = 2.0;
  const uint32_t stepnum =
      static_cast<uint32_t>(80000 / delta_t); //80.0ps
  const uint32_t ointerval =
      static_cast<uint32_t>(100 / delta_t); // 100fs
  const uint32_t initstep = particles[0].timestep;

  // MAIN LOOP
  std::cerr << "start main loop" << std::endl;
#pragma omp parallel num_threads(ndev)
  {
    for (uint32_t j = 0; j < stepnum; ++j)
    {
#pragma omp for
      for (int i = 0; i < ndev; ++i)
      {
        particles.setGPU(i);
        particles[i].calcBlockID();

        particles[i].calcForce();
      }

      // merge forces in between each GPUs
      if (ndev > 1)
      {
        particles.exchangeForce();
      }

#pragma omp for
      for (int i = 0; i < ndev; ++i)
      {
        particles.setGPU(i);

        //	real t = particles[i].constTemp();
        //	std::cerr << "constTemp T= " << t << std::endl;
        real t = particles[i].calcTemp();
        if (i == 0)
          std::cerr << "T= " << t << std::endl;
        if (t > Temp * 1.2)
        {
#if defined(MVSTAT)
          particles[i].statMV2(mvstat);
#endif
          std::cerr << "abort" << std::endl;
          exit(1);
        }
        if (t > Temp * 1.1)
          std::cerr << "scaling temp by: " << particles[i].scaleTemp(Temp)
                    << std::endl;
        particles[i].TimeEvolution(delta_t);
        particles[i].treatPeriodicCondition();
      }

#pragma omp master
      if ((j + 1) % ointerval == 0)
      {
        particles[0].timestep = j + 1 + initstep;
        particles[0].getPosition();
        particles[0].putTMP(std::cout);
#if defined(MVSTAT)
        particles[0].statMV2(mvstat);
#endif
      }
    }
  }

#if defined(MVSTAT)
  mvstat.close();
#endif

  std::ofstream ofs("MD3done");
  boost::archive::binary_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("cudaParticles", particles[0]);
  ofs.close();

  return 0;
}
