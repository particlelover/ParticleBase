#include <iostream>
#include "CUDAenv.hh"
#include "cudaParticleDEM.hh"
#include "AdaptiveTime.hh"
#include <fstream>
#include <random>

#if defined(USEMULTIPARTICLEBLOCK)
# define PARTICLEBLOCK  ParticleBlockType::many
# define MESHSIZE (R0*2)
#else
# define PARTICLEBLOCK  ParticleBlockType::single
# define MESHSIZE (R0*2/sqrt(3.0))
#endif

typedef std::vector<class ParticleBase> GlobalTable;

void createInitialState(CUDAenv<cudaParticleDEM> &particles) {
  /*
   * units of this simulation is:
   * length: cm
   * mass: g
   * time: s
   */

  std::mt19937 engine;

  GlobalTable G1;
  const real R0=0.20;
  const real lunit = 2 * R0;
  real cell[6] = {0, 20, 0, 20, 0, 25};

  real WeighFe = 7.874 * 4.0 / 3.0 * M_PI * R0*R0*R0;

  // from SPH5_2
  const signed int L1 = 20 / lunit - 1;
  const signed int L2 = 20 / lunit - 1;
  const signed int L3 = 25 / lunit - 1;
  const double lunit_h = lunit / 2;

  const signed int H5 = 5 / lunit - 1;
  const signed int H10 = 10 / lunit - 1;
  const signed int H15 = 15 / lunit - 1;

  /*
   * border: 20x20x25 cm box
   */
  std::normal_distribution<> s1(0.0, R0 / 10.0);
  for (int i=0;i<L3;++i)
    for (int j=0;j<L2;++j)
      for (int k=0;k<L1;++k) {
        // i: Z, j: Y, k: X
        if (
          ((i==0) && (j<L2/2) && (k<L1/2)) ||
          ((i==H5) && (j<L2/2) && (k>=L1/2)) ||
          ((i==H10 && (j>=L2/2) && (k>=L1/2))) ||
          ((i==H15 && (j>=L2/2) && (k<L1/2+1))) || // floor
          ((k==0) && (j<L2/2)) ||
          ((k==0) && (j>=L2/2) && (i>=H15)) ||
          ((k==L1-1) && (j<L2/2) && (i>=H5)) ||
          ((k==L1-1) && (j>=L2/2) && (i>=H10)) ||
          ((j==0) && (k<L1/2)) ||
          ((j==0) && (k>=L1/2) && (i>=H5)) ||
          ((j==L2-1) && (k<L1/2) && (i>=H15)) ||
          ((j==L2-1) && (k>=L1/2) && (i>=H10)) || // wall
          ((j==L2/2) && (k<L1/2)) ||
          ((j==L2/2) && (k>=L1/2) && (i>=H5) && (i<H10)) ||
          ((k==L1/2) && (j<L2/2) && (i<H5)) ||
          ((k==L1/2) && (j>=L2/2) && (i>=H10) && (i<H15))
        ) {
          ParticleBase pb;
          pb.r[0] = k*lunit+lunit_h + s1(engine);
          pb.r[1] = j*lunit+lunit_h + s1(engine);
          pb.r[2] = i*lunit+lunit_h + s1(engine);
          pb.m = WeighFe;
          pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
          pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
          pb.isFixed = true;
          pb.type = 0;
          G1.push_back(pb);
        }
      }
  uint32_t N1=G1.size();
  std::cerr << "N= " << N1 << std::endl;

  /*
   * fluid particles
   */
  // wall position Y:L2/2, L2; X:0, L1/2
  for (int i=H15+2;i<L3;++i)
    for (int j=L2/2+2;j<L2-1;++j)
      for (int k=2;k<L1/2-1;++k) {
        // i: Z, j: Y, k: X
        ParticleBase pb;
        pb.r[0] = k * lunit;
        pb.r[1] = j * lunit;
        pb.r[2] = i * lunit;
        pb.m = WeighFe;
        pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
        pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
        pb.isFixed = false;
        pb.type = 1;
        G1.push_back(pb);
      }
  uint32_t N = G1.size();
  std::cerr << "N= " << N << std::endl;

  const int ndev = particles.nDevices();
  particles.setup();
#pragma omp parallel for
  for (int i=0;i<ndev;++i) {
    particles.setGPU(i);
    particles[i].switchBlockAlgorithm(PARTICLEBLOCK);
    particles[i].setup(N);
    particles[i].setCell(cell);
    particles[i].import(G1);
    particles[i].timestep = 0;
    particles[i].autotunetimestep = true;
    //particles.setDEMProperties(1.0e11, 0.5, 0.3, 0.9, 0.2,
    //particles.setDEMProperties(1.0e5, 0.10, 0.45, 0.9, 0.05,
    // Fe: Young Modulus 211GPa, Poisson ratio 0.29, density 7.874gcm-3
    // 211GPa = 2.11x10^12 g cm^-1 sec^-2
    const double e =0.85;
    const double gamma = - log(e) / sqrt(M_PI*M_PI + log(e)*log(e)) * 2;
    std::cerr << "2 gamma:" << gamma << std::endl;
    particles[i].setDEMProperties(2.11e09, 0.40, 0.29, gamma, 0.10, R0);
    particles[i].setInertia(R0);
    particles[i].setupCutoffBlock(MESHSIZE * 0.9, false);

    // putTMPselected
    particles[i].setupSelectedTMP(N1, N-N1, 0, N1);
    std::cerr << "particles, moving= " << N-N1 << " total= " << N << std::endl;

    particles[i].checkPidRange(i, ndev, N1, N);
  }
  std::cerr << "setup done" << std::endl;

  particles[0].putUnSelected("dump.DEM6box");
}

int main(int argc, char **argv) {
  class CUDAenv<cudaParticleDEM> particles;
  const int ndev = particles.nDevices();

  if (argc==2) {
    std::cerr << "reading serialization file " << argv[1] << std::endl;
    particles.setup();

    particles.readSerialization(argv[1]);
  } else {
    createInitialState(particles);

    for (int i=0;i<ndev;++i) {
      particles.setGPU(i);
      particles[i].calcBlockID();
    }
  }


  particles[0].getSelectedTypeID();
  particles[0].getSelectedPosition();
  particles[0].putTMP(std::cout);
  particles[0].waitPutTMP();

  const real stepmax = 1.50;
  const real intaval  = 0.005;
  const uint32_t initstep = particles[0].timestep;
  const real initDeltaT = 0.000008;
  const real ulim = 0.01 * 4;
  const real llim = ulim / 16.0;
  const real R0 = 0.20;
  std::vector<int> res(ndev);

  const real param_g = 9.8e2;

#pragma omp parallel num_threads(ndev)
  {
    AdaptiveTime<CUDAenv<cudaParticleDEM> > thistime;
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
      particles[i].selectBlocks();
      particles[i].calcVinit(initDeltaT);
    }

    uint32_t j = 0;

    while (thistime() < stepmax) {
#pragma omp master
      {
        if (thistime.isRollbacking()) {
          std::cerr << "now rollbacking:" << std::flush;
        }
        if (j%50==0) std::cerr << j << " ";
      }
#pragma omp for
      for (int i=0;i<ndev;++i) {
        res[i] = 0;
        particles.setGPU(i);
        particles[i].calcBlockID();

        particles[i].selectBlocks();
        particles[i].calcForce(thistime.currentDeltaT());
      }

      if (ndev>1) {
        particles.exchangeForceSelected(ExchangeMode::force);   // exchange forces
        particles.exchangeForceSelected(ExchangeMode::torque);   // exchange torques
      }

#pragma omp for
      for (int i=0;i<ndev;++i) {
        particles[i].calcAcceleration();
        particles[i].addAccelerationZ(-param_g);
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
        particles[i].treatRefrectCondition();
      }

#pragma omp master
      if (thistime() >= nextoutput) {
        std::cerr << std::endl << "(" << thistime() << ") ";
        nextoutput += intaval;
        particles[0].timestep = j+1+initstep;
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

  particles.writeSerialization("DEM6done");

  return 0;
}
