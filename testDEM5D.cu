#include <iostream>
#include "CUDAenv.hh"
#include "cudaParticleDEM.hh"
#include "AdaptiveTime.hh"
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <math.h>
#include <random>

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
  const real R0 = 0.50;
  const real lunit = 2 * R0;
  const real L1 = 65.0;
  const real L2 = 65.0;
  const real L3 = 10.0;
  const real L4 = 60.0;
  real cell[6] = {0.0, L1, 0.0, L2, 0.0, L4};

  real WeighFe = 7.874 * 4.0 / 3.0 * M_PI * R0 * R0 * R0;

  // border
  std::normal_distribution<> s1(0.0, R0 / 100.0);
  const int lsize1 = L1 / (lunit) + 1;
  const int lsize2 = L2 / (lunit) + 1;
  const int lsize3 = L3 / (lunit) + 1;

  for (int k=0;k<lsize3;++k)
    for (int i=0;i<lsize1;++i)
      for (int j=0;j<lsize2;++j) {
        if (((i<1) || (lsize1-2<i) || (j<1) || (lsize2-2<j)) ||
            (k==0)) {
          ParticleBase pb;
          if ((0<i) && (i<lsize1-1) && (0<j) && (j<lsize2-1)) {
            pb.r[0] = i * lunit + s1(engine);
            pb.r[1] = j * lunit + s1(engine);
          } else {
            pb.r[0] = i * lunit;
            pb.r[1] = j * lunit;
          }
          pb.r[2] = k * lunit;
          pb.m = WeighFe;
          pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
          pb.a[0] = pb.a[1] = pb.a[2] = 0.0;
          pb.isFixed = true;
          pb.type = 0;
          G1.push_back(pb);
        }
      }
  uint32_t N1 = G1.size();
  std::cerr << "N= " << N1 << std::endl;

  // moving
  std::normal_distribution<> s2(0.0, R0 / 25.0);
  const int P1 = 25 / lunit / 1.3;
  const int P2 = L4 / lunit - 2;
  uint32_t _l = 0;

  for (int k=0;k<P2;k+=1)
    for (int i=0;i<P1;i+=1)
      for (int j=0;j<P1;j+=1) {
        ParticleBase pb;
        pb.r[0] = (i*1.3) * lunit + s2(engine) + 20.0;
        pb.r[1] = (j*1.3) * lunit + s2(engine) + 20.0;
        pb.r[2] = (k+1  ) * lunit + lunit / 3;
        pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
        pb.a[0] = pb.a[1] = pb.a[2] = 0.0;
        pb.isFixed = false;
        pb.type = (_l++ % 3) == 0 ? 1 : 2;
        pb.m = WeighFe * (pb.type==1) ? 1.0 : (0.8*0.8*0.8); // (0.4/0.5)^3
        G1.push_back(pb);
      }
  uint32_t N = G1.size();
  std::cerr << "N= " << N << std::endl;

  const int ndev = particles.nDevices();
  particles.setup();
#pragma omp parallel for
  for (int i=0;i<ndev;++i) {
    particles.setGPU(i);

    particles[i].switchBlockAlgorithm(ParticleBlockType::many);
    particles[i].setup(N);
    particles[i].setCell(cell);

    particles[i].import(G1);
    particles[i].autotunetimestep = true;
    //particles.setDEMProperties(1.0e11, 0.5, 0.3, 0.9, 0.2,
    //particles.setDEMProperties(1.0e5, 0.10, 0.45, 0.9, 0.05,
    // Fe: Young Modulus 211GPa, Poisson ratio 0.29, density 7.874gcm-3
    // 211GPa = 2.11x10^12 g cm^-1 sec^-2

    std::vector<real> rad(N);
    for (int j=0;j<N1;++j) {
      rad[j] = R0;
    }
    for (int j=N1;j<N;++j) {
      rad[j] = R0 * (((j-N1)%3==0) ? 1.0 : 0.8);
    }
    const real e = 0.85;
    const real gamma = -log(e) / sqrt(M_PI*M_PI + log(e)*log(e)) * 2;
    std::cerr << "2 gamma:" << gamma << std::endl;
    particles[i].setDEMProperties(2.11e10, 0.40, 0.29, gamma, 0.10, rad);
    particles[i].setInertia(rad);
    particles[i].setupCutoffBlock(R0*2, false);

    // putTMPselected
    particles[i].setupSelectedTMP(N1, N-N1, 0, N1);

    particles[i].checkPidRange(i, ndev, N1, N);
  }
  particles[0].timestep = 0;
  particles[0].putUnSelected("dump.DEM5box");
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
  const real intaval = 0.005;
  const uint32_t initstep = particles[0].timestep;
  const real initDeltaT = 0.000008;
  const real ulim = 0.01 * 4;
  const real llim = ulim / 16.0;
  const real R0 = 0.40;
  std::vector<int> res(ndev);

  const real param_g = 9.8e2;

#pragma omp parallel num_threads(ndev)
  {
    AdaptiveTime<CUDAenv<cudaParticleDEM>> thistime;
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
        particles.exchangeForceSelected(ExchangeMode::force); // exchange forces
        particles.exchangeForceSelected(ExchangeMode::torque); // exchange torques
      }

#pragma omp for
      for (int i=0;i<ndev;++i) {
        particles[i].calcAcceleration();
        particles[i].addAccelerationZ(-param_g);
        particles[i].TimeEvolution(thistime.currentDeltaT());

        real _r = 0.0;
        uint32_t _r1 = 0;
        res[i] = particles[i].inspectVelocity((2*R0) / thistime.currentDeltaT(), ulim, llim, _r, _r1);
        //std::cerr << "TimeMesh: " << thistime() << " " << thistime.currentDeltaT() << "\t" << _r << "\t" << _r1 << std::endl;
      }

      // progress simulation time
      j = thistime.Progress(particles, res, j);

#pragma omp for
      for (int i=0;i<ndev;++i) {
        particles[i].treatRefrectCondition();
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

  particles.writeSerialization("DEM5done");

  return 0;
}
