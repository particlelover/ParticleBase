#include <iostream>
#include "CUDAenv.hh"
#include "cudaParticleDEM.hh"
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>
#include <math.h>
#include <random>

typedef std::vector<class ParticleBase> GlobalTable;

void createInitialState(CUDAenv<cudaParticleDEM> &particles)
{
  /*
   * units of this simulation is:
   * length: cm
   * mass: g
   * time: s
   */

  std::mt19937 engine;

  const real R0 = 0.40;
  GlobalTable G1;
  const real L1 = 125.0;
  const real L2 = 40.0;
  const real L3 = L2 * 0.6;
  const real lunit = 2 * R0;
  real cell[6] = {0.0, L1, 0.0, L2, 0.0, L3 * 2.5 + 6.3 + R0};

  real WeighFe = 7.874 * 4.0 / 3.0 * M_PI * R0 * R0 * R0;

  /*
  for (int i=2;i<lsize-2;i+=1)
    for (int j=2;j<lsize-2;j+=1)
      for (int k=2+4;k<lsize-2+4;k+=1) {
*/
  std::normal_distribution<> s1(0.0, R0 / 10.0);
  const int P1 = 24 / lunit / 1.3;
  const int P2 = L3 * 2.5 / lunit;
  for (int k = 0; k < P2; k += 1)
    for (int i = 0; i < P1; i += 1)
      for (int j = 0; j < P1; j += 1)
      {
        ParticleBase pb;
        pb.r[0] = (i * 1.3) * lunit + s1(engine) + 35.0;
        pb.r[1] = (j * 1.3) * lunit + s1(engine) + (L2 - 24) / 2;
        pb.r[2] = (k)*lunit + 6.3 + R0;
        pb.m = WeighFe;
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
  const int lsize = L1 / (lunit);
  const int lsize2 = L2 / (lunit);
  const int lsize3 = L3 / (lunit);
  std::normal_distribution<> s2(0.0, R0 / 100.0);
  for (int k = 0; k < lsize3; ++k)
    for (int i = 0; i < lsize; ++i)
      for (int j = 0; j < lsize2; ++j)
      {
        if ((((i < 1) || (lsize - 2 < i) ||
              (j < 1) || (lsize2 - 2 < j)) &&
             (k * lunit > 3 * log(1.8 - 1.6 * (real)(i) / (real)(lsize)) + 5)) ||
            ((k == 0) && ((0 < i) && (i < lsize - 1) && (0 < j) && (j < lsize2 - 1))))
        {
          ParticleBase pb;
          pb.r[0] = i * lunit + s2(engine);
          pb.r[1] = j * lunit + s2(engine);
          if (k == 0)
            pb.r[2] = (3 * log(1.8 - 1.6 * (real)(i) / (real)(lsize)) + 5);
          else
            pb.r[2] = k * lunit;
          pb.m = WeighFe;
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

  const int ndev = particles.nDevices();
  particles.setup();
#pragma omp parallel for
  for (int i = 0; i < ndev; ++i)
  {
    particles.setGPU(i);

    particles[i].setup(N);
    particles[i].setCell(cell);

    particles[i].import(G1);
    //particles.setDEMProperties(1.0e11, 0.5, 0.3, 0.9, 0.2,
    //particles.setDEMProperties(1.0e5, 0.10, 0.45, 0.9, 0.05,
    // Fe: Young Modulus 211GPa, Poisson ratio 0.29, density 7.874gcm-3
    // 211GPa = 2.11x10^12 g cm^-1 sec^-2
    const real e = 0.85;
    const real gamma = -log(e) / sqrt(M_PI * M_PI + log(e) * log(e)) * 2;
    std::cerr << "2 gamma:" << gamma << std::endl;
    particles[i].setDEMProperties(2.11e07, 0.40, 0.29, gamma, 0.10,
                                  R0);
    particles[i].setInertia(R0);
    particles[i].setupCutoffBlock(R0 * 2, false);

    // putTMPselected
    particles[i].setupSelectedTMP(0, N1, N1, N - N1);

    particles[i].checkPidRange(i, ndev, 0, N1);
  }
  particles[0].timestep = 0;
  particles[0].putUnSelected("DEM4box.dump");
}

int main(int argc, char **argv)
{
  class CUDAenv<cudaParticleDEM> particles;
  const int ndev = particles.nDevices();

  if (argc == 2)
  {
    std::cerr << "reading serialization file " << argv[1] << std::endl;
    particles.setup();

    particles.readSerialization(argv[1]);
  }
  else
  {
    createInitialState(particles);

    for (int i = 0; i < ndev; ++i)
    {
      particles.setGPU(i);
      particles[i].calcBlockID();
    }
  }

  particles[0].getSelectedTypeID();
  particles[0].getSelectedPosition();
  particles[0].putTMP(std::cout);

  const real stepmax = 1.50;
  const real intaval = 0.005;
  const uint32_t initstep = particles[0].timestep;

  const real R0 = 0.40;
  std::vector<int> res(ndev);

#pragma omp parallel num_threads(ndev)
  {
    real thistime = 0.0;
    real nextoutput = intaval;
    real deltaT = 0.000008;
    uint32_t wait_change_deltaT = 0;

#pragma omp master
    {
      std::cerr << "DeltaT(0)"
                << ":" << thistime << ":" << 0 << ":" << deltaT << std::endl;
      std::cerr << "End Time: " << stepmax << std::endl;
    }

#pragma omp for
    for (int i = 0; i < ndev; ++i)
    {
      particles.setGPU(i);
      particles[i].selectBlocks();
      particles[i].calcVinit(deltaT);
    }

    uint32_t j = 0;
    bool rollback_in_progress = false;
    while (thistime < stepmax)
    {
      if (rollback_in_progress)
      {
        std::cerr << "now rollbacking:" << std::flush;
      }
#pragma omp master
      {
        if (j % 50 == 0)
          std::cerr << j << " ";
      }
#pragma omp for
      for (int i = 0; i < ndev; ++i)
      {
        res[i] = 0;
        particles.setGPU(i);
        particles[i].calcBlockID();

        particles[i].selectBlocks();
        particles[i].calcForce(deltaT);
      }

      if (ndev > 1)
      {
        particles.exchangeForceSelected(0); // exchange forces
        particles.exchangeForceSelected(2); // exchange torques
      }

#pragma omp for
      for (int i = 0; i < ndev; ++i)
      {
        particles[i].calcAcceleration();
        particles[i].addAccelerationZ(-9.8e2);
        particles[i].TimeEvolution(deltaT);

        real _r = 0.0;
        //res[i] = particles[i].inspectVelocity((2*R0)/deltaT, 0.01, 0.01/16, _r, true);
        //std::cerr << "TimeMesh: " << thistime << " " << deltaT << "\t" << _r << std::endl;
        res[i] = particles[i].inspectVelocity((2 * R0) / deltaT, 0.01, 0.01 / 16, _r, false);
      }

      if (!rollback_in_progress)
      {
        thistime += deltaT;
        ++j;

        int resmax = *std::max_element(res.begin(), res.end());

        if (resmax == 1)
        {
          /* DO rollback process!
       * position r at the time t is not safe to calculate the force!
       *
       * r(t+dt) => r(t) (by v(t+dt/2)
       * v(t+dt/2) => v(t-dt/2) (by a(t))
       * r(t) => r(t-dt) (by v(t-dt/2)
       * calcA (by r(t-dt)
       * v(t-dt/2) => v(t-dt-???)
       * ???: dt/2 with new time step dt
       */
#pragma omp for
          for (int i = 0; i < ndev; ++i)
          {
            particles[i].rollback(deltaT);
          }
          rollback_in_progress = true;
        }
        else if ((resmax == -1) && (wait_change_deltaT == 0))
        {
          // v(t+delta_t) => v(t)
          static const real growthratio = sqrt(2.0);
#pragma omp for
          for (int i = 0; i < ndev; ++i)
          {
            particles[i].calcVinit((3 - growthratio) * deltaT);
          }
          deltaT *= growthratio;
          wait_change_deltaT = 1;
          std::cerr << std::endl
                    << "DeltaT(+)"
                    << ":" << thistime << ":" << j << ":" << deltaT << std::endl;
        }
        if (wait_change_deltaT > 0)
          --wait_change_deltaT;
      }
      else
      {
        //rollback_in_progress
        std::cerr << "rollback_step2:" << std::flush;
#pragma omp for
        for (int i = 0; i < ndev; ++i)
        {
          particles[i].rollback2(deltaT);
        }
        rollback_in_progress = false;

        thistime += -deltaT;
        deltaT /= 4.0;
        wait_change_deltaT = 3;

        std::cerr << "done: " << std::flush;
        std::cerr << std::endl
                  << "DeltaT(-)"
                  << ":" << thistime << ":" << j << ":" << deltaT << std::endl;
      }

#pragma omp for
      for (int i = 0; i < ndev; ++i)
      {
        particles[i].treatAbsoluteCondition();
      }

#pragma omp master
      if (thistime >= nextoutput)
      {
        std::cerr << std::endl
                  << "(" << thistime << ") ";
        nextoutput += intaval;
        particles[0].timestep = j + 1 + initstep;
        particles[0].getSelectedPosition();
        particles[0].putTMP(std::cout);
      }
    }
    std::cerr << "DeltaT(0)"
              << ":" << thistime << ":" << j << ":" << deltaT << std::endl;
  }

  particles.writeSerialization("DEM4done");

  return 0;
}
