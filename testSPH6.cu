#include <iostream>
#include "CUDAenv.hh"
#include "cudaParticleSPH_NS.hh"
#include <fstream>
#include <random>

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

  std::mt19937 engine;

  GlobalTable G1;
  const double lunit = 0.4;
  const double sph_h = lunit / 0.5938629;
  real cell[6] = {0, 20, 0, 20, 0, 25};

  const double rho_0 = 20.0 / (4.0/3.0*M_PI*(sph_h)*(sph_h)*(sph_h));
  const double m_0 = 1.0 / rho_0;
  // 0.9799 for 20m/(4/3 pi1.7^3)=1.0g/cm^3

  std::cerr << "SPH kernel h = " << (sph_h)
            << " lunit = " << lunit
            << " mean number density = " << rho_0
            << std::endl;


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
  for (int i=0;i<L3;++i)
    for (int j=0;j<L2;++j)
      for (int k=0;k<L1;++k) {
        // i: Z, j: Y, k: X
        if (
          ((i==0) && (j<L2/2) && (k<L1/2)) ||
          ((i==H5) && (j<L2/2) && (k>=L1/2)) ||
          ((i==H10 && (j>=L2/2) && (k>=L1/2))) ||
          ((i==H15 && (j>=L2/2) && (k<L1/2))) || // floor
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
          for (int l=0;l<2;++l) { // two border particles set on same position
            ParticleBase pb;
            pb.r[0] = k*lunit+lunit_h;
            pb.r[1] = j*lunit+lunit_h;
            pb.r[2] = i*lunit+lunit_h;
            pb.m = m_0*2.5; // x2.5 by water
            pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
            pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
            pb.isFixed = true;
            pb.type = 0;
            G1.push_back(pb);
          }
        }
      }
  uint32_t N1=G1.size();
  std::cerr << "N= " << N1 << std::endl;

  /*
   * fluid particles
   */
  // wall position Y:L2/2, L2; X:0, L1/2
  std::normal_distribution<> s1(0.0, lunit_h / 50.0);
   for (int i=H15+2;i<L3;++i)
    for (int j=L2/2+2;j<L2-1;++j)
      for (int k=2;k<L1/2-1;++k) {
        // i: Z, j: Y, k: X
        ParticleBase pb;
        pb.r[0] = k*lunit + s1(engine);
        pb.r[1] = j*lunit + s1(engine);
        pb.r[2] = i*lunit + s1(engine);
        pb.m = m_0; // 21m/(4/3 pi0.5^3)=1.0g/cm^3
        pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
        pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
        pb.isFixed = false;
        pb.type = 1;
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

    // putTMPselected
    particles[i].setupSelectedTMP(N1, N-N1, 0, N1);
    std::cerr << "particles, moving= " << N-N1 << " total= " << N << std::endl;
  }
  std::cerr << "setup done" << std::endl;

  particles[0].putUnSelected("dump.SPH6box");
}


int main(int argc, char **argv) {
  class CUDAenv<cudaParticleSPH_NS> particles;
  const int ndev = particles.nDevices();

  if (argc==2) {
    std::cerr << "reading serialization file " << argv[1] << std::endl;
    particles.setup();

    particles.readSerialization(argv[1]);
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
  const uint32_t stepmax  = 1.50 / deltaT;
  const uint32_t intaval  = 0.005 / deltaT;
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
      if (j%50==0) std::cerr << j << " ";
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

  particles.writeSerialization("SPH6done");

  return 0;
}
