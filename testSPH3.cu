#include <iostream>
#include "CUDAenv.hh"
#include "cudaParticleSPH_NS.hh"


#define SPH_H 0.4

typedef std::vector<class ParticleBase> GlobalTable;


void createInitialState(CUDAenv<cudaParticleSPH_NS> &particles) {
  const char *XXX="XXX";

  /** read LAMMPS dump file for the ellipsoids
   *  made by ELpack with atom size 0.6
   *
   * units of this simulation is:
   * length: cm
   * mass: g
   * time: s
   */

  /* 40x40x40 box for inner
   * border thickness SPH_H*4 = 4.8
   */
  real cell[6] = {-(SPH_H)*5, 20.0+(SPH_H)*5, -(SPH_H)*5, 20.0+(SPH_H)*5, -(SPH_H)*5, 30.0+(SPH_H)*5};
  const double lunit = (SPH_H)*0.5938629;

  const double rho_0 = 20.0 / (4.0/3.0*M_PI*(SPH_H)*(SPH_H)*(SPH_H));
  const double m_0 = 1.0 / rho_0;
    // 0.9799 for 20m/(4/3 pi1.7^3)=1.0g/cm^3

  std::cerr << "SPH kernel h = " << (SPH_H)
            << " lunit = " << lunit
            << " mean number density = " << rho_0
            << std::endl;


  GlobalTable G1;
  {
    std::ifstream ifil;
    ifil.open(XXX);
    if (!ifil.is_open()) {
      std::cerr << "file 'XXX' not exist" << std::endl;
      exit(0);
    }

    std::string D;
    getline(ifil, D);
    getline(ifil, D);
    getline(ifil, D);
    uint32_t N;
    ifil >> N;
    getline(ifil, D);
    G1.resize(N*2);

    getline(ifil, D);
    getline(ifil, D);
    getline(ifil, D);
    getline(ifil, D);
    getline(ifil, D);

    for (uint32_t i=0;i<N;++i) {
      uint32_t p1, p2;
      double r1, r2, r3;
      ifil >> p1 >> p2 >> r1 >> r2 >> r3;
      ParticleBase pb;
      pb.r[0] = r1;
      pb.r[1] = r2;
      pb.r[2] = r3 +(20.0-17.45);
      pb.m = m_0*2.5;   // x2.5 by water
      pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
      pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
      pb.isFixed = true;
      pb.type = 2;
      G1[(p1-1)*2]   = pb;
      G1[(p1-1)*2+1] = pb;
    }

    std::cerr << "read " << N << " atoms from " << XXX << std::endl;

    ifil.close();
  }

  const double lunit_h = lunit / 2;
  const signed int L = 20 / lunit;
  const signed int L2 = 10 / lunit;
  // border
  // cell size 40x40x40;
  for (int i=-2;i<L+2;++i)
    for (int j=-2;j<L+2;++j)
      for (int k=-2;k<L+L2;++k) {
        if ((i<0)||(L+2-3<i) ||
            (j<0)||(L+2-3<j) ||
            (k<0)) {
          ParticleBase pb;
          pb.r[0] = i*lunit+lunit_h;
          pb.r[1] = j*lunit+lunit_h;
          pb.r[2] = k*lunit+lunit_h;
          pb.m = m_0*2.5;   // x2.5 by water
          pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
          pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
          pb.isFixed = true;
          pb.type = 1;
          G1.push_back(pb);
        }
      }
  uint32_t N1=G1.size();
  std::cerr << "N= " << N1 << std::endl;


  for (int i=1;i<L-1;++i)
    for (int j=1;j<L-1;++j)
      for (int k=L+5;k<L+L2;++k) {
        ParticleBase pb;
        pb.r[0] = i*lunit+lunit_h;
        pb.r[1] = j*lunit+lunit_h;
        pb.r[2] = k*lunit+lunit_h;
        pb.m = m_0; // 21m/(4/3 pi0.5^3)=1.0g/cm^3
        pb.v[0] = pb.v[1] = pb.v[2] = 0.0;
        pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
        pb.isFixed = false;
        pb.type = 0;
        G1.push_back(pb);
      }
  uint32_t N = G1.size();
  std::cerr << "N= " << N << std::endl;


  std::valarray<real> mu(1.0e5, N);
  mu[std::slice(N1, N-N1, 1)] = 8.9e-3;
  std::valarray<real> c1(5.44e3/2, N);    //1500m/s = 1.5e5 cm/s for water
  c1[std::slice(N1, N-N1, 1)] = 1.5e3/2;  // 5440m/s for glass


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
    particles[i].setupSelectedTMP(N1, N-N1, 0, N1);
  }
  std::cerr << "setup done" << std::endl;
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

    for (int i=0;i<ndev;++i) {
      // calculate densities for the first output
      particles[i].calcBlockID();
      particles[i].calcDensity();
    }
  }

  for (int i=0;i<ndev;++i) {
    particles[i].setBlockRange(particles[i].numBlocks(), ndev, i);
  }


  particles[0].putUnSelected("SPH3box.dump");
  particles[0].getSelectedTypeID();
  particles[0].getSelectedPosition();
  particles[0].putTMP(std::cout);


  const real deltaT = 0.000040;
  const uint32_t stepmax = 1.20 / deltaT;
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
      particles.setGPU(i);
      particles[i].calcBlockID();

      particles[i].calcKernels(); // do nothing
      particles[i].calcDensity(); // calc mass density field and its reciprocal 1/rho

      particles[i].selectBlocks();
      particles[i].setSelectedRange(particles[i].numSelectedBlocks(), ndev, i);
      particles[i].calcForce();   // do nothing
      particles[i].calcAcceleration();
    }


    if (ndev>1) {
      particles.exchangeAccelerations();
    }

#pragma omp for
    for (int i=0;i<ndev;++i) {
      particles.setGPU(i);
      particles[i].addAccelerationZ(-9.8e2);
      particles[i].TimeEvolution(deltaT);
      particles[i].treatAbsoluteCondition();
    }


#pragma omp master
    if ((j+1)%intaval==0) {
      particles[0].timestep = j+1+initstep;
      particles[0].getSelectedPosition();
      particles[0].putTMP(std::cout);
    }
  }
}

  particles.writeSerialization("SPH3done");

  return 0;
}
