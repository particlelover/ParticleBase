#include <iostream>
#include "CUDAenv.hh"
#include "GaussianThermo.hh"
#include "kernelfuncs.h"
#include <random>

//#define MVSTAT

typedef std::vector<class ParticleBase> GlobalTable;

void createInitialState(CUDAenv<GaussianThermo> &particles) {
  /*
   * Super Critical CO2
   * sigma/epsilon
   * 3.941/195.2; 3.912/225.3; 3.720/236.1; 3.262/500.7
   *
   * calculation condition
   * T=311K, rho*=1.2 (rho=0.00748)
   * R.Ishii, J. Chem. Phys., 105, 22 (1996)
   *
   * units: \AA, fs, 1/Na g
   */
  std::mt19937 engine;

  const real kB = 8.3145e-7;  // kB in \AA, fs, g/NA unit
  real cell[6] = {0.0, 300.0, 0.0, 300.0, 0.0, 120.0};
  const double rho = 0.00748;
  uint32_t N = static_cast<uint32_t>(((cell[1]-cell[0])*(cell[3]-cell[2])*(cell[5]-cell[4])) * rho);
  const real Temp = 311;


  uint32_t n1, n2, n3;
  double l1, l2, l3;
  {
    const double vol = (cell[1]-cell[0])*(cell[3]-cell[2])*(cell[5]-cell[4]);
    const double l0 = pow(vol / N, 1.0/3.0);
    n1 = static_cast<uint32_t>((cell[1]-cell[0])/l0)+1;
    n2 = static_cast<uint32_t>((cell[3]-cell[2])/l0)+1;
    n3 = static_cast<uint32_t>((cell[5]-cell[4])/l0)+1;
    //std::cerr << n1 << "," << n2 << "," << n3 << std::endl;
    assert(n1*n2*n3>N);
    l1 = (cell[1]-cell[0])/n1;
    l2 = (cell[3]-cell[2])/n2;
    l3 = (cell[5]-cell[4])/n3;
  }


  GlobalTable G1;
  G1.resize(N);
  for (int i=0;i<N;++i) {
    ParticleBase pb;

    real m1, m2, m3;
    m1 = l1*(i%n1+0.5);
    m2 = (((i-(i%n1))/n1)%n2+0.5)*l2;
    m3 = ((uint32_t)(i/(n1*n2))+0.5)*l3;
    pb.r[0] = m1;
    pb.r[1] = m2;
    pb.r[2] = m3;
/*
    pb.r[0] = RR.rnd(cell[1]-cell[0]);
    pb.r[1] = RR.rnd(cell[3]-cell[2]);
    pb.r[2] = RR.rnd(cell[5]-cell[4]);
*/
    // CO2
    pb.m = 44.01;
    pb.type = 0;
    std::normal_distribution<> t1(0.0, sqrt(kB * Temp / pb.m));
    pb.v[0] = t1(engine);
    pb.v[1] = t1(engine);
    pb.v[2] = t1(engine);
    pb.a[0] = 0.0; pb.a[1] = 0.0; pb.a[2] = 0.0;
    G1[i] = pb;
  }


  const int ndev = particles.nDevices();
  particles.setup();
#pragma omp parallel for
  for (int i=0;i<ndev;++i) {
    particles.setGPU(i);
    particles[i].setup(N);
    particles[i].setCell(cell);

    particles[i].import(G1);
    particles[i].setM();

    particles[i].setE(1e-12);
    particles[i].kB = kB;
    particles[i].timestep = 0;
    particles[i].rmax2 = 19.6*19.6;
    particles[i].useNList();
    /* from the Maxwell-Boltzmann distribution, \f$ <v> = \sqrt {\frac{8k_B T}{\pi m} }\f$
     * v_ave = Math.sqrt((8 * 8.3145e-7 * 311) / (3.14 * 44.01)) = 0.0038690358024039089145
     2 * v_ave * 50steps = 0.0038690358024039089145 * 100 = 0.38690358024039089145
     */
    particles[i].thickness = 0.4;
  }
}


int main(int argc, char **argv) {
  class CUDAenv<GaussianThermo> particles;
  const int ndev = particles.nDevices();

  if (argc==2) {
    std::cerr << "reading serialization file " << argv[1] << std::endl;
    particles.setup();

    particles.readSerialization(argv[1]);
  } else {
    createInitialState(particles);
  }

#pragma omp parallel for
  for (int i=0;i<ndev;++i) {
    particles.setGPU(i);
    std::vector<real> LJpara;
    const int elemnum = 1;
    LJpara.resize(2*elemnum*elemnum);
    LJpara[0] = 3.941; LJpara[1] = 195.2*particles[i].kB;
    particles[i].setLJparams(LJpara, elemnum);
  }
  const real Temp = 311;

#if defined(MVSTAT)
  std::ofstream mvstat;
  mvstat.open("mv2stat4");
#endif

  if (argc==1) {
    // if use with cutoff block
#pragma omp parallel for
    for (int i=0;i<ndev;++i) {
      particles.setGPU(i);
      particles[i].setupCutoffBlock(20.0);
      particles[i].calcBlockID();
#if defined(MVSTAT)
      particles[i].statMV2(mvstat);
#endif

      particles[i].adjustVelocities(Temp);
    }
  }


  for (int i=0;i<ndev;++i) {
    particles[i].setBlockRange(particles[i].numBlocks(), ndev, i);
  }

  particles[0].getTypeID();
  particles[0].getPosition();
  particles[0].putTMP(std::cout);
#if defined(MVSTAT)
  particles[0].statMV2(mvstat);
#endif


  const real delta_t = 1.0;
  const uint32_t stepnum =
    static_cast<uint32_t>(80000 / delta_t); //80.0ps
  const uint32_t ointerval =
    static_cast<uint32_t>(100 / delta_t); // 100fs
  const uint32_t initstep = particles[0].timestep;

  const uint32_t neighborListInterval = 100;

  // MAIN LOOP
  std::cerr << "start main loop" << std::endl;
#pragma omp parallel num_threads(ndev)
{
  for (uint32_t j=0;j<stepnum;++j) {
#pragma omp for
    for (int i=0;i<ndev;++i) {
      particles.setGPU(i);
      if (j%neighborListInterval==0) {
        particles[i].calcBlockID();
        particles[i].makeNList();
      }

      particles[i].calcForce();
    }


    // merge forces in between each GPUs
    if (ndev>1) {
      particles.exchangeForce();
    }

#pragma omp for
    for (int i=0;i<ndev;++i) {
      particles.setGPU(i);

      //real t = particles[i].constTemp();
      //std::cerr << "constTemp T= " << t << std::endl;
      real t = particles[i].calcTemp();
      if (i==0)
        std::cerr << "T= " << t << std::endl;
      if ((t>Temp*1.2) || isnan(t)) {
#if defined(MVSTAT)
        particles[i].statMV2(mvstat);
#endif
        std::cerr << "abort" << std::endl;
        exit(1);
      }
      if (t>Temp*1.1)
        std::cerr << "scaling temp by: " << particles[i].scaleTemp(Temp)
                  << std::endl;
      particles[i].TimeEvolution(delta_t);
      particles[i].treatPeriodicCondition();
    }

    // calculate/put energy
#pragma omp master
    if ((j+1)%25==0) {
      std::cerr << "K-P\t" << 
      particles[0].calcKineticE() << "\t" << particles[0].calcPotentialE() << std::endl;
    }

#pragma omp master
    if ((j+1)%ointerval==0) {
      particles[0].timestep = j+1+initstep;
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

  particles.writeSerialization("MD4done");

  return 0;
}
