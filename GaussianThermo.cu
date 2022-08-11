#include "GaussianThermo.hh"
#include "kernelfuncs.h"
#include "kerneltemplate.hh"

GaussianThermo::~GaussianThermo() {
  if (vd!=NULL) cudaFree(vd);
}

void GaussianThermo::setup(int n) {
  cudaParticleMD::setup(n);

  cudaMalloc((void **)&vd, sizeof(double)*N*3);
  if (withInfo) ErrorInfo("malloc vd[] on GPU");
}

void GaussianThermo::import(const std::vector<ParticleBase> &P) {
  cudaParticleMD::import(P);

  std::vector<double> V(N*3, 0.0);
#pragma omp parallel for
  for (int i=0;i<N;++i) {
    V[i]   = P[i].v[0];
    V[i+N] = P[i].v[1];
    V[i+N*2] = P[i].v[2];
  }
  cudaMemcpy(vd, &(V[0]), sizeof(double)*N*3, cudaMemcpyHostToDevice);
  if (withInfo) ErrorInfo("vd memcpy to Device");
}
void GaussianThermo::TimeEvolution(real dt) {

  // 1) calc A(t) vector
  // A(t) = v(t-dt)+(F(t)+F(t-dt))dt/2m - v(t-dt)xi(t-dt)dt/2
  calcGaussianThermoA1_F4<<<MPnum, THnum1D>>>(reinterpret_cast<double*>(tmp81N),
    dt, vd, F, Fold, minv, N, xi);

  // 2) solve v_i = A_i - dt/2 v_i xi(v) by Newton Raphson Method
  // by A_i - v_i - dt/2 v_i xi(v) = 0
  double v0;
  uint32_t _N = 0;
  double mv2inv = 1.0 / calcMV2();  // \sum mv^2 (use tmp3N)
  do {
    if (_N++>50) {
      std::cerr << "iteration not converged" << std::endl;
      std::cerr << v0 << ">" << _e << std::endl;
      exit(0);
    }

    // 2.2)
    calcGaussianThermoFoverF_F4<<<MPnum, THnum1D>>>(reinterpret_cast<double*>(tmp81N),
      dt, vd, F, m, reinterpret_cast<double*>(tmp3N), N, xi, mv2inv);
    // v[] also updated; f/f' => tmp3N

    // 2.3) // evaluate f/f'
    accumulate<<<1, threadsMax, sizeof(double)*threadsMax>>>(
      reinterpret_cast<double*>(tmp3N), N, reinterpret_cast<double*>(tmp3N));
    cudaMemcpy(&v0, reinterpret_cast<double*>(tmp3N), sizeof(double), cudaMemcpyDeviceToHost);
    //DOT(hdl, N*4, (float*)tmp3N, 1, (float*)tmp3N, 1, &v0);
    v0 = sqrt(v0/(N*3));


    // 2.4) update xi, mv2
    innerProduct_F4<<<1, threadsMax, sizeof(double)*threadsMax>>>(F, vd, N, reinterpret_cast<double*>(tmp3N));
    cudaMemcpy(&xi, tmp3N, sizeof(double), cudaMemcpyDeviceToHost);
    // F.w, v.w should be 0
    //DOT(hdl, N*4, (float*)F, 1, (float*)v, 1, &xi);
    mv2inv = 1.0 / calcMV2(); // \sum mv^2 for next (use tmp3N)
    xi *= mv2inv; // recalculate xi

    //std::cerr << "v0=" << v0 << " xi=" << xi << "\t" <<std::flush;
  } while (v0>_e);


  // 3) calc r(t+dt)
  propagateVelocityVerletGaussianThermo_F4<<<MPnum, THnum1D>>>(r, dt, vd, F, Fold, minv, N, xi);

  if (withInfo) ErrorInfo("GaussianThermo::TimeEvolution");
}

real GaussianThermo::calcMV2(void) {
  calcMV2_F4<<<MPnum, THnum1D>>>(vd, m, reinterpret_cast<double*>(tmp3N), N);
  double d;
  accumulate<<<1, threadsMax, sizeof(double)*threadsMax>>>(
    reinterpret_cast<double*>(tmp3N), N, reinterpret_cast<double*>(tmp3N));
  cudaMemcpy(&d, reinterpret_cast<double*>(tmp3N), sizeof(double), cudaMemcpyDeviceToHost);

  return d;
}

real GaussianThermo::scaleTemp(real Temp) {
  const real T0 = calcTemp();
  const double s = sqrt(static_cast<double>(3*N-1)/static_cast<double>(3*N)*Temp/T0);

  mulArray<<<MPnum, THnum1D>>>(vd, s, N*3);

  return static_cast<real>(s);
}

void GaussianThermo::adjustVelocities(real Temp, bool debug) {
  real v1 = sqrt(kB * Temp / m0);
  uint32_t __thnum = std::min((uint32_t)1024, threadsMax);

  if (debug) {
    std::cerr << std::endl << std::endl << "adjustVelocity currentTemp\t" << calcTemp()
      << "\tTarget Temp " << Temp << std::endl;

    adjustVelocity_VD<<<1, __thnum, sizeof(double)*__thnum*6>>>(vd, __thnum, N, v1, reinterpret_cast<float*>(tmp3N));
    cudaDeviceSynchronize();
    pthread_mutex_lock(&(mutTMP));
    cudaMemcpy(&(TMP[0]), tmp3N, sizeof(float)*6, cudaMemcpyDeviceToHost);
    std::cerr << "velocity statistics";
    for (int i=0;i<6;++i)
      std::cerr << "\t" << TMP[i];
    std::cerr << std::endl;
    pthread_mutex_unlock(&(mutTMP));
    std::cerr << "after adjusted T=" << calcTemp() << std::endl;
  } else {
    adjustVelocity_VD<<<1, __thnum, sizeof(double)*__thnum*6>>>(vd, __thnum, N, v1);
  }
}
