#include "GaussianThermo.hh"
#include "kernelfuncs.h"

GaussianThermo::~GaussianThermo()
{
}

void GaussianThermo::setup(int n)
{
  cudaParticleMD::setup(n);
}

void GaussianThermo::TimeEvolution(real dt)
{

  // 1) calc A(t) vector
  // A(t) = v(t-dt)+(F(t)+F(t-dt))dt/2m - v(t-dt)xi(t-dt)dt/2
  // a (acceleration) is used for this A1 because a is not used at velocity-velret
  calcGaussianThermoA1<<<MPnum, THnum1D>>>(a, dt, v, F, Fold, minv, N, xi);

  // 2) solve v_i = A_i - dt/2 v_i xi(v) by Newton Raphson Method
  // by A_i - v_i - dt/2 v_i xi(v) = 0
  //
  real v0;
  uint32_t _N = 0;
  real mv2inv = 1.0 / calcMV2(); // \sum mv^2 (use tmp3N)
  do
  {
    if (_N++ > 50)
    {
      std::cerr << "iteration not converged" << std::endl;
      std::cerr << v0 << ">" << _e << std::endl;
      exit(0);
    }

    // 2.2)
    calcGaussianThermoFoverF<<<MPnum, THnum1D>>>(a, dt, v, F, m, tmp3N, N, xi, mv2inv);
    // v[] also updated; f/f' => tmp3N

    // 2.3)	// evaluate f/f'
    DOT(hdl, N * 3, tmp3N, 1, tmp3N, 1, &v0);
    v0 = sqrt(v0 / (N * 3));

    // 2.4) update xi, mv2
    DOT(hdl, N * 3, F, 1, v, 1, &xi);
    mv2inv = 1.0 / calcMV2(); // \sum mv^2 for next (use tmp3N)
    xi *= mv2inv;             // recalculate xi

    //	std::cerr << "v0=" << v0 << " xi=" << xi << "\t" <<std::flush;
  } while (v0 > _e);

  // 3) calc r(t+dt)
  propagateVelocityVerletGaussianThermo<<<MPnum, THnum1D>>>(r, dt, v, F, Fold, minv, N, xi);

  if (withInfo)
    ErrorInfo("GaussianThermo::TimeEvolution");
}
