#if !defined(__ADAPTIMETIME_HH__)
#define __ADAPTIMETIME_HH__

/** a service class for adaptive time mesh control
 *
 */

template <class T>
class AdaptiveTime
{
public:
  void init(real initial_delta_t) { deltaT = initial_delta_t; }

  real operator()(void) const { return thistime; }

  bool isRollbacking(void) const { return rollback_in_progress; }

  real currentDeltaT(void) const { return deltaT; }
  /** 
   * resmax==-1: deltaT *= sqrt(2), plus time evolution
   * resmax== 0: ordinary time evolution
   * resmax== 1: deltaT /= 4 with roll backing
   * @param res
   * @return true: increment the loop counter, false: second stage of roll backing, and no need to increment the loop counter
   */
  uint32_t Progress(T &particles, const std::vector<int> &res, uint32_t j)
  {
    const uint32_t ndev = particles.nDevices();
    if (!rollback_in_progress)
    {
      thistime += deltaT;
      ++j;

      const int resmax = *std::max_element(res.begin(), res.end());

      if (resmax == 1)
      {
        // rollback from t+dt => t-dt, and delta_t /= 4
#pragma omp for
        for (int i = 0; i < ndev; ++i)
        {
          particles[i].rollback(deltaT); //XXX
        }
        rollback_in_progress = true;
      }
      else if ((resmax == -1) && (wait_change_deltaT == 0))
      {
        // v(t+delta_t) => v(t)
#pragma omp for
        for (int i = 0; i < ndev; ++i)
        {
          particles[i].calcVinit((3.0 - growthratio) * deltaT); //XXX
        }
        deltaT *= growthratio;
        wait_change_deltaT = 1;
#pragma omp master
        {
          if (statOutput)
            std::cerr << std::endl
                      << "DeltaT(+)"
                      << ":" << thistime << ":" << j << ":" << deltaT << std::endl;
        }
      }
      if (wait_change_deltaT > 0)
        --wait_change_deltaT;
    }
    else
    {
      //rollback_in_progress
#pragma omp master
      {
        if (statOutput)
          std::cerr << "rollback_step2:" << std::flush;
      }
#pragma omp for
      for (int i = 0; i < ndev; ++i)
      {
        particles[i].rollback2(deltaT); //XXX
      }
      rollback_in_progress = false;

      thistime += -deltaT;
      deltaT /= 4.0;
      wait_change_deltaT = 3;

#pragma omp master
      {
        if (statOutput)
          std::cerr << "done: " << std::flush;
        if (statOutput)
          std::cerr << std::endl
                    << "DeltaT(-)"
                    << ":" << thistime << ":" << j << ":" << deltaT << std::endl;
      }
    }
    return j;
  }

  void PrintStat(uint32_t j = 0)
  {
#pragma omp master
    {
      if (statOutput)
        std::cerr << "DeltaT(0)"
                  << ":" << thistime << ":" << j << ":" << deltaT << std::endl;
    }
  }

  bool statOutput = false;

private:
  real deltaT = 0.0;
  double thistime = 0.0;
  bool rollback_in_progress = false;
  uint32_t wait_change_deltaT = 0;
  const real growthratio = sqrt(2.0);
};

#endif
