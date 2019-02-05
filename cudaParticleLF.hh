#if !defined(CUDAPARTICLELF)
#define CUDAPARTICLELF

#include "cudaParticleBase.hh"

/** intermediate class which implements Leap-Frog time evolution
 *
 */
class cudaParticleLF : public virtual cudaParticleBase
{
protected:
public:
  /** calculates \f$ v(0-\Delta t/2) \f$ from F(0) and v(0); F(0) is also calculated from calcForce()
   *
   * \f[
   * v(-{\Delta t}/2) = v(0) - {\Delta t/}/2 F(0)/m
   * \f]
   *
   * @param dt	Delta_t
   */
  void calcVinit(real dt);

  /** do time evolution by 1-dimentional grids and blocks by Leap Frog method
   *
   * calculates position and velocity from force F
   * as \f{eqnarray*}{
   *
   * v(t+dt/2) &=& v(t-dt/2) + dt * F(t)/m \\
   * r(t+dt)   &=& r(t) + dt * v(t)
   *
   * \f}
   *
   * @param dt	Delta_t
   */
  void TimeEvolution(real dt);

  /** rollback the position and velocity from t+dt to t-dt
   *
   */
  void rollback(real dt);

private:
  friend class boost::serialization::access;
#if !defined(SWIG)
  BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif

  template <class Archive>
  void save(Archive &ar, const uint32_t version) const
  {
    ar << boost::serialization::make_nvp("particleBase",
                                         boost::serialization::base_object<cudaParticleBase>(*this));
  }

  template <class Archive>
  void load(Archive &ar, const uint32_t version)
  {
    ar >> boost::serialization::make_nvp("particleBase",
                                         boost::serialization::base_object<cudaParticleBase>(*this));
  }
};

#endif /* CUDAPARTICLELF */
