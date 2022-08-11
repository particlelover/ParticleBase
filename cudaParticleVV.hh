#if !defined(CUDAPARTICLEVV)
#define CUDAPARTICLEVV

#include "cudaParticleBase.hh"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/split_member.hpp>

/** intermediate class which implements Velocity-Verlet time evolution
 *
 *  @author  Shun Suzuki
 *  @date    2012 Feb.
 *  @version @$Id:$
 */
class cudaParticleVV : public virtual cudaParticleBase {
protected:
  float4 *Fold; //!< previous Force (array on GPU size 3N)

public:
  cudaParticleVV() : Fold(NULL) {};

  /** construct arrays on GPU for n particles
   *
   * @param n number of particles
   */
  void setup(int n);

  /** do time evolution by 1-dimentional grids and blocks by Velocity Verlet method
   *
   * calculates position and velocity from force F and previous force Fold
   * as \f{eqnarray*}{
   *
   * v(t)    &=& v(t-dt) + dt/m * (F(t)+F(t-dt))/2 \\
   * r(t+dt) &=& r(t) + dt * v(t) + dt^2/m * F(t)/2
   *
   * \f}
   *
   * @param dt  Delta_t
   */
  void TimeEvolution(real dt);

  /** clearing array for force F by 1-dimentional grids and blocks
   *
   */
  void clearForce(void);

private:
  friend class boost::serialization::access;
#if !defined(SWIG)
  BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif

  template<class Archive>
  void save(Archive& ar, const uint32_t version) const;

  template<class Archive>
  void load(Archive& ar, const uint32_t version);
};

template<class Archive>
void cudaParticleVV::save(Archive& ar, const uint32_t version) const {
  ar << boost::serialization::make_nvp("particleBase",
        boost::serialization::base_object<cudaParticleBase>(*this));

}

template<class Archive>
void cudaParticleVV::load(Archive& ar, const uint32_t version) {
  ar >> boost::serialization::make_nvp("particleBase",
        boost::serialization::base_object<cudaParticleBase>(*this));

  Fold = a;
}


#endif /* CUDAPARTICLEVV */
