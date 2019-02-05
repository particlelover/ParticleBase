#if !defined(GAUSSIANTHERMO)
#define GAUSSIANTHERMO

#include "cudaParticleMD.hh"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/split_member.hpp>

/** Gaussian Thermostat for the Velocity-Verlet
 *
 */
class GaussianThermo : public virtual cudaParticleMD
{
protected:
  real xi; //!< \f$ \dot{\xi} \f$ term for correction of forces

  real _e; //!< threshold for the Newton-Raphson in calculating \xi for velocity

public:
  GaussianThermo() : xi(0), _e(1e-8){};

  ~GaussianThermo();

  /** construct arrays on GPU for n particles
   *
   * @param n	number of particles
   */
  void setup(int n);

  /** do time evolution by 1-dimentional grids and blocks by Velocity Verlet method
   * with Gaussian Thermostat
   *
   * calculates position and velocity from force F and previous force Fold
   * as \f{eqnarray*}{
   *
   * v(t)    &=& v(t-dt) + dt/m * (F(t)+F(t-dt))/2 
   * - (v(t-dt)\dot{\xi}(t-dt)+v(t)\dot{\xi}(t))\frac{dt^2}{2} \\
   * \dot{\xi}(t)  &=& \sum F(t)\cdot v(t) / \sum m v(t)^2 \\
   * r(t+dt) &=& r(t) + dt * v(t) + dt^2/m * F(t)/2
   *  - v(t)\cdot \dot{\xi}(t)\cdot \frac{dt^2}{2}
   *
   * \f}
   * \f$\dot{\xi}(t)\f$ for the first equation is calculated in previous step
   * and v(t) is calcuated by Newton-Raphson method
   *
   * @param dt	Delta_t
   */
  void TimeEvolution(real dt);

  void setE(real _e0) { _e = _e0; };

private:
  friend class boost::serialization::access;
#if !defined(SWIG)
  BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif

  template <class Archive>
  void save(Archive &ar, const uint32_t version) const;

  template <class Archive>
  void load(Archive &ar, const uint32_t version);
};

template <class Archive>
void GaussianThermo::save(Archive &ar, const uint32_t version) const
{
  ar << boost::serialization::make_nvp("particleMD",
                                       boost::serialization::base_object<cudaParticleMD>(*this));

  ar << boost::serialization::make_nvp("xi", xi);
  ar << boost::serialization::make_nvp("_e", _e);
}

template <class Archive>
void GaussianThermo::load(Archive &ar, const uint32_t version)
{
  ar >> boost::serialization::make_nvp("particleMD",
                                       boost::serialization::base_object<cudaParticleMD>(*this));

  ar >> boost::serialization::make_nvp("xi", xi);
  ar >> boost::serialization::make_nvp("_e", _e);
}

#endif /* GAUSSIANTHERMO */
