#if !defined(PARTICLEBASE)
#define PARTICLEBASE

/** data class to store particle position/mass/velocity/acceleration
 *
 *  @author  Shun Suzuki
 *  @date    2009 Sep.
 *  @version @$Id:$
 */
class ParticleBase {
public:
  double r[3];  //!< position
  double m;     //!< mass
  double v[3];  //!< velocity
  bool isFixed; //!< flag for the fixed particles
  double a[3];  //!< acceleration
  uint32_t type;  //!< particle type
};

#endif
