#if !defined(PUTTMPSELECTED)
#define PUTTMPSELECTED

#include <string>
#include <iostream>
#include <vector>
#include <valarray>
#include "cudaParticleBase.hh"

/** obtain and put selected particles which are not fixed
 *
 */
class putTMPselected : virtual public cudaParticleBase {
  std::vector<uint32_t> selected;   //! start and its size for unfixed particles
  std::vector<uint32_t> unselected; //! start and its size for FIXED particles

  friend void *putTMPinThread(void *arg); //!< putTMPinThread() is friend of mine to permit to read TMP[] directly from another thread
  friend class cudaParticleSPH_NS;

public:

  /** setup for selected/unselected particle range
   *
   * @param s1  start ID for selected particles
   * @param s2  number of the selected particles
   * @param u1  start ID for unselected particles
   * @param u2  number of the unselected particles
   */
  void setupSelectedTMP(uint32_t s1, uint32_t s2, uint32_t u1, uint32_t u2);

  //! replacement for getTypeID()
  void getSelectedTypeID(void);

  //! replacement for getPosition()
  void getSelectedPosition(void);

  //! put unselected particles only!
  void putUnSelected(const char *filename);


  template<class Archive>
  void serialize(Archive& ar, const uint32_t version);
};

template<class Archive>
void putTMPselected::serialize(Archive& ar, const uint32_t version) {
  ar & boost::serialization::make_nvp("selected", selected);
  ar &  boost::serialization::make_nvp("unselected", unselected);
}
#endif
