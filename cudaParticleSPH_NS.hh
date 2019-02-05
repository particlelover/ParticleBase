#include "cudaParticleSPHBase.hh"
#include "putTMPselected.hh"

#define BINDEX_SEARCH_WIDTH 64

/** a class for the Navier-Stokes simulation with SPH particles
 *
 */
class cudaParticleSPH_NS : public cudaParticleSPHBase, public putTMPselected
{
protected:
  real *mu0; //!< shear viscosity

  real *c2; //!< (sound velocity)^2

  real *mu; //!< viscosity for field value (calculated as \mu \rho)
public:
  cudaParticleSPH_NS() : mu0(NULL), c2(NULL), mu(NULL), ___p1(BINDEX_SEARCH_WIDTH), ___p2(BINDEX_SEARCH_WIDTH){};

  ~cudaParticleSPH_NS();

  /** construct arrays on GPU and inner TMP array for n particles
   *
   * @param n	number of particles
   */
  void setup(int n);

  void getPosition(void);

  virtual std::string additionalTag(void) const { return " n rho p"; };

  virtual std::string additionalOutput(uint32_t i) const;

  /** calculate accelerations dirctly from all i-j pairs by 1-dimentional grids and blocks
   *  by Navier-Stokes Master equation
   *
   * @param sortedOutput	if true, outputs calculated acceleration with i0 instead of pid[i0]
   */
  void calcAcceleration(bool sortedOutput = false);

  /** calcForce does nothing for the Navier-Stokes simulation
   *
   */
  void calcForce(void){};

  /** override SPHBase::calcDensity() to calculate pressure with mass/numver density
   *
   * @param sortedOutput	if true, array for num/rho/mu(size 3N) will be shuffled
   */
  void calcDensity(bool sortedOutput = false);

  /** post process of the calc density after the exchange
   *
   *  if calcDensity() was performed with sortedOutput=false, this method invoked directly from calcDensity()
   */
  void calcDensityPost(bool sortedOutput = false);

  /** set properties for SPH calculation
   *
   * @param _mu0	shear viscosity mu for each SPH particles
   * @param _c1	sound velocity c for each SPH particles (for the calculation of pressure)
   * @param _h	radius of the SPH particles
   */
  void setSPHProperties(const std::valarray<real> &_mu0, std::valarray<real> _c1, real _h);

  real inspectDensity(void);

  /** restore a[] by pid; a[pid[i]] = a[i]
   *
   */
  void RestoreAcceleration(void);

  /** get particle ID p1 and p2 from the selected block range
   *
   *
   */
  void getExchangePidRange1(void);

  /** get particle ID p3 and p4 from the selected block range
   *
   *
   */
  void getExchangePidRange2(void);

  /** get num/rho/mu with ID range (p1, p2) (override cudaSelectedBlock)
   *
   * @param typeID	4: num/rho/mu
   */
  void getForceSelected(const int typeID);

  /** push a fragments of moving particles to another board
   *  (without summation)  (override cudaSelectedBlock)
   *
   * @param typeID	4: num/rho/mu
   */
  void importForceSelected(const cudaParticleSPH_NS &A, const int typeID,
                           bool directAccess = false, int idMe = 0, int idPeer = 0);

private:
  //! temporal area to obtain a part of the bindex[]
  std::vector<uint32_t> ___p1, ___p2;

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
void cudaParticleSPH_NS::save(Archive &ar, const uint32_t version) const
{
  ar << boost::serialization::make_nvp("cudaParticleSPHBase",
                                       boost::serialization::base_object<cudaParticleSPHBase>(*this));
  ar << boost::serialization::make_nvp("putTMPselected",
                                       boost::serialization::base_object<putTMPselected>(*this));

  std::vector<real> _tmp(N * 3);
  const size_t sizeN = sizeof(real) * N;

  cudaMemcpy(&(_tmp[0]), mu0, sizeN, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("viscosityMu", _tmp);

  cudaMemcpy(&(_tmp[0]), c2, sizeN, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("soundVelocityC2", _tmp);
  cudaMemcpy(&(_tmp[0]), mu, sizeN, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("fieldViscosity", _tmp);
}

template <class Archive>
void cudaParticleSPH_NS::load(Archive &ar, const uint32_t version)
{
  ar >> boost::serialization::make_nvp("cudaParticleSPHBase",
                                       boost::serialization::base_object<cudaParticleSPHBase>(*this));
  ar >> boost::serialization::make_nvp("putTMPselected",
                                       boost::serialization::base_object<putTMPselected>(*this));

  if (resizeInLoad)
  {
    //    if (mu0!=NULL)  cudaFree(mu0);
    //    cudaMalloc((void **)&mu0, sizeof(real)*N);
    mu = &(num[N * 2]);

    if (c2 != NULL)
      cudaFree(c2);
    cudaMalloc((void **)&c2, sizeof(real) * N);
    if (mu != NULL)
      cudaFree(mu);
    cudaMalloc((void **)&mu, sizeof(real) * N);
  }

  TMP.resize(N * 6);

  std::vector<real> _tmp(N * 3);
  const size_t sizeN = sizeof(real) * N;

  ar >> boost::serialization::make_nvp("viscosityMu", _tmp);
  cudaMemcpy(mu0, &(_tmp[0]), sizeN, cudaMemcpyHostToDevice);

  ar >> boost::serialization::make_nvp("soundVelocityC2", _tmp);
  cudaMemcpy(c2, &(_tmp[0]), sizeN, cudaMemcpyHostToDevice);
}
