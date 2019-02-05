#if !defined(CUDAPARTICLEDEM)
#define CUDAPARTICLEDEM

#include "cudaParticleLF.hh"
#include "cudaParticleRotation.hh"
#include "cudaSelectedBlock.hh"
#include "putTMPselected.hh"
#include "cudaSparseMat.hh"
#include "kerneltemplate.hh"

/** successor class for the DEM simulation
 *
 */
class cudaParticleDEM : public cudaParticleLF, public cudaParticleRotation, public cudaSelectedBlock, public putTMPselected
{
protected:
  real *r0; //!< pointer to the array on GPU for radius (size N)

  real E;
  real mu;
  real s2;
  real gamma;

  real mu_r;

  //#if !defined(WITHJLIST)
  real *tmp81N_TRQ; //!< 3N array for 27blocks on GPU
                    //#endif

  cudaSparseMat contactMat[2];
  uint32_t _N; //!< (_N%2) determines which contactMat would be used

#if defined(WithBlock2)
  uint32_t *bid_by_pid; //!< bid table directly converted from pid
#endif

public:
  //#if !defined(WITHJLIST)
  cudaParticleDEM() : r0(NULL), tmp81N_TRQ(NULL), _N(0) { SingleParticleBlock = false; };
  //#else
  //  cudaParticleDEM() : r0(NULL), _N(0) { SingleParticleBlock = false; };
  //#endif

  ~cudaParticleDEM();

  /** construct arrays on GPU for n particles
   *
   * @param n	number of particles
   */
  void setup(int n);

  /** calculate contact forces
   *
   */
  void calcForce(void){};
  void calcForce(real dt);

  /** do time evolution by 1-dimentional grids and blocks for both translation and rotation
   *
   *
   * @param dt	Delta_t
   */
  void TimeEvolution(real dt);

  /** set property parameters for DEM
   *
   * @param _E	Young Modulus
   * @param _mu	friction constant
   * @param _sigma	Poisson Ratio
   * @param _gamma	dumping factor for normal direction
   * @param _mu_r
   * @param _r0		radius for all DEM particles
   */
  void setDEMProperties(real _E, real _mu, real _sigma, real _gamma, real _mu_r,
                        const std::vector<real> &_r0);

  /** set property parameters for DEM
   *
   * @param _E	Young Modulus
   * @param _mu	friction constant
   * @param _sigma	Poisson Ratio
   * @param _gamma	dumping factor for normal direction
   * @param _mu_r
   * @param r0_all	radius for all DEM particles
   */
  void setDEMProperties(real _E, real _mu, real _sigma, real _gamma, real _mu_r,
                        real r0_all);

#if defined(WithBlock2)
  /** override/kill cudaSelectedBlock::selectBlocks()
   *
   */
  void selectBlocks(void) {}
#endif

  /** override calcBID(); in WithBlock4, sort is no longer needed
   *
   */
  void calcBlockID(void);

  /** get Forces with moving particle ID range (p3, p4) (override cudaSelectedBlock)
   *
   * @param typeID	0: force, 2: torque
   */
  void getForceSelected(const int typeID);

  /** push a fragments of moving particles to another board
   *  (without summation)  (override cudaSelectedBlock)
   *
   * @param typeID	0: force, 2: torque
   */
  void importForceSelected(const cudaParticleDEM &A, const int typeID,
                           bool directAccess = false, int idMe = 0, int idPeer = 0);

  /**
   *
   */
  void rollback(real dt);

  /**
   *
   */
  void rollback2(real dt);

private:
  class calcCoord C;
  class DEMContact P;

  void setFunctorMembers(void);

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
void cudaParticleDEM::save(Archive &ar, const uint32_t version) const
{
  ar << boost::serialization::make_nvp("cudaParticleLF",
                                       boost::serialization::base_object<cudaParticleLF>(*this));
  ar << boost::serialization::make_nvp("cudaParticleRotation",
                                       boost::serialization::base_object<cudaParticleRotation>(*this));
  ar << boost::serialization::make_nvp("selectedBlock",
                                       boost::serialization::base_object<cudaSelectedBlock>(*this));
  ar << boost::serialization::make_nvp("putTMPselected",
                                       boost::serialization::base_object<putTMPselected>(*this));

  std::vector<real> _tmp(N);
  const size_t sizeN = sizeof(real) * N;

  cudaMemcpy(&(_tmp[0]), r0, sizeN, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("radiusR0", _tmp);

  ar << boost::serialization::make_nvp("E", E);
  ar << boost::serialization::make_nvp("mu", mu);
  ar << boost::serialization::make_nvp("s2", s2);
  ar << boost::serialization::make_nvp("gamma", gamma);

  const int __I = ((_N - 1) % 2);
  ar << boost::serialization::make_nvp("cudaSparseMat", contactMat[__I]);
}

template <class Archive>
void cudaParticleDEM::load(Archive &ar, const uint32_t version)
{
  ar >> boost::serialization::make_nvp("cudaParticleLF",
                                       boost::serialization::base_object<cudaParticleLF>(*this));
  ar >> boost::serialization::make_nvp("cudaParticleRotation",
                                       boost::serialization::base_object<cudaParticleRotation>(*this));

  ar >> boost::serialization::make_nvp("selectedBlock",
                                       boost::serialization::base_object<cudaSelectedBlock>(*this));
  ar >> boost::serialization::make_nvp("putTMPselected",
                                       boost::serialization::base_object<putTMPselected>(*this));
  if (resizeInLoad)
  {
    if (r0 != NULL)
      cudaFree(r0);

    cudaMalloc((void **)&r0, sizeof(real) * N);

#if !defined(WITHJLIST)
    if (tmp81N_TRQ != NULL)
      cudaFree(tmp81N_TRQ);
    cudaMalloc((void **)&tmp81N_TRQ, sizeof(real) * 3 * N * 27);
#endif

#if defined(WithBlock2)
    if (bid_by_pid != NULL)
      cudaFree(bid_by_pid);
    cudaMalloc((void **)&bid_by_pid, sizeof(uint32_t) * N);
#endif
  }

  std::vector<real> _tmp(N);
  const size_t sizeN = sizeof(real) * N;

  ar >> boost::serialization::make_nvp("radiusR0", _tmp);
  cudaMemcpy(r0, &(_tmp[0]), sizeN, cudaMemcpyHostToDevice);

  ar >> boost::serialization::make_nvp("E", E);
  ar >> boost::serialization::make_nvp("mu", mu);
  ar >> boost::serialization::make_nvp("s2", s2);
  ar >> boost::serialization::make_nvp("gamma", gamma);

  setFunctorMembers();

  contactMat[0].setup(N, threadsMax);
  contactMat[1].setup(N, threadsMax);

  clearArray<uint32_t><<<MPnum / 2, THnum2D>>>(contactMat[0].rowPtr, N + 1);
  clearArray<uint32_t><<<MPnum / 2, THnum2D>>>(contactMat[1].rowPtr, N + 1);

  _N = 0;
  ar >> boost::serialization::make_nvp("cudaSparseMat", contactMat[1]);
}
#endif /* CUDAPARTICLEDEM */
