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
 * overriding float4 r[] as r.w = radius
 *
 *  @author  Shun Suzuki
 *  @date    2012 Aug.
 *  @version @$Id:$
 */
class cudaParticleDEM : public cudaParticleLF, public cudaParticleRotation, public cudaSelectedBlock, public putTMPselected {
protected:
  real E;
  real mu;
  real s2;
  real gamma;

  real mu_r;

  real *tmp81N_TRQ; //!< 3N array for 27blocks on GPU

  cudaSparseMat contactMat[2];
  uint32_t _N;    //!< (_N%2) determines which contactMat would be used

  uint32_t *bid_by_pid; //!< bid table directly converted from pid

public:
  cudaParticleDEM() : tmp81N_TRQ(NULL), _N(0), autotunetimestep(false) { SingleParticleBlock = true; };

  ~cudaParticleDEM();

  /** construct arrays on GPU for n particles
   *
   * @param n number of particles
   */
  void setup(int n);

  /** calculate contact forces
   *
   */
  void calcForce(void) {};
  void calcForce(real dt);

  /** do time evolution by 1-dimentional grids and blocks for both translation and rotation
   *
   *
   * @param dt  Delta_t
   */
  void TimeEvolution(real dt);

  /** set property parameters for DEM
   *
   * @param _E  Young Modulus
   * @param _mu friction constant
   * @param _sigma  Poisson Ratio
   * @param _gamma  dumping factor for normal direction
   * @param _mu_r
   * @param r0_all  radius for all DEM particles
   */
  void setDEMProperties(real _E, real _mu, real _sigma, real _gamma, real _mu_r,
                        real r0_all);
  void setDEMProperties(real _E, real _mu, real _sigma, real _gamma, real _mu_r,
                        const std::vector<real> &_r0);

  /** override/kill cudaSelectedBlock::selectBlocks()
   *
   */
  void selectBlocks(void) {}


  /** override calcBID(); in WithBlock4, sort is no longer needed
   *
   */
  void calcBlockID(void);

  /** get Forces with moving particle ID range (p3, p4) (override cudaSelectedBlock)
   *
   * @param typeID  0: force, 2: torque
   */
  void getForceSelected(const ExchangeMode typeID);

  /** push a fragments of moving particles to another board
   *  (without summation)  (override cudaSelectedBlock)
   *
   * @param typeID  0: force, 2: torque
   */
  void importForceSelected(const cudaParticleDEM &A, const ExchangeMode typeID,
    bool directAccess=false, int idMe=0, int idPeer=0);

  /**
   *
   */
  void rollback(real dt);

  /**
   *
   */
  void rollback2(real dt);

  bool autotunetimestep; //!< true when AdaptiveTimeMesh introduced

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
void cudaParticleDEM::save(Archive& ar, const uint32_t version) const {
  ar << boost::serialization::make_nvp("cudaParticleLF",
        boost::serialization::base_object<cudaParticleLF>(*this));
  ar << boost::serialization::make_nvp("cudaParticleRotation",
        boost::serialization::base_object<cudaParticleRotation>(*this));
  ar << boost::serialization::make_nvp("selectedBlock",
        boost::serialization::base_object<cudaSelectedBlock>(*this));
  ar << boost::serialization::make_nvp("putTMPselected",
        boost::serialization::base_object<putTMPselected>(*this));

  ar << boost::serialization::make_nvp("E", E);
  ar << boost::serialization::make_nvp("mu", mu);
  ar << boost::serialization::make_nvp("s2", s2);
  ar << boost::serialization::make_nvp("gamma", gamma);

  const int __I = ((_N-1)%2);
  ar << boost::serialization::make_nvp("cudaSparseMat",contactMat[__I]);
}

template<class Archive>
void cudaParticleDEM::load(Archive& ar, const uint32_t version) {
  ar >> boost::serialization::make_nvp("cudaParticleLF",
        boost::serialization::base_object<cudaParticleLF>(*this));
  ar >> boost::serialization::make_nvp("cudaParticleRotation",
        boost::serialization::base_object<cudaParticleRotation>(*this));

  ar >> boost::serialization::make_nvp("selectedBlock",
        boost::serialization::base_object<cudaSelectedBlock>(*this));
  ar >> boost::serialization::make_nvp("putTMPselected",
        boost::serialization::base_object<putTMPselected>(*this));
  if (resizeInLoad) {
    if (tmp81N_TRQ!=NULL) cudaFree(tmp81N_TRQ);
    if (bid_by_pid!=NULL) cudaFree(bid_by_pid);
    cudaMalloc((void **)&bid_by_pid, sizeof(uint32_t)*N);
  }

  ar >> boost::serialization::make_nvp("E", E);
  ar >> boost::serialization::make_nvp("mu", mu);
  ar >> boost::serialization::make_nvp("s2", s2);
  ar >> boost::serialization::make_nvp("gamma", gamma);



  contactMat[0].setup(N, threadsMax);
  contactMat[1].setup(N, threadsMax);

  clearArray<uint32_t><<<MPnum/2, THnum2D>>>(contactMat[0].rowPtr, N+1);
  clearArray<uint32_t><<<MPnum/2, THnum2D>>>(contactMat[1].rowPtr, N+1);

  _N = 0;
  ar >> boost::serialization::make_nvp("cudaSparseMat",contactMat[1]);
}
#endif /* CUDAPARTICLEDEM */
