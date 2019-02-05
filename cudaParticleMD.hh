#if !defined(CUDAPARTICLEMD)
#define CUDAPARTICLEMD

#include "cudaParticleVV.hh"
#include "cudaCutoffBlock.hh"
#include "cudaSortedPositions.hh"
#include <cublas_v2.h>
#include <iostream>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/split_member.hpp>

/** successor class for the simple MD simulation
 *
 * cudaParticleMD implements following basic methods for the simple 
 * Molecular Dynamics simulation.
 *
 * \li Lennard-Jones potential
 * \li Velocity-Verlet time evolution
 * \li periodic boundary condition
 * \li NVE ensemble with temperature control (scaling, constant Temperature)
 * \li dump atoms in LAMMPS dump format
 * \li soft core potential for the initial anealing
 * \li divide simulation cell into smaller blocks for the cutting off i-j pair search
 *
 *
 */
class cudaParticleMD : public cudaParticleVV, virtual public cudaCutoffBlock, virtual public cudaSortedPositions
{
protected:
  real *m; //!< pointer to the array on GPU for mass [N]

  cublasHandle_t hdl; //!< handlar for the CUBLAS operations
public:
  cudaParticleMD() : m(NULL), hdl(NULL){};

  virtual ~cudaParticleMD();

  /** construct arrays on GPU for n particles
   *
   * @param n	number of particles
   */
  void setup(int n);

  /** calculate forces from all i-j pairs by 1-dimentional grids and blocks
   *  using LJ potential
   *
   */
  void calcForce(void);

  /** set m = 1/(1/mass) for the calculation of temperature
   *
   */
  void setM(void);

  /** steal the calls to the cudaCutoffBlock::calcBlockID()
   *
   */
  void calcBlockID(void)
  {
    cudaCutoffBlock::calcBlockID();
    cudaSortedPositions::calcBlockID();
  }
  /** steal the calls to the cudaCutoffBlock::setupCutoffBlock()
   *
   */
  void setupCutoffBlock(real rmax, bool periodic = true)
  {
    cudaCutoffBlock::setupCutoffBlock(rmax, periodic);
    cudaSortedPositions::setupSortedPositions();
  }

protected:
  /** calculate the \f$ \sum m_i v_i^2 \f$
   *
   * @return	3N kB T
   */
  real calcMV2(void);

public:
  /** calculate the temperature from particles' velocities
   *
   * @return	temperature
   */
  real calcTemp(void)
  {
    return calcMV2() / (N * 3 * kB);
  }

  real kB; //!< Boltzmann constant in simulation unit

  /** correct forces by constant temperature scheme
   *  \f$ \lambda = \sum F_i v_i / \sum (m_i v_i^2), F_i - \lambda v_i = m a_i \f$
   *
   * @return	temperature
   */
  real constTemp(void);

  /** rescale velocities by the following ratio, where T0 is temperature
   * calculated from current velocities.
   *
   * \f[
   * s = \sqrt{(3N-1)/3N * T/T0}
   * \f]
   *
   * @param Temp	temperature to scale to
   * @return	scaling factor s
   */
  real scaleTemp(real Temp);

  /** adjust velocities by setting mean vx, vy, vz to zero
   *  and scale it to make standard deviations fit to specified temperature
   *
   * @param Temp	temperature to fit
   */
  void adjustVelocities(real Temp);

  /** proxy to the global function _setLJparams()
   * to avoid to include kerneltemplate.hh from main .cc source file
   *
   * @param p	vetor for LJ parameters as sigma, epsilon for all i-j types pair
   * @param elemnum	number of elements (<=MAXELEMNM)
   */
  void setLJparams(const std::vector<real> &p, uint32_t elemnum);

  /** performs initial annealing by Soft Core potential
   *
   * @param cell	cell borders
   * @param anealstep	number of step for the anealing
   * @param dt	time step \f$ \Delta t \f$
   * @param _rc	final rmax for the soft core potential
   * @param _f0	force coefficient for the soft core potential
   * @param T	temperature for the velocity scaling
   */
  void initialAnnealing(uint32_t anealstep, real dt, real _rc, real _f0, real T);

  /** output mv^2 for all particles
   *
   * @param o		output stream
   */
  void statMV2(std::ostream &o = std::cout);

  // dummy for SWIG
  void getForceSelected(const int typeID) {}
  void importForceSelected(const cudaParticleMD &A, const int typeID,
                           bool directAccess, int idMe, int idPeer) {}

  real rmax2; //!< square of cutoff distance rmax for I-J pair potential

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
void cudaParticleMD::save(Archive &ar, const uint32_t version) const
{
  ar << boost::serialization::make_nvp("particleVV",
                                       boost::serialization::base_object<cudaParticleVV>(*this));

  ar << boost::serialization::make_nvp("cutoffBlock",
                                       boost::serialization::base_object<cudaCutoffBlock>(*this));

  ar << boost::serialization::make_nvp("BoltzmannConstKb", kB);
  ar << boost::serialization::make_nvp("rmax2", rmax2);

  cudaError_t t = cudaGetLastError();
  if (t != 0)
    std::cerr << "cudaParticleVV::save: "
              << cudaGetErrorString(t) << std::endl;
}

template <class Archive>
void cudaParticleMD::load(Archive &ar, const uint32_t version)
{
  ar >> boost::serialization::make_nvp("particleVV",
                                       boost::serialization::base_object<cudaParticleVV>(*this));

  ar >> boost::serialization::make_nvp("cutoffBlock",
                                       boost::serialization::base_object<cudaCutoffBlock>(*this));

  ar >> boost::serialization::make_nvp("sortedPositions",
                                       boost::serialization::base_object<cudaSortedPositions>(*this));

  if (resizeInLoad)
  {
    if (m != NULL)
      cudaFree(m);
    cudaMalloc((void **)&m, sizeof(real) * N);

    if (hdl == NULL)
      cublasCreate(&hdl);
  }
  setM();

  ar >> boost::serialization::make_nvp("BoltzmannConstKb", kB);
  ar >> boost::serialization::make_nvp("rmax2", rmax2);

  cudaError_t t = cudaGetLastError();
  if (t != 0)
    std::cerr << "cudaParticleMD::load: "
              << cudaGetErrorString(t) << std::endl;
}

#endif /* CUDAPARTICLEMD */
