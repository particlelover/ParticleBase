#if !defined(CUDAPARTICLEBASE)
#define CUDAPARTICLEBASE

#include <vector>
#include <valarray>
#include <string>
#include <iostream>
#include <pthread.h>
#include "cudaFrame.hh"
#include "ParticleBase.hh"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/split_member.hpp>

/** base class for the basic particle styled simulation with CUDA
 *
 */
class cudaParticleBase : public cudaFrame
{
protected:
  uint32_t N; //!< number of particles

  real *r; //!< pointer to the array on GPU for x[N] y[N] z[N]

  real *v; //!< pointer to the array on GPU for vx[N] vy[N] vz[N]

  real *a; //!< pointer to the array on GPU for ax[N] ay[N] az[N]

  real *minv; //!< pointer to the array on GPU for 1/mass [N]

  real *F; //!< pointer to the array on GPU for Fx[N] Fy[N] Fz[N]

  real *tmp3N; //!< pointer to the array on GPU for the temporary area

  unsigned short *typeID; //!< pointer to the array on GPU for type ID

  char *move; //!< pointer to the array on GPU for the movable flag(0=stop) size N

  bool resizeInLoad; //!< check flag for the serialization

public:
  int MPnum;           //!< grid size for CUDA kernel
  int THnum1D;         //!< block size for 1D CUDA kernel (num of core x 4)
  int THnum2D;         //!< block size for 2D CUDA kernel (i-j pair) (num of core /2)
  uint32_t threadsMax; //!< max number of threads of this GPU
  uint32_t maxGrid;    //!< Max Dimension size of grid (X)

  cudaParticleBase() : r(NULL), minv(NULL), v(NULL), a(NULL), F(NULL), tmp3N(NULL),
                       typeID(NULL), move(NULL),
                       resizeInLoad(false), N(0){};

  virtual ~cudaParticleBase();

  /** construct arrays on GPU for n particles
   *
   * @param n	number of particles
   */
  void setup(int n);

  /** import particles from CPU code
   *
   * @param P	so called globaltable
   */
  void import(const std::vector<ParticleBase> &P);

  /** calculate forces from all i-j pairs by 1-dimentional grids and blocks
   *  successor class should implement this
   *
   * @param cell	cell boundary parameter
   */
  virtual void calcForce(void) = 0;

  /** do time evolution by 1-dimentional grids and blocks by Euler method
   *
   * @param dt	Delta_t
   */
  void TimeEvolution(real dt);

  /** get position r on GPU to CPU memory
   *
   *  only perfomrs cudaMemcpy; obtained data keep stored in TMP[]
   *  use putTMP() to put them to ostream
   */
  void getPosition(void);

  /** get Acceleration a on GPU to CPU memory
   *
   *  only perfomrs cudaMemcpy; obtained data keep stored in TMP[]
   *  use putTMP() to put them to ostream
   */
  void getAcceleration(void);

  /** get Force a on GPU to CPU memory
   *
   *  only perfomrs cudaMemcpy; obtained data keep stored in TMP[]
   *  use putTMP() to put them to ostream
   */
  void getForce(void);

  /** get Type ID from GPU to CPU memory
   *
   *  only perfomrs cudaMemcpy; obtained data are kept in TMP2[]
   *  for the LAMMPS dump style outputs.
   *  the size of TMP2[] defines the number of particles to output.
   */
  void getTypeID(void);

  /** clearing array for force F by 1-dimentional grids and blocks
   *
   */
  void clearForce(void);

  /** clearing array tmp3N[]
   *
   */
  void clearTmp3N(void);

  /** calculate accelerations from forces by 1-dimentional grids and blocks
   *
   */
  void calcAcceleration(void);

  /** treat periodic boundary condition of the simulation cell
   *
   */
  void treatPeriodicCondition(void);

  /** treat periodic boundary condition of the simulation cell
   *
   */
  void treatAbsoluteCondition(void);

  /** add force to particles in X direction
   *
   * @param fx		force added in X direction
   */
  void addForceX(real fx);

  /** add force to particles in Y direction
   *
   * @param fy		force added in Y direction
   */
  void addForceY(real fy);

  /** add force to particles in Z direction
   *
   * @param fz		force added in Z direction
   */
  void addForceZ(real fz);

  /** add acceleration to particles in X direction
   *
   * @param ax		acceleration added in X direction
   */
  void addAccelerationX(real ax);

  /** add acceleration to particles in Y direction
   *
   * @param ay		acceleration added in Y direction
   */
  void addAccelerationY(real ay);

  /** add acceleration to particles in Z direction
   *
   * @param az		acceleration added in Z direction
   */
  void addAccelerationZ(real az);

  /** inspect Velocity
   *
   * calculate the velocity of all particles, and the ratio of it
   * against the maximum velocity vlim
   * if this ratio is too large, programer should treat some rollback process
   *
   * @param vlim	limit velocity (2R/\Delta t)
   * @param lim_u upper	limit for the ratio
   * @param lim_l lower	limit for the ratio
   * @param _r  largest	*ratio*; largest v / v_limit
   * @param DEBUG	if true, obtain all vratio
   * @return	1: vratio>lim_u, 0: lim_u>vratio>lim_l, -1: lim_l>vratio
   */
  int inspectVelocity(real vlim, real lim_u, real lim_l, real &_r, bool debug = false);

  /** as an interface to the inspectVelocity() for swig/python
   *
   */
  int inspectVelocity(real vlim, real lim_u, real lim_l)
  {
    real _r = 0.0;
    return inspectVelocity(vlim, lim_u, lim_l, _r, false);
  }

  /** output particles property as a 3N array (velocity, force,...)
   *
   * @param A		array to dump
   * @param o		output stream
   */
  void dump3Narray(real *A, std::ostream &o = std::cout);

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
void cudaParticleBase::save(Archive &ar, const uint32_t version) const
{
  ar << boost::serialization::make_nvp("cudaFrame",
                                       boost::serialization::base_object<cudaFrame>(*this));

  ar << boost::serialization::make_nvp("particleNumN", N);

  std::vector<real> _tmp(N * 3);
  const size_t size3N = sizeof(real) * N * 3;

  cudaMemcpy(&(_tmp[0]), r, size3N, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("positionR", _tmp);

  cudaMemcpy(&(_tmp[0]), v, size3N, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("velocityV", _tmp);

  cudaMemcpy(&(_tmp[0]), a, size3N, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("accelerationA", _tmp);

  std::vector<real> _tmp1(N);
  const size_t sizeN = sizeof(real) * N;
  cudaMemcpy(&(_tmp1[0]), minv, sizeN, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("massInverseMinv", _tmp1);

  cudaMemcpy(&(_tmp[0]), F, size3N, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("forceF", _tmp);

  const size_t sizeN2 = sizeof(unsigned short) * N;
  std::vector<unsigned short> _tmp2(N);
  cudaMemcpy(&(_tmp2[0]), typeID, sizeN2, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("typeID", _tmp2);

  const size_t sizeN3 = sizeof(char) * N;
  std::vector<char> _tmp3(N);
  cudaMemcpy(&(_tmp3[0]), move, sizeN3, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("moveFlag", _tmp3);

  cudaError_t t = cudaGetLastError();
  if (t != 0)
    std::cerr << "cudaParticleBase::save: "
              << cudaGetErrorString(t) << std::endl;
}

template <class Archive>
void cudaParticleBase::load(Archive &ar, const uint32_t version)
{
  ar >> boost::serialization::make_nvp("cudaFrame",
                                       boost::serialization::base_object<cudaFrame>(*this));

  uint32_t _N;
  ar >> boost::serialization::make_nvp("particleNumN", _N);
  if (N != _N)
  {
    std::cerr << "resizing particleNum from " << N << " to " << _N << std::endl;
    resizeInLoad = true;
    N = _N;

    if (r != NULL)
      cudaFree(r);
    if (minv != NULL)
      cudaFree(minv);
    if (v != NULL)
      cudaFree(v);
    if (a != NULL)
      cudaFree(a);
    if (F != NULL)
      cudaFree(F);
    if (tmp3N != NULL)
      cudaFree(tmp3N);
    if (typeID != NULL)
      cudaFree(typeID);
    if (move != NULL)
      cudaFree(move);

    cudaMalloc((void **)&r, sizeof(real) * 3 * N);
    cudaMalloc((void **)&minv, sizeof(real) * N);
    cudaMalloc((void **)&v, sizeof(real) * 3 * N);
    cudaMalloc((void **)&a, sizeof(real) * 3 * N);
    cudaMalloc((void **)&F, sizeof(real) * 3 * N);
    cudaMalloc((void **)&tmp3N, sizeof(real) * 3 * N);
    cudaMalloc((void **)&typeID, sizeof(unsigned short int) * N);
    cudaMalloc((void **)&move, sizeof(char) * N);
  }
  else
  {
    resizeInLoad = false;
  }

  TMP.resize(N * 3);
  TMP2.resize(N);

  std::vector<real> _tmp(N * 3);
  const size_t size3N = sizeof(real) * N * 3;

  ar >> boost::serialization::make_nvp("positionR", _tmp);
  cudaMemcpy(r, &(_tmp[0]), size3N, cudaMemcpyHostToDevice);

  ar >> boost::serialization::make_nvp("velocityV", _tmp);
  cudaMemcpy(v, &(_tmp[0]), size3N, cudaMemcpyHostToDevice);

  ar >> boost::serialization::make_nvp("accelerationA", _tmp);
  cudaMemcpy(a, &(_tmp[0]), size3N, cudaMemcpyHostToDevice);

  std::vector<real> _tmp1(N);
  const size_t sizeN = sizeof(real) * N;
  ar >> boost::serialization::make_nvp("massInverseMinv", _tmp1);
  cudaMemcpy(minv, &(_tmp1[0]), sizeN, cudaMemcpyHostToDevice);

  ar >> boost::serialization::make_nvp("forceF", _tmp);
  cudaMemcpy(F, &(_tmp[0]), size3N, cudaMemcpyHostToDevice);

  const size_t sizeN2 = sizeof(unsigned short) * N;
  std::vector<unsigned short> _tmp2(N);
  ar >> boost::serialization::make_nvp("typeID", _tmp2);
  cudaMemcpy(typeID, &(_tmp2[0]), sizeN2, cudaMemcpyHostToDevice);

  const size_t sizeN3 = sizeof(char) * N;
  std::vector<char> _tmp3(N);
  ar >> boost::serialization::make_nvp("moveFlag", _tmp3);
  cudaMemcpy(move, &(_tmp3[0]), sizeN3, cudaMemcpyHostToDevice);

  cudaError_t t = cudaGetLastError();
  if (t != 0)
    std::cerr << "cudaParticleBase::load: "
              << cudaGetErrorString(t) << std::endl;
}

/** \mainpage cudaParticles
 *
 * \section Brief Description
 * Basic classes for the particle style simulations such as
 * Molecular Dynamics, Smoothed Particle Hydrodynamics,
 * Distinct Element Method, by using CUDA.
 *
 * Time evolutions of points, calculation of forces of i-j pairs,
 * pair potentials are defined in CUDA kernel functions.
 * CUBLAS is also used.
 *
 * Support serialization of particles object on GPU by boost.
 *
 * OpenMP and pthread are also used.
 *
 */
#endif /* CUDAPARTICLEBASE */
