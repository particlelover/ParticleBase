#if !defined(CUDAPARTICLEROTATION)
#define CUDAPARTICLEROTATION

#include <vector>
#include <valarray>
#include <string>
#include <iostream>
#include "cudaParticleBase.hh"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/split_member.hpp>


/** base class for the Eulerian Equation of Motion of particles
 *
 *  @author  Shun Suzuki
 *  @date    2012 Dec.
 *  @version @$Id:$
 */
class cudaParticleRotation : virtual public cudaParticleBase {
protected:

  float4 *w;  //!< pointer to the array on GPU for angular velocity omega_i[N] omega_j[N] omega_k[N]

  float4 *L;  //!< pointer to the array on GPU for angular momentum

  float4 *T;  //!< pointer to the array on GPU for Torque T_i[N] T_j[N] T_k[N]

  real *m0inv;  //!< pointer to the array on GPU for 1/(2/5r^2m) [N]

public:
  cudaParticleRotation() :
    w(NULL), m0inv(NULL), L(NULL), T(NULL)
  {};

  ~cudaParticleRotation();

  /** construct arrays on GPU for n particles
   *
   * @param n number of particles
   */
  void setup(int n);

  /** do time evolution by 1-dimentional grids and blocks by Euler method
   *
   * @param dt  Delta_t
   */
  void TimeEvolution(real dt);

  void setInertia(real r0_all);
  void setInertia(const std::vector<real> &_r0);

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
void cudaParticleRotation::save(Archive& ar, const uint32_t version) const {
  std::vector<real> _tmp(N*4);
  const size_t size3N = sizeof(float4) * N;

  cudaMemcpy(&(_tmp[0]), w, size3N, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("angularVelocityW", _tmp);

  cudaMemcpy(&(_tmp[0]), L, size3N, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("angularMomentumL", _tmp);

  cudaMemcpy(&(_tmp[0]), T, size3N, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("torqueT", _tmp);

  std::vector<real> _tmp1(N);
  const size_t sizeN = sizeof(real) * N;
  cudaMemcpy(&(_tmp1[0]), m0inv, sizeN, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("m0InverseM0inv", _tmp1);

  cudaError_t t = cudaGetLastError();
  if (t!=0)
    std::cerr << "cudaParticleRotation::save: "
              << cudaGetErrorString(t) << std::endl;
}

template<class Archive>
void cudaParticleRotation::load(Archive& ar, const uint32_t version) {
  if (resizeInLoad) {
    if (w!=NULL)      cudaFree(w);
    if (L!=NULL)      cudaFree(L);
    if (T!=NULL)      cudaFree(T);
    if (m0inv!=NULL)  cudaFree(m0inv);

    cudaMalloc((void **)&w, sizeof(float4)*N);
    cudaMalloc((void **)&L, sizeof(float4)*N);
    cudaMalloc((void **)&T, sizeof(float4)*N);
    cudaMalloc((void **)&m0inv, sizeof(real)*N);
  }

  std::vector<real> _tmp(N*4);
  const size_t size3N = sizeof(float4) * N;

  ar >> boost::serialization::make_nvp("angularVelocityW", _tmp);
  cudaMemcpy(w, &(_tmp[0]), size3N, cudaMemcpyHostToDevice);

  ar >> boost::serialization::make_nvp("angularMomentumL", _tmp);
  cudaMemcpy(L, &(_tmp[0]), size3N, cudaMemcpyHostToDevice);

  ar >> boost::serialization::make_nvp("torqueT", _tmp);
  cudaMemcpy(T, &(_tmp[0]), size3N, cudaMemcpyHostToDevice);

  std::vector<real> _tmp1(N);
  const size_t sizeN = sizeof(real) * N;
  ar >> boost::serialization::make_nvp("m0InverseM0inv", _tmp1);
  cudaMemcpy(m0inv, &(_tmp1[0]), sizeN, cudaMemcpyHostToDevice);

  cudaError_t t = cudaGetLastError();
  if (t!=0)
    std::cerr << "cudaParticleRotation::load: "
              << cudaGetErrorString(t) << std::endl;
}
#endif /* CUDAPARTICLEROTATION */
