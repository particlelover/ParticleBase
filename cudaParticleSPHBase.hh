#if !defined(CUDAPARTICLESPH)
#define CUDAPARTICLESPH

#include "cudaParticleLF.hh"
#include "cudaSelectedBlock.hh"
#include "kernelfuncs.h"
#include <cublas_v2.h>

/** successor class for the SPH particle
 *
 * num[], rho[] and others for the number density, mass density and its reciprocal
 * are placed on one float4 array, as .x, .y and .z = num, rho and rhoinv
 *
 *  @author  Shun Suzuki
 *  @date    2012 Feb.
 *  @version @$Id:$
 */
class cudaParticleSPHBase : virtual public cudaSelectedBlock, public cudaParticleLF {
protected:
  float4 *num;  //!< float4 array on GPU with size N; used for num, rho as .x, .y
  real *rhoinv; //!< reciprocal of mass density 1.0/rho (array on GPU with size N)

  real *m;  //!< 1D vector with a size N for the mass

  //real *w2D;  //!< kernel space for all i-j pair by 2D matrix (column-major; index is i+J*N) (array on GPU size N*N)

  //real *dW2D; //!< kernel space of dW/dr for all i-j pair by 2D matrix (column-major; index is i+J*N) (array on GPU size N*N)

  real h;   //!< SPH kernel radius

  real w0;  //!< (105)/(16M_PI h^7) for 3D Lucy

  cublasHandle_t hdl; //!< handlar for the CUBLAS operations
public:
  cudaParticleSPHBase() : num(NULL), rhoinv(NULL), m(NULL), hdl(NULL) {};

  ~cudaParticleSPHBase();

  /** construct arrays on GPU and inner TMP array for n particles
   *
   * @param n number of particles
   */
  void setup(int n);

  /** calculates kernel's and its gradient field from all i-j pair
   *  and store them to w2D and dW2D array with a size N*N
   *
   */
  void calcKernels(void);

  /** calculates number and mass density fields (n and rho)
   *
   */
  void calcDensity(void);

  /** set properties for SPH calculation
   *
   * @param _h  radius of the SPH particles
   */
  void setSPHProperties(real _h);

  /** set m = 1/(1/mass) for the calculation of temperature
   *
   */
  void setM(void);

  void rollback(real dt);
  void rollback2(real dt);

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
void cudaParticleSPHBase::save(Archive& ar, const uint32_t version) const {
  ar << boost::serialization::make_nvp("cudaParticleLF",
        boost::serialization::base_object<cudaParticleLF>(*this));

  ar << boost::serialization::make_nvp("cudaSelectedBlock",
        boost::serialization::base_object<cudaSelectedBlock>(*this));

  ar << boost::serialization::make_nvp("h",  h);
  ar << boost::serialization::make_nvp("w0", w0);
}

template<class Archive>
void cudaParticleSPHBase::load(Archive& ar, const uint32_t version) {
  ar >> boost::serialization::make_nvp("cudaParticleLF",
        boost::serialization::base_object<cudaParticleLF>(*this));

  ar >> boost::serialization::make_nvp("cudaSelectedBlock",
        boost::serialization::base_object<cudaSelectedBlock>(*this));

  ar >> boost::serialization::make_nvp("h", h);
  ar >> boost::serialization::make_nvp("w0", w0);

  if (resizeInLoad) {
    if (num!=NULL) cudaFree(num);
    if (m!=NULL)   cudaFree(m);
    if (rhoinv!=NULL) cudaFree(rhoinv);
    //if (w2D!=NULL) cudaFree(w2D);
    //if (dW2D!=NULL) cudaFree(dW2D);

    cudaMalloc((void **)&num, sizeof(float4)*N);
    //setArray<<<4, 256>>>(one, 1.0, N);
    cudaMalloc((void **)&m, sizeof(real)*N);
    cudaMalloc((void **)&rhoinv, sizeof(real)*N);
    //cudaMalloc((void **)&w2D, sizeof(real)*N*N);
    //cudaMalloc((void **)&dW2D, sizeof(real)*N*N);
  }

  if (hdl==NULL) cublasCreate(&hdl);
  setM();
}

#endif /* CUDAPARTICLESPH */
