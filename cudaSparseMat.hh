#if !defined(CUDASPARSEMAT)
#define CUDASPARSEMAT
#include <boost/serialization/vector.hpp>
#include <boost/serialization/split_member.hpp>

typedef REAL real;

/** an implementation of sparse matrix by CSR format
 *
 */
class cudaSparseMat
{
public:
  uint32_t *rowPtr; //!< 1D array on GPU; part of CSR format; size N+1, start from 0
  uint32_t *colIdx; //!< 1D array on GPU; part of CSR format; size NNZ
  real *val;        //!< 1D array on GPU; part of CSR format; size NNZ

protected:
  uint32_t NNZ;  //!< number of non-zero element in sparse matrix
  uint32_t nmax; //!< real size of colIdx, val array

private:
  uint32_t _N; //!< number of particles/size of rowPtr are copied
  int TH;

public:
  cudaSparseMat() : NNZ(0), nmax(0), _N(0),
                    rowPtr(NULL), colIdx(NULL), val(NULL){};

  ~cudaSparseMat();

  /** makes rowPtr[] 
   *
   * @param N	number of row (rowPtr size is N+1)
   */
  void setup(uint32_t N, uint32_t threadsMax);

  /** makes rowPtr filled by the coordination number
   *
   * @return	number of non-zero elements
   */
  uint32_t makeRowPtr(void);

  /** resize colIdx[] and val[] by cudaMallocPitch()
   * is new nnz is larger than current size
   *
   * @param n	new size required to colIdx and val
   */
  void Resize(uint32_t n);

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
void cudaSparseMat::save(Archive &ar, const uint32_t version) const
{
  ar << boost::serialization::make_nvp("NNZ", NNZ);
  ar << boost::serialization::make_nvp("_N", _N);
  ar << boost::serialization::make_nvp("TH", TH);

  std::vector<uint32_t> _tmp(_N + 1);
  size_t sizeN = sizeof(uint32_t) * (_N + 1);
  cudaMemcpy(&(_tmp[0]), rowPtr, sizeN, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("rowPtr", _tmp);

  _tmp.resize(NNZ);
  sizeN = sizeof(uint32_t) * NNZ;
  cudaMemcpy(&(_tmp[0]), colIdx, sizeN, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("colIdx", _tmp);

  std::vector<real> _tmp2(NNZ);
  sizeN = sizeof(real) * NNZ;
  cudaMemcpy(&(_tmp2[0]), val, sizeN, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("val", _tmp2);
}

template <class Archive>
void cudaSparseMat::load(Archive &ar, const uint32_t version)
{
  const uint32_t __N = _N;
  uint32_t _NNZ = 0;
  ar >> boost::serialization::make_nvp("NNZ", _NNZ);
  ar >> boost::serialization::make_nvp("_N", _N);
  ar >> boost::serialization::make_nvp("TH", TH);

  Resize(_NNZ);

  std::vector<uint32_t> _tmp(_N + 1);
  size_t sizeN = sizeof(uint32_t) * (_N + 1);
  ar >> boost::serialization::make_nvp("rowPtr", _tmp);

  if ((__N != _N) && (rowPtr != NULL))
  {
    cudaFree(rowPtr);
    rowPtr = NULL;
  }
  if (rowPtr == NULL)
    cudaMalloc((void **)&rowPtr, sizeof(uint32_t) * (_N + 1));
  cudaMemcpy(rowPtr, &(_tmp[0]), sizeN, cudaMemcpyHostToDevice);

  _tmp.resize(NNZ);
  sizeN = sizeof(uint32_t) * NNZ;
  ar >> boost::serialization::make_nvp("colIdx", _tmp);
  cudaMemcpy(colIdx, &(_tmp[0]), sizeN, cudaMemcpyHostToDevice);

  std::vector<real> _tmp2(NNZ);
  sizeN = sizeof(real) * NNZ;
  ar >> boost::serialization::make_nvp("val", _tmp2);
  cudaMemcpy(val, &(_tmp2[0]), sizeN, cudaMemcpyHostToDevice);
}
#endif
