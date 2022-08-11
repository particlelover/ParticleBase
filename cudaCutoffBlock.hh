#if !defined(CUDACUTOFFBLOCK)
#define CUDACUTOFFBLOCK

#include <vector>
#include "cudaParticleBase.hh"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/split_member.hpp>
#include <numeric>

/** switchng parameter for the cutoff algorithm
 * 
 * many(0): one block can contains some particle
 * single(1): only one particle in one block
 */
enum class ParticleBlockType {
  many,
  single,
};

/** implementation of cut off block, which divide the simulation cell in each direction
 *
 *
 *
 *  @author  Shun Suzuki
 *  @date    2012 May.
 *  @version @$Id:$
 */
class cudaCutoffBlock : public virtual cudaParticleBase {
protected:
  std::vector<real> blocklen; //!< length of each cutoff block (size 3) + cellmin (size 3)

  std::vector<uint32_t> blocknum; //!< number of cutoff block in each direction (size 3)

  uint32_t totalNumBlock; //!< total number of cutoff blocks

  uint32_t *bid;  //!< pointer to the array on GPU for the block ID (size N), sorted with pid

  uint32_t *pid;  //!< pointer to the array on GPU for the particle ID (size N), particle pid[i] locates in bid[i]

  uint32_t *bindex; //!< pointer to the array on GPU for the block index (size totalNumBlock+1)
  //!< partial sum of number of particles in each block

  uint32_t *blockNeighbor;  //!< neighbor judgement for I-J block pair (size totalNumBlock*27); I*27+[0,27] has block ID for J

  real *tmp81N; //!< 3N array for 27blocks on GPU

  uint32_t myBlockNum;    //!< number of blocks for this GPU
  uint32_t myBlockOffset; //!< offset of the block ID for this GPU

  uint32_t THnum2D2;  //!< number of threads used for calcF_IJpairWithBlock()

  bool SingleParticleBlock = false;

  template<class T>
  void calcBlockRange(uint32_t blockNum, uint32_t N, uint32_t myID, T setPara) {
    uint32_t tmpN = blockNum / N;
    uint32_t remained = blockNum - N * tmpN;
    std::vector<uint32_t> V(N);
    for (uint32_t i=0;i<remained;++i) {
      V[i] = tmpN + 1;
    }
    for (uint32_t i=remained;i<N;++i) {
      V[i] = tmpN;
    }

    std::vector<uint32_t> V2(N+1);
    std::partial_sum(V.begin(), V.end(), &(V2[1]));

    setPara(V2[myID], V[myID]);
  }

public:

  cudaCutoffBlock() :
    blocklen(6), blocknum(3), bindex(NULL),
    tmp81N(NULL),
    bid(NULL), pid(NULL), blockNeighbor(NULL) {};


  ~cudaCutoffBlock();

  /** construct arrays on GPU for n particles
   *
   */
  void setupCutoffBlock(real rmax, bool periodic=true);

  /** calc block ID of each particle
   *
   */
  void calcBlockID(void);


  /** import Accelerations from other GPU
   *
   * import accelerations in TMP array in another particles object into
   * tmp3N array on GPU, and then merge them to *a
   */
  void importAcceleration(const cudaCutoffBlock &A,
    bool directAccess=false, int idMe=0, int idPeer=0);

  /** import Forces from other GPU
   *
   * import forces in TMP array in another particles object into
   * tmp3N array on GPU, and then merge them to *F
   */
  void importForce(const cudaCutoffBlock &A,
    bool directAccess=false, int idMe=0, int idPeer=0);

  uint32_t numBlocks(void) {
    return totalNumBlock;
  }

  void setBlockRange(uint32_t blockNum, uint32_t N, uint32_t myID);

  /** change the SingleParticleBlock flag in true/false
   * @param t many(0) SingleParticleBlock=false (WithBlock2 algorithm)
   * @param t single(1) SingleParticleBlock=true (WithBlock4 algorithm)
   */
  void switchBlockAlgorithm(const ParticleBlockType t);
private:
  friend class boost::serialization::access;
#if !defined(SWIG)
  BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif

  template<class Archive>
  void save(Archive& ar, const uint32_t version) const;

  template<class Archive>
  void load(Archive& ar, const uint32_t version);

  bool uninitialized = true;
};

template<class Archive>
void cudaCutoffBlock::save(Archive& ar, const uint32_t version) const {

  ar << boost::serialization::make_nvp("totalNumBlock", totalNumBlock);
  ar << boost::serialization::make_nvp("blocklen", blocklen);
  ar << boost::serialization::make_nvp("blocknum", blocknum);
  ar << boost::serialization::make_nvp("SingleParticleBlock", SingleParticleBlock);
  ar << boost::serialization::make_nvp("uninitialized", uninitialized);


  std::vector<uint32_t> _tmp1(N);
  const size_t sizeN1 = sizeof(uint32_t) * N;

  cudaMemcpy(&(_tmp1[0]), bid, sizeN1, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("bid", _tmp1);

  std::vector<uint32_t> _tmp2(N);
  const size_t sizeN2 = sizeof(uint32_t) * N;

  cudaMemcpy(&(_tmp2[0]), pid, sizeN2, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("pid", _tmp2);

  _tmp2.resize(totalNumBlock+1);
  const size_t sizeN22 = sizeof(uint32_t) * (totalNumBlock+1);
  cudaMemcpy(&(_tmp2[0]), bindex, sizeN22, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("bindex", _tmp2);


  const int J27 = (SingleParticleBlock) ? 125 : 27;
  std::vector<uint32_t> _tmp3(totalNumBlock*J27);
  const size_t sizeN3 = sizeof(uint32_t) * totalNumBlock*J27;

  cudaMemcpy(&(_tmp3[0]), blockNeighbor, sizeN3, cudaMemcpyDeviceToHost);
  ar << boost::serialization::make_nvp("blockNeighbor", _tmp3);


  cudaError_t t = cudaGetLastError();
  if (t!=0)
    std::cerr << "cudaCutoffBlock::save: "
              << cudaGetErrorString(t) << std::endl;
}

template<class Archive>
void cudaCutoffBlock::load(Archive& ar, const uint32_t version) {

  ar >> boost::serialization::make_nvp("totalNumBlock", totalNumBlock);
  ar >> boost::serialization::make_nvp("blocklen", blocklen);
  ar >> boost::serialization::make_nvp("blocknum", blocknum);
  ar >> boost::serialization::make_nvp("SingleParticleBlock", SingleParticleBlock);
  ar >> boost::serialization::make_nvp("uninitialized", uninitialized);

  const int J27 = (SingleParticleBlock) ? 125 : 27;
  if (resizeInLoad) {
    if (bid!=NULL)    cudaFree(bid);
    if (pid!=NULL)    cudaFree(pid);
    if (bindex!=NULL) cudaFree(bindex);
    if (blockNeighbor!=NULL) cudaFree(blockNeighbor);
    if (tmp81N!=NULL) cudaFree(tmp81N);
    cudaMalloc((void **)&bid, sizeof(uint32_t)*N);
    cudaMalloc((void **)&pid, sizeof(uint32_t)*N);
    cudaMalloc((void **)&bindex, sizeof(uint32_t)*(totalNumBlock+1));
    cudaMalloc((void **)&blockNeighbor, sizeof(uint32_t)*totalNumBlock*J27);
    cudaMalloc((void **)&tmp81N, sizeof(real)*3*N*27);
  }


  std::vector<uint32_t> _tmp1(N);
  const size_t sizeN1 = sizeof(uint32_t) * N;

  ar >> boost::serialization::make_nvp("bid", _tmp1);
  cudaMemcpy(bid, &(_tmp1[0]), sizeN1, cudaMemcpyHostToDevice);

  std::vector<uint32_t> _tmp2(N);
  const size_t sizeN2 = sizeof(uint32_t) * N;

  ar >> boost::serialization::make_nvp("pid", _tmp2);
  cudaMemcpy(pid, &(_tmp2[0]), sizeN2, cudaMemcpyHostToDevice);

  _tmp2.resize(totalNumBlock+1);
  const size_t sizeN22 = sizeof(uint32_t) * (totalNumBlock+1);
  ar >> boost::serialization::make_nvp("bindex", _tmp2);
  cudaMemcpy(bindex, &(_tmp2[0]), sizeN22, cudaMemcpyHostToDevice);


  std::vector<uint32_t> _tmp3(totalNumBlock*J27);
  const size_t sizeN3 = sizeof(uint32_t) * totalNumBlock*J27;

  ar >> boost::serialization::make_nvp("blockNeighbor", _tmp3);
  cudaMemcpy(blockNeighbor, &(_tmp3[0]), sizeN3, cudaMemcpyHostToDevice);


  myBlockOffset = 0;
  myBlockNum = totalNumBlock;

  THnum2D2 = THnum2D;
  {
    int _N = (real)(N) / totalNumBlock;
    while (THnum2D2<_N) THnum2D2 *= 2;
  }
  std::cerr << "THnum2D2: " << THnum2D2 << std::endl;

  cudaError_t t = cudaGetLastError();
  if (t!=0)
    std::cerr << "cudaParticleMD::load: "
              << cudaGetErrorString(t) << std::endl;
}

#endif
