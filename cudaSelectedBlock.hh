#if !defined(SELECTEDBLOCK)
#define SELECTEDBLOCK

#include "cudaCutoffBlock.hh"

/** select/extract the blocks for the cutting off the i--j pair search
 *
 */
class cudaSelectedBlock : public cudaCutoffBlock
{
protected:
  uint32_t *selectedBlock; //!< pointer to the array on GPU for the block ID (size N), sorted with pid

  long numSelected; //!< number of blocks which contains MOVABLE particles

#if defined(WithBlock2)
  long p1, p2;     //!< [start, end) pid, when ndev>1
  uint32_t p3, p4; //!< range of ID for MOVING particles
#endif
public:
  cudaSelectedBlock() : selectedBlock(NULL), myBlockSelected(0), myOffsetSelected(0)
#if defined(WithBlock2)
                        ,
                        p1(0), p2(N), p3(0), p4(UINT_MAX)
#endif
                                                 {};

  ~cudaSelectedBlock();

  void setupCutoffBlock(real rmax, bool periodic = true);

  /** inspect all blocks and marks blocks which has un-fixed(moving) particles.
   *  list of these blocks are also generated
   *
   */
  void selectBlocks(void);

  int myBlockSelected;  //!< number of blocks (which has movable particles) for my GPU
  int myOffsetSelected; //!< offset in the selectedBlock[] for my GPU

  long numSelectedBlocks(void) const { return numSelected; }

#if defined(WithBlock2)
  void checkPidRange(int devId = 0, int ndev = 1, uint32_t Mstart = 0, uint32_t Mend = UINT_MAX)
  {
    long _N = N / ndev;
    p1 = _N * devId;
    p2 = (devId + 1 == ndev) ? N : _N * (devId + 1);

    uint32_t _M = (Mend == UINT_MAX) ? N : Mend;
    _M -= Mstart;
    _M /= ndev;
    p3 = Mstart + _M * devId;
    p4 = (devId + 1 == ndev) ? ((Mend == UINT_MAX) ? N : Mend) : (_M * (devId + 1) + Mstart);

    std::cerr << "prange:" << p1 << ":" << p2 << ", " << p3 << ":" << p4 << std::endl;
  }
#endif

  /** get Forces with moving particle ID range (p3, p4)
   *
   * @param typeID	3: coordination number
   */
  void getForceSelected(const int typeID = 0);

  /** push a fragments of moving particles to another board
   *  (without summation)
   *
   * @param typeID	3: coordination number
   */
  void importForceSelected(const cudaSelectedBlock &A, const int typeID = 0,
                           bool directAccess = false, int idMe = 0, int idPeer = 0);

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
void cudaSelectedBlock::save(Archive &ar, const uint32_t version) const
{
  ar << boost::serialization::make_nvp("cudaCutoffBlock",
                                       boost::serialization::base_object<cudaCutoffBlock>(*this));
#if defined(WithBlock2)
  ar << boost::serialization::make_nvp("p1", p1);
  ar << boost::serialization::make_nvp("p2", p2);
  ar << boost::serialization::make_nvp("p3", p3);
  ar << boost::serialization::make_nvp("p4", p4);
#endif
}

template <class Archive>
void cudaSelectedBlock::load(Archive &ar, const uint32_t version)
{
  ar >> boost::serialization::make_nvp("cudaCutoffBlock",
                                       boost::serialization::base_object<cudaCutoffBlock>(*this));

#if defined(WithBlock2)
  ar >> boost::serialization::make_nvp("p1", p1);
  ar >> boost::serialization::make_nvp("p2", p2);
  ar >> boost::serialization::make_nvp("p3", p3);
  ar >> boost::serialization::make_nvp("p4", p4);
#endif

  if (resizeInLoad)
  {
    if (selectedBlock != NULL)
      cudaFree(selectedBlock);
    cudaMalloc((void **)&selectedBlock, sizeof(uint32_t) * totalNumBlock);
  }
}

#endif
