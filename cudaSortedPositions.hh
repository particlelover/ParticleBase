#if !defined(CUDASORTEDPOSITIONS)
#define CUDASORTEDPOSITIONS

#include "cudaCutoffBlock.hh"

class cudaSortedPositions : virtual public cudaCutoffBlock
{
protected:
  real *r_s;                //!< positions sorted by bid
  unsigned short *typeID_s; //!< typeid sorted by bid

  template <class Archive>
  void load(Archive &ar, const uint32_t version);

public:
  cudaSortedPositions() : r_s(NULL), typeID_s(NULL){};

  ~cudaSortedPositions();

  void setupSortedPositions(void);

  /**
   *
   * calls cudaCutoffBlock::calcBlockID() and then copies r[] and typeid[]
   * with sorting by pid[]
   */
  void calcBlockID(void);
};

template <class Archive>
void cudaSortedPositions::load(Archive &ar, const uint32_t version)
{

  if (resizeInLoad)
  {
    if (typeID_s != NULL)
      cudaFree(typeID_s);
    if (r_s != NULL)
      cudaFree(r_s);
    cudaMalloc((void **)&r_s, sizeof(real) * 3 * N);
    cudaMalloc((void **)&typeID_s, sizeof(unsigned short int) * N);
  }
}
#endif
