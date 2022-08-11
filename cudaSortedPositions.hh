#if !defined(CUDASORTEDPOSITIONS)
#define CUDASORTEDPOSITIONS

#include "cudaCutoffBlock.hh"

class cudaSortedPositions: virtual public cudaCutoffBlock {
protected:
  float4 *r_s;  //!< positions sorted by bid
  unsigned short *typeID_s; //!< typeid sorted by bid

  void reset(void) {
    if (resizeInLoad) {
      if (typeID_s!=NULL) cudaFree(typeID_s);
      if (r_s!=NULL)      cudaFree(r_s);
      cudaMalloc((void **)&r_s, sizeof(float4)*N);
      cudaMalloc((void **)&typeID_s, sizeof(unsigned short int)*N);
    }
  };

public:
  cudaSortedPositions() :
    r_s(NULL), typeID_s(NULL)
  {};

  ~cudaSortedPositions();

  void setupSortedPositions(void);

  /**
   *
   * calls cudaCutoffBlock::calcBlockID() and then copies r[] and typeid[]
   * with sorting by pid[]
   */
  void calcBlockID(void);
};
#endif
