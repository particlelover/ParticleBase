#if !defined(CUDAFRAME)
#define CUDAFRAME

#include <string>
#include <iostream>
#include <valarray>
#include <pthread.h>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/nvp.hpp>
#include <time.h>

typedef REAL  real;

/** base class for the framework using CUDA
 *
 *  @author  Shun Suzuki
 *  @date    2012 Feb.
 *  @version @$Id:$
 */
class cudaFrame {
protected:
  bool withInfo;  //!< flag for information outputs

  std::valarray<real> TMP;  //!< tmp array for the transfer in between CPU to GPU

  std::valarray<unsigned short> TMP2; //!< tmp array for typeID

  real cell[9]; //!< pointer to the array for cell[6] to write to LAMMPS dump
  //!< xmin, xmax, ymin, ymax, zmin, zmax, (x/2)^2, (y/2)^2, (z/2)^2

public:
  int MPnum;  //!< grid size for CUDA kernel
  int THnum1D;  //!< block size for 1D CUDA kernel (fixed to 1024)
  int THnum1DX; //!< block size for 1D CUDA kernel (num of core x 4)
  int THnum2D;  //!< block size for 2D CUDA kernel (i-j pair) (num of core /2)
  uint32_t threadsMax;  //!< max number of threads of this GPU
  uint32_t maxGrid; //!< Max Dimension size of grid (X)

  uint32_t timestep;  //!< timestep to write to LAMMPS dump

  cudaFrame();

  ~cudaFrame();

  /** output error messages to std::cout if cudaGetLastError !=0
   *
   * @param s string to add to message
   */
  void ErrorInfo(const std::string &s) const;


  /** put TMP[] to ostream from splitted thread
   *
   * @param o stream such as std::cout or opened file stream
   */
  void putTMP(std::ostream &o=std::cout);

  /** set cell parameters
   *
   * @param _cell argument for the cell[6]
   */
  void setCell(real *_cell);

  /** define additional tag for each successor class
   *
   */
  virtual std::string additionalTag(void) const { return "";};

  /** define additional output for each successor class
   *
   * @param i particle ID to output
   * @return  a string to output
   */
  virtual std::string additionalOutput(uint32_t i) const { return "";};

  /** wait completion of running output process
   *
   */
  void waitPutTMP(void) {
    pthread_mutex_lock(&(mutTMP));
    pthread_mutex_unlock(&(mutTMP));
  }

private:
  friend void *putTMPinThread(void *arg); //!< putTMPinThread() is friend of mine to permit to read TMP[] directly from another thread

protected:
  pthread_mutex_t mutTMP; //!< mutex in between ::getPosition() and putTMPinThread()
  pthread_t thID;         //!< thread ID used in putTMP()
  std::ostream *oTMP;     //!< pointer to output stream obtained at putTMP() to share with putTMPinThread

private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const uint32_t version) {

    ar & boost::serialization::make_nvp("cellsize", cell);

    ar & boost::serialization::make_nvp("timestep", timestep);
  }

private:
  time_t starttime; //!< object generated time

  /** get time of now, and output this time to ostream
   *
   * @param o ostream for the output (default std::cout)
   * @return  time of now in sec from EPOC
   */
  time_t nowtime(std::ostream &o=std::cout);
};

void *putTMPinThread(void *arg);

#endif /* CUDAFRAME */
