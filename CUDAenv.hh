#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "ExchangeMode.hh"

/** information about CUDA/GPU environment
 *
 * obtain and store CUDA/GPU enviroment informations
 *
 *
 *  @author  Shun Suzuki
 *  @date    2010 Nov.
 *  @version @$Id:$
 */
template <typename T>
class CUDAenv {
public:
  CUDAenv();
  ~CUDAenv();
  void display(std::ostream &o) const;  //!< display all info
  void setGPU(int i); //!< select GPU device

  std::string getName(void) const { return name[currentGPU]; };
  double getCapability(void) const { return capability[currentGPU]; };
  uint32_t getNumMP(void) const { return numMP[currentGPU]; };
  uint32_t getNumCpMP(void) const { return numCpMP[currentGPU]; };
  uint32_t getMemSize(void) const { return globalMemSize[currentGPU]; };
  uint32_t getNumTh(void) const { return numTh[currentGPU]; };
  uint32_t getMaxGrid(void) const { return maxGrid[currentGPU]; };
  uint32_t getShMem(void) const { return shMem[currentGPU]; };

  void setThnum(T &p) {
    int THnum2D;
    THnum2D = std::min(getNumCpMP() / ((getCapability() < 3) ? 2 : 4),
                       static_cast<uint32_t>(sqrt(getNumTh())));
    p.MPnum = getNumMP();
    p.THnum1D = std::min(1024u, getNumTh());
    p.THnum1DX = getNumCpMP() * 4;
    p.THnum2D = THnum2D;
    p.threadsMax = getNumTh();
    p.maxGrid = getMaxGrid();
    std::cerr << "CUDA thread parameters: MPnum: " << p.MPnum
              << ", THnum1D: " << p.THnum1D
              << ", THnum1DX: " << p.THnum1DX
              << ", THnum2D: " << THnum2D << std::endl;
  }

  int nDevices(void) { return ndev; }

  T& operator[](int i) { return *P[i]; }

  void setup(void);

  void exchangeForce(void);
  void exchangeAccelerations(void);

  /** exchange Force, Torque with optimizing its host/device exchanges,
   *  in most case only moving particles of each devices are exchanged
   *
   * @param typeID  0: force, 1: acceleration, 2: torque, 3: coordination number (DEM), 4: num/rho/mu (SPH_NS)
   */
  void exchangeForceSelected(const ExchangeMode typeID=ExchangeMode::force);

  void readSerialization(std::string filename);
  void writeSerialization(std::string filename) const;

protected:
  int ndev; //!< number of GPU devices
  int currentGPU; //!< GPU ID for currently use

  std::vector<std::vector<bool>> gpudirectEnable;
  bool enableAllGPUdirect;

private:
  std::vector<std::string> name;  //!< name of GPU
  std::vector<double> capability; //!< cuda compute capability
  std::vector<uint32_t> numMP;    //!< number of multiProcessor(MP)
  std::vector<uint32_t> numCpMP;  //!< number of Cores per MP
  std::vector<uint32_t> globalMemSize;  //!< size of global memory
  std::vector<uint32_t> numTh;    //!< number of Threads per block
  std::vector<uint32_t> maxGrid;  //!< Max Dimension size of grid (X)
  std::vector<uint32_t> shMem;  //!< size of the shared memory

  std::vector<T *> P;
};

int numberOfCores(int, int);

template <typename T>
CUDAenv<T>::CUDAenv() {
  std::cerr << "checking GPU devices" << std::endl;
  cudaError_t t = cudaGetDeviceCount(&ndev);
  std::cerr << "GPU status: "
            << cudaGetErrorString(t) << std::endl;
  if (t!=0) exit(t);

  currentGPU = 0;

  std::cerr << "found " << ndev << " GPU" << std::endl;
  struct cudaDeviceProp prop;
  for (uint32_t i=0;i<ndev;++i) {
    std::cerr << "  Device " << i << ": ";
    t = cudaGetDeviceProperties(&prop, i); // cudaDeviceGetAttribute()?
    if (t!=0) std::cerr << cudaGetErrorString(t) << std::endl;
    std::cerr << prop.name << std::endl;

    name.push_back(prop.name);
    capability.push_back(prop.major + static_cast<double>(prop.minor)*0.1);
    numMP.push_back(prop.multiProcessorCount);
    numCpMP.push_back(numberOfCores(prop.major, prop.minor));
    globalMemSize.push_back(prop.totalGlobalMem);
    numTh.push_back(prop.maxThreadsPerBlock);
    maxGrid.push_back(prop.maxGridSize[0]);
    shMem.push_back(prop.sharedMemPerBlock);

/*
  std::cout
  << "  max blocks  per grid:  " << prop.maxGridSize[0] << "x" << prop.maxGridSize[1] << std::endl
  << "  amount of shared memory per block: " << prop.sharedMemPerBlock
  << std::endl;
*/
  }

  enableAllGPUdirect = true;
  if (ndev>1) {
    gpudirectEnable.resize(ndev);
    for (int i=0;i<ndev;++i) {
      gpudirectEnable[i].resize(ndev);
      for (int j=0;j<ndev;++j) {
        gpudirectEnable[i][j] = false;
        if (i!=j) {
          int A;
          cudaDeviceCanAccessPeer(&A, i, j);
          std::cerr << "GPUDirect: peer access from " << i << " to " << j;
          std::cerr << ((A==1) ? " enabled" : " disabled");
          std::cerr << std::endl;
          if (A==1) {
            cudaSetDevice(i);
            cudaDeviceEnablePeerAccess(j, 0);
            gpudirectEnable[i][j] = true;
          } else {
            enableAllGPUdirect = false;
          }
        }
      }
    }
    std::cerr << "All GPUdirect enable: ";
    std::cerr << ((enableAllGPUdirect) ? " true" : " false");
    std::cerr << std::endl;
  }
  display(std::cerr);
}

template <typename T>
CUDAenv<T>::~CUDAenv() {
  for (int i=0;i<P.size();++i) {
    delete P[i];
  }
}

template <typename T>
void CUDAenv<T>::setup(void) {
  for (int i=0;i<ndev;++i) {
    T *p = new T;
    setGPU(i);
    P.push_back(p);

    setThnum(*p);
  }
}

template <typename T>
void CUDAenv<T>::display(std::ostream &o) const {
  for (uint32_t i=0;i<ndev;++i) {
    o
      << std::setw(5)
      << std::setprecision(5);
    o
      << name[i]
      << " (compute capability "
      << capability[i] << ")" << std::endl
      << "  number of multiprocessor: " << numMP[i] << std::endl
      << "  number of cores / MP: " << numCpMP[i] << std::endl
      << "  global memory size: " << static_cast<double>(globalMemSize[i]) /1024 /1024 /1024 << " [GB]" << std::endl
      << "  max threads per block: " << numTh[i] << std::endl
      << "  max block per grid: " << maxGrid[i] << std::endl
      << "  shared memory size: " << static_cast<double>(shMem[i]) /1024  << " [KB]" << std::endl
      ;
  }
}

template <typename T>
void CUDAenv<T>::setGPU(int i) {
  currentGPU = i;
  cudaSetDevice(i);
}

template <typename T>
void CUDAenv<T>::exchangeForce(void) {
  if (ndev>1) {
    if (!enableAllGPUdirect) {
#pragma omp for
      for (int i=0;i<ndev;++i) {
        //setGPU(i);
        P[i]->getForce();
      }
    }
#pragma omp for
    for (int i=0;i<ndev;++i) {
      //setGPU(i);
      for (int k=0;k<ndev;++k)
        if (i!=k) {
          P[i]->importForce(*P[k], gpudirectEnable[i][k], i, k);
          //std::cerr << "importing " << k << " to " << i << std::endl;
        }
    }
  }
}
template <typename T>
void CUDAenv<T>::exchangeAccelerations(void) {
  if (ndev>1) {
    if (!enableAllGPUdirect) {
#pragma omp for
      for (int i=0;i<ndev;++i) {
        //setGPU(i);
        P[i]->getAcceleration();
      }
    }
#pragma omp for
    for (int i=0;i<ndev;++i) {
      //setGPU(i);
      for (int k=0;k<ndev;++k)
        if (i!=k) {
          P[i]->importAcceleration(*P[k], gpudirectEnable[i][k], i, k);
          //std::cerr << "importing " << k << " to " << i << std::endl;
        }
    }
  }
}

template <typename T>
void CUDAenv<T>::exchangeForceSelected(const ExchangeMode typeID) {
  if (ndev>1) {
    if (!enableAllGPUdirect) {
#pragma omp for
      for (int i=0;i<ndev;++i) {
        // each particles objects obtains fragment of moving particles
        P[i]->getForceSelected(typeID);
      }
    }
#pragma omp for
    for (int i=0;i<ndev;++i) {
      for (int _k=1;_k<ndev;++_k) {
        const int k = (_k + i) % ndev;
        // push particles obtained from another board
        P[i]->importForceSelected(*P[k], typeID, gpudirectEnable[i][k], i, k);
        //std::cerr << "importing " << k << " to " << i << std::endl;
      }
    }
  }
}

template <typename T>
void CUDAenv<T>::readSerialization(std::string filename) {

  for (int i=0;i<ndev;++i) {
    setGPU(i);

    std::ifstream ifs(filename.c_str());
    boost::archive::binary_iarchive ia(ifs);
    ia >> boost::serialization::make_nvp("cudaParticles", *P[i]);
    ifs.close();
  }
}
template <typename T>
void CUDAenv<T>::writeSerialization(std::string filename) const {
  std::ofstream ofs(filename.c_str());
  boost::archive::binary_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("cudaParticles", *P[0]);
  ofs.close();
}
