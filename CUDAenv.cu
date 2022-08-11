#include "CUDAenv.hh"
#include <cuda_runtime_api.h>


int numberOfCores(int major, int minor) {
  int N = 0;
  if (major == 1) {
    if ((minor == 0) || (minor == 1) || (minor == 2) || (minor == 3)) N = 8;
  } else if (major == 2) {
    if (minor == 0) N = 32;
    else if (minor == 1) N = 48;
  } else if (major == 3) {
    N = 192;
  } else if (major == 5) {
    N = 128;
  } else if (major == 5) {
    if (minor == 0) N = 64;
    else N = 128;
  } else {
    std::cerr << "unsupported version:" << major << "." << minor << std::endl;
  }
    /*
    // modefied from NVIDIA_GPU_Computing_SDK/common/inc/helper_cuda_drvapi.h
        sSMtoCores nGpuArchCoresPerSM[] = 
        { { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
          { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
          { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
          { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
          { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
          { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
          {   -1, -1 }
        };
    */

  return N;
}
