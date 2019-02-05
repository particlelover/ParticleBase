#include "cudaFrame.hh"

#include <iostream>

cudaFrame::cudaFrame() : withInfo(true)
{
  std::cerr << "particles object generated at: ";
  starttime = nowtime(std::cerr);

  pthread_mutex_init(&mutTMP, NULL);
}

cudaFrame::~cudaFrame()
{
  pthread_mutex_lock(&mutTMP);

  std::cerr << "particles object destroyed at: ";
  time_t T2 = nowtime(std::cerr) - starttime;
  std::cerr << "life time of this object: " << T2 << " sec" << std::endl;
}

void cudaFrame::ErrorInfo(const std::string &s) const
{
#if defined(DEBUG)
  cudaThreadSynchronize();
#endif

  cudaError_t t = cudaGetLastError();
  if (t != 0)
  {
    std::cerr
        << s << ": "
        << cudaGetErrorString(t) << std::endl;
    exit(0);
  }
}

void cudaFrame::putTMP(std::ostream &o)
{
  oTMP = &o;
  pthread_mutex_lock(&(mutTMP));

  //  pthread_t thID;
  pthread_create(&thID, NULL, putTMPinThread, (void *)this);

  //  putTMPinThread(o);
  /*
  for (int i=0;i<N;++i)
    o << i << "   "
      << TMP[i] << ", " << TMP[i+N] << ", " << TMP[i+N*2] << std::endl;
*/
}

void *putTMPinThread(void *arg)
{
  class cudaFrame *a = (class cudaFrame *)arg;
  std::cerr << "output in TMPthread: " << a->timestep << " elapsed ";
  time_t T = time(NULL);
  std::cerr << T - a->starttime << " sec" << std::endl;

  pthread_detach(a->thID);
  //  uint32_t N = a->N;
  const std::valarray<real> &TMP = a->TMP;
  //  real *TMP = &(a->TMP[0]);
  const uint32_t N = a->TMP2.size();

  const std::valarray<unsigned short> &TMP2 = a->TMP2;
  real const *cell = a->cell;

  *(a->oTMP) << "ITEM: TIMESTEP" << std::endl
             << a->timestep << std::endl
             << "ITEM: NUMBER OF ATOMS" << std::endl
             << N << std::endl
             << "ITEM: BOX BOUNDS" << std::endl
             << cell[0] << " " << cell[1] << std::endl
             << cell[2] << " " << cell[3] << std::endl
             << cell[4] << " " << cell[5] << std::endl
             << "ITEM: ATOMS id type x y z" << std::endl;

  for (uint32_t i = 0; i < N; ++i)
  {
    *(a->oTMP) << i << "  "
               << TMP2[i] << " "
               << TMP[i] << " " << TMP[i + N] << " " << TMP[i + N * 2];
    *(a->oTMP) << a->additionalOutput(i);
    *(a->oTMP) << std::endl;
  }
  pthread_mutex_unlock(&(a->mutTMP));

  return NULL;
}

void cudaFrame::setCell(real *_cell)
{
  for (int i = 0; i < 6; ++i)
    cell[i] = _cell[i];
  for (int i = 0; i < 3; ++i)
  {
    cell[i + 6] = (cell[i * 2 + 1] - cell[i * 2]) / 2;
    cell[i + 6] *= cell[i + 6];
  }
}

time_t cudaFrame::nowtime(std::ostream &o)
{
  time_t T = time(NULL);
  o << ctime(&T);
  return T;
}
