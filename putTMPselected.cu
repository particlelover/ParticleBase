#include "putTMPselected.hh"
#include <fstream>

void putTMPselected::setupSelectedTMP(uint32_t s1, uint32_t s2, uint32_t u1, uint32_t u2)
{
  selected.resize(2);
  unselected.resize(2);

  selected[0] = s1;
  selected[1] = s2;
  unselected[0] = u1;
  unselected[1] = u2;
}

void putTMPselected::getSelectedTypeID(void)
{
  TMP2.resize(selected[1]);

  cudaMemcpy(&(TMP2[0]), &(typeID[selected[0]]), sizeof(unsigned short) * selected[1], cudaMemcpyDeviceToHost);
  if (withInfo)
    ErrorInfo("do getTypeID");
}

void putTMPselected::getSelectedPosition(void)
{
  const size_t sizeN = sizeof(real) * selected[1];

  pthread_mutex_lock(&mutTMP);
  cudaMemcpy(&(TMP[0]), &(r[selected[0]]), sizeN, cudaMemcpyDeviceToHost);
  cudaMemcpy(&(TMP[selected[1]]), &(r[selected[0] + N]), sizeN, cudaMemcpyDeviceToHost);
  cudaMemcpy(&(TMP[selected[1] * 2]), &(r[selected[0] + N * 2]), sizeN, cudaMemcpyDeviceToHost);
  pthread_mutex_unlock(&mutTMP);

  if (withInfo)
    ErrorInfo("do getPosition");
}

void putTMPselected::putUnSelected(const char *filename)
{
  std::valarray<unsigned short> _TMP2;
  _TMP2.resize(unselected[1]);
  cudaMemcpy(&(_TMP2[0]), &(typeID[unselected[0]]), sizeof(unsigned short) * unselected[1], cudaMemcpyDeviceToHost);
  cudaMemcpy(&(TMP[0]), &(r[unselected[0]]), sizeof(real) * unselected[1], cudaMemcpyDeviceToHost);
  cudaMemcpy(&(TMP[unselected[1]]), &(r[unselected[0] + N]), sizeof(real) * unselected[1], cudaMemcpyDeviceToHost);
  cudaMemcpy(&(TMP[unselected[1] * 2]), &(r[unselected[0] + N * 2]), sizeof(real) * unselected[1], cudaMemcpyDeviceToHost);

  std::ofstream ofil;
  ofil.open(filename);
  if (!ofil.is_open())
  {
    std::cerr << "cannot open " << filename << std::endl;
    return;
  }

  const uint32_t N = _TMP2.size();

  ofil << "ITEM: TIMESTEP" << std::endl
       << 0 << std::endl
       << "ITEM: NUMBER OF ATOMS" << std::endl
       << N << std::endl
       << "ITEM: BOX BOUNDS" << std::endl
       << cell[0] << " " << cell[1] << std::endl
       << cell[2] << " " << cell[3] << std::endl
       << cell[4] << " " << cell[5] << std::endl
       << "ITEM: ATOMS id type x y z" << std::endl;

  for (uint32_t i = 0; i < N; ++i)
  {
    ofil << i << "  "
         << _TMP2[i] << " "
         << TMP[i] << " " << TMP[i + N] << " " << TMP[i + N * 2]
         << std::endl;
  }

  ofil.close();
}
