#include "kernelconfig.h"

kernelConfig::kernelConfig(const int gridSize, const int blockSize, const int shmemSize, const cudaFuncCache cfc, const string label)
{
  __gridSize  = gridSize;
  __blockSize = blockSize;
  __shmemSize = shmemSize;
  __cfc       = cfc;
  __label     = label;
}

void kernelConfig::printDataFields()
{
  cerr << " ---------------------------- " << endl;
  cerr << "     __label = " << __label     << endl;
  cerr << "  __gridSize = " << __gridSize  << endl;
  cerr << " __blockSize = " << __blockSize << endl;
  cerr << " __shmemSize = " << __shmemSize << endl;
  cerr << "       __cfc = " << __cfc       << endl;
  cerr << " ---------------------------- " << endl;
}

