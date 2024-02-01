#ifndef _KERNELCONFIG_H
#define _KERNELCONFIG_H

#include <cuda.h>
#include <iostream>
#include <string>
using namespace std;

#define   NOSHMEM   0

inline int nblocks( const int npts, const int TPB )
{
  return ceil( (float)npts / TPB );
}
inline int sm_size_N( const int TPB, const int N )
{
  return ( (TPB+N-2)/N + 1 )*(N+6)*sizeof(double);
}

class kernelConfig
{
 public:
  kernelConfig(const int gridSize=0, const int blockSize=0, const int shmemSize=0, const cudaFuncCache cfc=cudaFuncCachePreferNone, const string label="no_label");
  void printDataFields();
  
  __host__ inline int get_gridSize () const     { return __gridSize;  }
  __host__ inline int get_blockSize() const     { return __blockSize; }
  __host__ inline int get_shmemSize() const     { return __shmemSize; }
  __host__ inline cudaFuncCache get_cfc() const { return __cfc;       }
  __host__ inline string get_label() const      { return __label;     }

 private:
  int __gridSize;
  int __blockSize;
  int __shmemSize;
  cudaFuncCache __cfc;
  string __label;
};

#endif
