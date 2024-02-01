#ifndef _KERNELS_H
#define _KERNELS_H

__host__ __device__ __forceinline__ void GPU_map_1D_to_5D( int index, int *i1, const int N1, int *i2, const int N2, int *i3, const int N3, int *i4, const int N4, int *i5, const int N5 )
{
  *i1    = index;
  index  = index/N5;
  *i5    = *i1-index*N5;
  *i1    = index;
  index  = index/N4;
  *i4    = *i1-index*N4;
  *i1    = index;
  index  = index/N3;
  *i3    = *i1-index*N3;
  *i1    = index;
  index  = index/N2;
  *i2   = *i1-index*N2;
  *i1    = index;
}

__host__ __device__ __forceinline__ void GPU_map_1D_to_4D( int index, int *i1, const int N1, int *i2, const int N2, int *i3, const int N3, int *i4, const int N4 )
{
  *i1    = index;
  index  = index/N4;
  *i4    = *i1-index*N4;
  *i1    = index;
  index  = index/N3;
  *i3    = *i1-index*N3;
  *i1    = index;
  index  = index/N2;
  *i2   = *i1-index*N2;
  *i1    = index;
}

__host__ __device__ __forceinline__ void GPU_map_1D_to_3D( int index, int *i1, const int N1, int *i2, const int N2, int *i3, const int N3 )
{
  *i1    = index;
  index  = index/N3;
  *i3    = *i1-index*N3;
  *i1    = index;
  index  = index/N2;
  *i2   = *i1-index*N2;
  *i1    = index;
}

__host__ __device__ __forceinline__ void GPU_map_1D_to_2D( int index, int *i1, const int N1, int *i2, const int N2 )
{
  *i1    = index;
  index  = index/N2;
  *i2    = *i1-index*N2;
  *i1    = index;
}

#endif
