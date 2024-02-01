#ifndef _DISCRMESHES_H
#define _DISCRMESHES_H

#include <iostream>

using namespace std;

#include "discrdim.h"

/*
 begin section: 'Meshes_04'
 written: 20230516
 last modified: 20230519
 author: Francesco Vecil

 description:
    This class is meant to contain the whole architecture of the discretization meshes.

 modification
    20230519 -- the class is re-designed with pointers to the discrDim objects,
                and the constructor takes pointers as arguments
*/
class discrMeshes
{
public:
  discrMeshes(const int par_NSBN, discrDim* par_X, discrDim* par_Z, discrDim* par_W, discrDim* par_PHI, discrDim* par_ZEXT);

public:
  __host__ __device__ inline int        get_NSBN()  const { return dm_NSBN; }
  __host__ __device__ inline discrDim*  get_X()     const { return dm_X;    }
  __host__ __device__ inline discrDim*  get_Z()     const { return dm_Z;    }
  __host__ __device__ inline discrDim*  get_W()     const { return dm_W;    }
  __host__ __device__ inline discrDim*  get_PHI()   const { return dm_PHI;  }
  __host__ __device__ inline discrDim*  get_ZEXT()  const { return dm_ZEXT; }

private:
  int dm_NSBN;
  discrDim* dm_X;
  discrDim* dm_Z;
  discrDim* dm_W;
  discrDim* dm_PHI;
  discrDim* dm_ZEXT;
};
/*
 end section: 'Meshes_04'
*/

#endif
