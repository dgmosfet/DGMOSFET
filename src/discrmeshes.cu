#include "discrmeshes.h"

/*
 begin section : 'Meshes_07'
 written       : 20230516
 last modified : 20230516
 author        : Francesco Vecil

 description:
    Constructor for the discretization meshes.
*/

discrMeshes::discrMeshes(const int par_NSBN, discrDim* par_X, discrDim* par_Z, discrDim* par_W, discrDim* par_PHI, discrDim* par_ZEXT)
{
  dm_NSBN = par_NSBN;
  dm_X    = par_X;
  dm_Z    = par_Z;
  dm_W    = par_W;
  dm_PHI  = par_PHI;
  dm_ZEXT = par_ZEXT;
}

/*
 end section: 'Meshes_07'
*/
