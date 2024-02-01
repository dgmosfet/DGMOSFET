#ifndef _ADIMPARAMS_H
#define _ADIMPARAMS_H

#include <iostream>

using namespace std;

#include "physdevice.h"
#include "physconsts.h"
#include "rescalingparams.h"

/*
 begin section: 'AdimPar02'
 written: 20230515
 last modified: 20230515
 author: Francesco Vecil

 description:
    This class will contain the dimensionless parameters.
*/
class adimParams
{
public:
  adimParams(const physDevice *pd, const physConsts *pc, const rescalingParams *rp); /*
											    Constructor of the class receives as parameters 
											    the physical description of the device, the physical constants
											    and the rescaling parameters.
											  */

  void printParams();
  
  /*
    The list of getters follows.
   */
  __host__ __device__ inline double    get_cs1() const { return __cs1;    }
  __host__ __device__ inline double    get_eta() const { return __eta;    }
  __host__ __device__ inline double     get_cp() const { return __cp;     }
  __host__ __device__ inline double     get_cg() const { return __cg;     }
  __host__ __device__ inline double    get_csr() const { return __csr;    }
  
private:
  double __cs1; //      ((hbar*hbar)/(_ZWIDTH*_ZWIDTH*me*_epsstar))
  double __eta; //      (_ZWIDTH/_XLENGTH)
  double __cp;  //      ((poscharge*_nstar*_ZWIDTH*_ZWIDTH)/(_potstar*eps0))
  double __cg;  //      for Gummel method
  double __csr; //      for Scattering Roughness
};
/*
 end section: 'AdimPar02'
*/

#endif
