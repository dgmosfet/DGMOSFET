#ifndef _RESCALINGPARAMS_H
#define _RESCALINGPARAMS_H

#include <iostream>

using namespace std;

#include "physdevice.h"
#include "physconsts.h"

/*
 begin section: 'AdimPar01'
 written: 20230515
 last modified: 20230515
 author: Francesco Vecil

 description:
    This class will contain the rescaling parameters.
*/
class rescalingParams
{
public:
  rescalingParams(const physDevice *pd, const physConsts *pc); /*
								 Constructor of the class receives as parameters 
								 the physical description of the device and the physical constants.
							       */

  void printParams();
  
  /*
    The list of getters follows.
   */
  __host__ __device__ inline double    get_xstar() const { return __xstar;    }
  __host__ __device__ inline double    get_zstar() const { return __zstar;    }
  __host__ __device__ inline double    get_tstar() const { return __tstar;    }
  __host__ __device__ inline double  get_potstar() const { return __potstar;  }
  __host__ __device__ inline double    get_kstar() const { return __kstar;    }
  __host__ __device__ inline double  get_epsstar() const { return __epsstar;  }
  __host__ __device__ inline double    get_vstar() const { return __vstar;    }
  __host__ __device__ inline double get_maxwstar() const { return __maxwstar; }
  __host__ __device__ inline double  get_chistar() const { return __chistar;  }
  __host__ __device__ inline double    get_nstar() const { return __nstar;    }
  __host__ __device__ inline double  get_rhostar() const { return __rhostar;  }
  __host__ __device__ inline double    get_jstar() const { return __jstar;    }
  __host__ __device__ inline double   get_scstar() const { return __scstar;   }
  __host__ __device__ inline double    get_astar() const { return __astar;    }
  
private:
  double __xstar;
  double __zstar;
  double __tstar;
  double __potstar;
  double __kstar;
  double __epsstar;
  double __vstar;
  double __maxwstar;
  double __chistar;
  double __nstar;
  double __rhostar;
  double __jstar;
  double __scstar;
  double __astar;
};
/*
 end section: 'AdimPar01'
*/

#endif
