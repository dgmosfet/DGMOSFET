#ifndef _DISCRDIM_H
#define _DISCRDIM_H

#include <iostream>

using namespace std;


/*
 begin section: 'Meshes_01'
 written: 20230515
 last modified: 20230516
 author: Francesco Vecil

 description:
    The objects of this class will be the meshes in the different dimensions.

 notes:
    (i) 2023/05/16 : an enumerative type has been added to take into account the
                     peculiarities of the different meshes, in the sense that
		     w is half-shifted to avoid zero, and
		     phi does not take into account the last point

    (ii) 2023/05/16 : I have modifies the class, so that 'mesh' just returns the
                      value, but no static allocation of memory is needed to store
		      the values of the different discretization points.
		      Formerly, there was a double* _mesh field.
*/
enum meshType { std_mesh = 0, W_mesh = 1, PHI_mesh = 2 };
inline std::ostream& operator<<( std::ostream& os, const meshType& mt )
{
  switch( mt )
    {
    case std_mesh      : os << "std_mesh"; break;
    case W_mesh        : os << "W_mesh"; break;
    case PHI_mesh      : os << "PHI_mesh"; break;
    }
  return os;
}

class discrDim
{
public:
  /*
    Constructor and copy constructor of the class.
  */
  discrDim(const int par_N = 0, const double par_min = 0, const double par_max = 0, const meshType = std_mesh );
  discrDim(const discrDim &dd2);

  void printParams();
  
  /*
    The list of getters follows.
   */
  __host__ __device__ inline int       get_N()           const { return dd_N;         }
  __host__ __device__ inline double    get_delta()       const { return dd_delta;     }
  __host__ __device__ inline double    get_delta_m1()    const { return dd_delta_m1;  }
  __host__ __device__ inline double    get_delta_m2()    const { return dd_delta_m2;  }
  __host__ __device__ inline double    get_min()         const { return dd_min;       }
  __host__ __device__ inline double    get_max()         const { return dd_max;       }
  __host__ __device__ inline meshType  get_mt()          const { return dd_mt;        }
  __host__ __device__ inline double    mesh(const int i) const
  {
    switch( dd_mt )
      {
      case W_mesh :
	return dd_min + (static_cast<double>(i)+0.5)*dd_delta;
	break;
	
      default :
	return dd_min + static_cast<double>(i)*dd_delta;
	break;
      }
  }
  
private:
  int dd_N;
  double dd_delta;
  double dd_delta_m1;
  double dd_delta_m2;
  double dd_min;
  double dd_max;
  meshType dd_mt;
};
/*
 end section: 'Meshes_01'
*/

#endif
