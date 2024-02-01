#ifndef _MESHPARAMS_H
#define _MESHPARAMS_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#define     _ERR_MESHPAR_NOTOPEN                          32 /*
							       While constructing the physDevice class, the configuration file
							       coudl not be opened.
							     */


#define     _ERR_MESHPAR_TOKNOSEP                         33 /*
							       While reading the physDevice configuration file, the separator token '='
							       has not been found for a specific token. The syntax is therefore incorrect.
							     */

#define     _ERR_MESHPAR_EMPTYVAL                         34 /*
							       While reading the physDevice configuration file, the value
							       field for the given token is empty.
							     */
/*
 begin section : 'meshPar01'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 description:
    The following class contains the number of points for each dimension of the problem.

 Technical notes
 ===============
*/
class meshParams
{
public:
  meshParams();

  void readConfFile(const string filename); /*
					      This function is meant to read a configuration file,
					      written with a certain defined syntax, and hence to modify
					      the default values for the different fields.
					    */

  void printDataFields(); /*
			    This method just prints all the data fields of the class.
			   */

  void readInline(int _argc, char **_argv); /*
					      This function reads inline information and modifies tha
					      data structures accordingly.
					    */

  __host__ __device__ inline int     get_NSBN()              const { return __NSBN;              }
  __host__ __device__ inline int     get_NX()                const { return __NX;                }
  __host__ __device__ inline int     get_NZ()                const { return __NZ;                }
  __host__ __device__ inline int     get_NW()                const { return __NW;                }
  __host__ __device__ inline int     get_BARN()              const { return __BARN;              }
  __host__ __device__ inline int     get_NPHI()              const { return __NPHI;              }
  __host__ __device__ inline int     get_NZEXT()             const { return __NZEXT;             }
  __host__ __device__ inline double  get_CFL()               const { return __CFL;               }
  __host__ __device__ inline int     get_STEPMAX()           const { return __STEPMAX;           }
  __host__ __device__ inline double  get_TMAX()              const { return __TMAX;              }
  __host__ __device__ inline int     get_FREQ()              const { return __FREQ;              }
  
public:
  int                __NSBN;   /* 
				  number of subbands taken into account                            
				  default value is set to 6
			       */
  
  int                  __NX;   /* 
				  number of discretization points along the X-dimension
				  it is related to the size of source - channel - drain
				  because it must capture the interfaces between the different zones
				  with  5 nm - 10 nm - 5 nm device, therefore, 
				  the number of points must be    
				  N*4 + 1
				  default value is set to 65
			       */

  int                  __NZ;   /* 
				  number of discretization points along the Z-dimension            
				  it is related to the size of oxide - silicon - oxide
				  because it must capture the interfaces between the different zones
				  with a 1 nm - 6 nm - 8 nm device, therefore,
				  the number of points must be
				  N*8 + 1
				  default value is set to 65
			       */

  int                  __NW;   /* 
				  number of discretization points along the W-dimension            
				  default value is set to 300
			       */

  int                __BARN;   /* 
				  extension of the W-domain                            
			          the larger this value, the larger the value of wmax  
			          which represents the border of the W-domain;
			          in general, the higher the applied voltage, the
			          larger this value needs to be to capture the
			          electrons acceleration     
				  default value is set to 15
			       */

  int                __NPHI;   /* 
				  number of discretization points along the PHI-dimension     
				  note: it must be even     
				  default value is set to 48
			       */

  int               __NZEXT;   /* 
				  parameter for surface roughness scattering (specs to be written) 
			       */

  double              __CFL;   /* 
				  Courant Friedrichs Lewy condition                                
				  default value is set to 0.6
			       */

  int             __STEPMAX;   /* 
				  maximum number of simulation time steps performed                
				  default value is set to 99999999
			       */

  double             __TMAX;   /* 
				  simulation time
				  default value is set to 5 picoseconds
			       */

  int                __FREQ;   /* 
				  saving-to-file rate in term of time steps                        
				  default value is set to 500
			       */
};
/*
 end section: 'meshPar01'
*/

#endif
