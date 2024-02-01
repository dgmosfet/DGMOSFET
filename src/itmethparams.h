#ifndef _ITMETHPARAMS_H
#define _ITMETHPARAMS_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#define     _ERR_ITMETHPAR_NOTOPEN                          38 /*
								 While constructing the physDevice class, the configuration file
								 coudl not be opened.
							       */


#define     _ERR_ITMETHPAR_TOKNOSEP                         39 /*
								 While reading the physDevice configuration file, the separator token '='
								 has not been found for a specific token. The syntax is therefore incorrect.
							       */

#define     _ERR_ITMETHPAR_EMPTYVAL                         40 /*
								 While reading the physDevice configuration file, the value
								 field for the given token is empty.
							       */

/*
 begin section: 'itmethPar01'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 description:
    The following class contains the tolerance parameters for the iterative methods.

 Technical notes
 ===============
*/
class itmethParams
{
public:
  itmethParams();

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

  __host__ __device__ inline double  get_TOL_NR_INITCOND()    const { return __TOL_NR_INITCOND;       }
  __host__ __device__ inline double  get_TOL_NR_THERMEQUIL()  const { return __TOL_NR_THERMEQUIL;     }
  __host__ __device__ inline double  get_TOL_NR_POTENTIAL()   const { return __TOL_NR_POTENTIAL;      }
  __host__ __device__ inline int  get_ITERMAX_NR_POTENTIAL()  const { return __ITERMAX_NR_POTENTIAL;  }
  __host__ __device__ inline double  get_TOL_EIGVALS()        const { return __TOL_EIGVALS;           }
  __host__ __device__ inline double  get_TOL_EIGVECS_IPIM()   const { return __TOL_EIGVECS_IPIM;      }
  __host__ __device__ inline double  get_TOL_LINSYS_SRJ()     const { return __TOL_LINSYS_SRJ;        }
  __host__ __device__ inline int  get_ITERMAX_LINSYS_SRJ()    const { return __ITERMAX_LINSYS_SRJ;    }
  
public:
  double  __TOL_NR_INITCOND;   /* 
				  parameter for Newton-Raphon for the INITCOND iterative problem
			       */
  
  double __TOL_NR_THERMEQUIL;  /* 
				  parameter for Newton-Raphon for the THERMEQUIL iterative problem
			       */
  
  double __TOL_NR_POTENTIAL;   /* 
				  parameter for Newton-Raphon for the POTENTIAL iterative problem
			       */
  
  int __ITERMAX_NR_POTENTIAL;   /* 
				   parameter for Newton-Raphon for the POTENTIAL iterative problem
				*/
  
  double   __TOL_EIGVALS;      /* 
				  parameter for multi-section iterative method for eigenvalues
			       */
  
  double __TOL_EIGVECS_IPIM;   /* 
				  parameter for Inverse Power Iterative Method for eigenvectors
			       */
  
  double   __TOL_LINSYS_SRJ;   /* 
				  parameter for the Scheduled Relaxed Jacobi iterative method
			       */

  int __ITERMAX_LINSYS_SRJ;    /* 
				  parameter for Newton-Raphon for the POTENTIAL iterative problem
			       */
};
/*
 end section: 'itmethPar01'
*/

#endif
