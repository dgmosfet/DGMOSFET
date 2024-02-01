#ifndef _SCATTPARAMS_H
#define _SCATTPARAMS_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#define     _ERR_SCATTPAR_NOTOPEN                          35 /*
							       While constructing the physDevice class, the configuration file
							       coudl not be opened.
							     */


#define     _ERR_SCATTPAR_TOKNOSEP                         36 /*
							       While reading the physDevice configuration file, the separator token '='
							       has not been found for a specific token. The syntax is therefore incorrect.
							     */

#define     _ERR_SCATTPAR_EMPTYVAL                         37 /*
							       While reading the physDevice configuration file, the value
							       field for the given token is empty.
							     */

/*
 begin section: 'scattPar01'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 description:
    The following class contains the parameters relative to the scattering phenomena.

 Technical notes
 ===============
*/
class scattParams
{
public:
  scattParams();

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

  __host__ __device__ inline bool    get_PHONONS()           const { return __PHONONS;           }
  __host__ __device__ inline bool    get_ROUGHNESS()         const { return __ROUGHNESS;         }
  __host__ __device__ inline double  get_LENGTH_SR()         const { return __LENGTH_SR;         }
  __host__ __device__ inline double  get_DELTA_STRETCH_SR()  const { return __DELTA_STRETCH_SR;  }
  __host__ __device__ inline double  get_DELTA_SR()          const { return __DELTA_SR;          }
  
public:
  bool            __PHONONS;   /* 
				  phonon scatterings is active (true)/inactive (false)             
				  default value is set to TRUE
			       */

  bool          __ROUGHNESS;   /* 
				  surface roughness scattering is active (true)/inactive (false)   
				  default value is set to FALSE
			       */

  double        __LENGTH_SR;   /* 
				  parameter for surface roughness scattering (specs to be written) 
			       */

  double __DELTA_STRETCH_SR;   /* 
				  parameter for surface roughness scattering (specs to be written) 
			       */

  double         __DELTA_SR;   /* 
				  parameter for surface roughness scattering (specs to be written) 
			       */
};
/*
 end section: 'scattPar01'
*/

#endif
