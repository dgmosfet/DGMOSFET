#ifndef _PHISDEVICE_H
#define _PHISDEVICE_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#define     _ERR_PHYSDEV_NOTOPEN                          41 /*
							       While constructing the physDevice class, the configuration file
							       coudl not be opened.
							     */


#define     _ERR_PHYSDEV_TOKNOSEP                         42 /*
							       While reading the physDevice configuration file, the separator token '='
							       has not been found for a specific token. The syntax is therefore incorrect.
							     */

#define     _ERR_PHYSDEV_EMPTYVAL                         43 /*
							       While reading the physDevice configuration file, the value
							       field for the given token is empty.
							     */

enum ContactMaterial  { ALLUMINIUM, TUNGSTEN };
inline std::ostream& operator<<( std::ostream& os, const ContactMaterial& cm )
{
  switch( cm )
    {
    case ALLUMINIUM : os << "ALLUMINIUM"; break;
    case TUNGSTEN   : os << "TUNGSTEN"; break;
    }
  return os;
}

/*
 begin section: 'PhysDev01'
 written: 20230328
 last modified: 20230424
 author: Francesco Vecil

 Description
 ===========
    This class, in reality it is more a data structure,
    contains all the information related to the physical device
    (nothing here should be related to the simulation parameters).
    Units are meant in the International System.

 Technical notes
 ===============
 modification 1 (20230329):
    At the beginning, class physDevice was integrated into class MOSFETProblemCuda,
    but it seemed more suitable to define it outside, and insert a physDevice object
    into a MOSFETProblemCuda object.

 modification 2 (20230424):
    The method 'readInline' has been added: it will read the parameters inline, 
    following syntax, for example,
    --NSBN=12
    and modify the value inside the data structure accordingly.

 modification 3 (20230426):
    The getters functions are being implemented.
*/
class physDevice
{
public:
  physDevice(); /*
		  This is the class CONSTRUCTOR "by default",
		  in the sense that it will set the magnitudes to the default ones,
		  specified in the comments down here.
		*/

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

  __host__ __device__ inline double get_LSOURCE()                  const { return __LSOURCE;                         } 
  __host__ __device__ inline double get_LCHANNEL()                 const { return __LCHANNEL;                        }
  __host__ __device__ inline double get_LDRAIN()                   const { return __LDRAIN;                          }
  __host__ __device__ inline double compute_XLENGTH()              const { return __LSOURCE + __LCHANNEL + __LDRAIN; }
  __host__ __device__ inline double get_LGATE()                    const { return __LGATE;                           }
  __host__ __device__ inline double get_ZWIDTH()                   const { return __ZWIDTH;                          }
  __host__ __device__ inline double get_LSiO2()                    const { return __LSiO2;                           }
  __host__ __device__ inline double get_NDHIGH()                   const { return __NDHIGH;                          }
  __host__ __device__ inline double get_NDLOW()                    const { return __NDLOW;                           }
  __host__ __device__ inline double get_VCONF_WELL()               const { return __VCONF_WELL;                      }
  __host__ __device__ inline ContactMaterial get_CONTACTMATERIAL() const { return __CONTACTMATERIAL;                 }
  __host__ __device__ inline double get_VUPPERGATE()               const { return __VUPPERGATE;                      }
  __host__ __device__ inline double get_VLOWERGATE()               const { return __VLOWERGATE;                      }
  __host__ __device__ inline double get_VBIAS()                    const { return __VBIAS;                           }
  
private:
  double                   __LSOURCE; /*
					the length of the source zone
					unit is [m]
					default value is 5 nm
				       */
  
  double                  __LCHANNEL; /*
					the length of the channel zone
					unit is [m]
					default value is 10 nm
				       */
  
  double                    __LDRAIN; /*
					the length of the drain zone
					unit is [m]
					default value is 5 nm
				       */
  
  double                     __LGATE; /*
					the length of the two gates
					unit is [m]
					default value is 12 nm
				       */
  
  double                    __ZWIDTH; /*
					the length of the device along the confined dimension (Z),
					including the Silicon oxide layers
					unit is [m]
					default value is 8 nm
				       */
  
  double                     __LSiO2; /*
					the length of the Silicon oxide layers along the confined dimension (Z),
					unit is [m]
					default value is 1 nm
				       */
  
  double                    __NDHIGH; /*
					the value of the doping inside the SOURCE and DRAIN zones
					unit is [m**(-3)]
					default value is 1.E26
				       */
  
  double                     __NDLOW; /*
					the value of the doping inside the CHANNEL zone
					unit is [m**(-3)]
					default value is 1.E18
				       */
  
  double                __VCONF_WELL; /*
					the value of the potential wall across the Silicon / Silicon oxide interface
					unit is [V]
					default value is -3.15
		         	       */
  
  ContactMaterial  __CONTACTMATERIAL; /*
					the material of which the SOURCE and DRAIN metallic contacts are made
					possible values are: ALLUMINIUM, TUNGSTEN
					default value is ALLUMINIUM
				       */
  
  double                __VUPPERGATE; /*
					the applied potential at the upper gate
					unit is [V]
					default value is 0.5
				       */
  
  double                __VLOWERGATE; /*
					the applied potential at the lower gate
					unit is [V]
					default value is 0.5
				       */
  
  double                     __VBIAS; /*
					the source-to-drain applied bias
					unit is [V]
					default value is 0.1
				       */
};
/*
 end section: 'PhysDev01'
*/

#endif
