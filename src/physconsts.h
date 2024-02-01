#ifndef _PHISCONSTS_H
#define _PHISCONSTS_H

#include <cmath>
#include <iostream>
using namespace std;

/*
 begin section: 'PhysCon01'
 written: 20230515
 last modified: 20230515
 author: Francesco Vecil

 description:
    This class will contain the physical constants, declares as 'static', as they do not depend on the object.

 note:
    (i) Since C++11, the keyword 'constexpr' must be used instead of 'const' for non-integral values.
*/
class physConsts
{
public:
  void printConsts();
  
public:
  static constexpr    int           __NVALLEYS = 3;
  static constexpr double               __hbar = 6.626068e-34*.5*M_1_PI;
  static constexpr double                 __me = 9.10938188e-31;
  static constexpr double          __poscharge = 1.60217656535e-19;
  static constexpr double               __eps0 = 8.8541878176e-12;
  static constexpr double             __epsrSi = 11.7;
  static constexpr double           __epsrSiO2 = 3.9;
  static constexpr double                 __kb = 1.3806503e-23;
  static constexpr double                 __eV = 1.60217656535e-19;
  static constexpr double        __latticetemp = 300.;
  static constexpr double              __rhoSi = 2329.;
  static constexpr double                 __ul = 9040.;
  static constexpr double         __workfuncAl = 4.05*__eV;
  static constexpr double          __workfuncW = 4.50*__eV;
  static constexpr double         __affinitySi = 4.05*__eV;
};
/*
 end section: 'PhysCon01'
*/

#endif
