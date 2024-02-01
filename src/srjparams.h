#ifndef _SRJPARAMS_H
#define _SRJPARAMS_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#include "discrdim.h"

/*
 begin section: 'SRJ_04'
 written: 20230607
 last modified: 20230607
 author: Francesco Vecil

 description:
    This class is meant to contain the parameters for the Scheduled Relaxed Jacobi (SRJ) iterative method.
*/
class srjParams
{
public:
  srjParams(const int N, const int P);
  srjParams(const int N, const int P, const string);
  void printDataFields();
  void compute_M();
  void compute_q();
  void compute_relaxparams();

public:
  __host__ inline int     get_P        (           )  const { return __P;           }
  __host__ inline double  get_omega    (const int i)  const { return __omega   [i]; }
  __host__ inline double  get_beta     (const int i)  const { return __beta    [i]; }
  __host__ inline int     get_q        (const int i)  const { return __q       [i]; }
  __host__ inline int     get_M        (           )  const { return __M;           }
  __host__ inline double  get_relaxpar (const int i)  const { return __relaxpar[i]; }

private:
  int __P;
  double* __omega;
  double* __beta;
  int* __q;
  int __M;
  double* __relaxpar;
};
/*
 end section: 'SRJ_04'
*/

#endif
