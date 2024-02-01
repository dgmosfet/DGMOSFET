#ifndef _SOLVERPARAMS_H
#define _SOLVERPARAMS_H

#include "meshparams.h"
#include "scattparams.h"
#include "itmethparams.h"

/*
 begin section : 'InheritanceExperiment01'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 description: ...
*/
class solverParams : public meshParams, public scattParams, public itmethParams
{
 public:
  void readInline(int _argc, char **_argv);
  void printDataFields();
};
/*
 end section: 'InheritanceExperiment01'
*/

#endif
