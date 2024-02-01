#include "solverparams.h"

void solverParams::readInline(int _argc, char **_argv)
{
  meshParams::readInline( _argc, _argv );
  scattParams::readInline( _argc, _argv );
  itmethParams::readInline( _argc, _argv );

  return;
}

void solverParams::printDataFields()
{
  meshParams::printDataFields();
  scattParams::printDataFields();
  itmethParams::printDataFields();

  return;
}
