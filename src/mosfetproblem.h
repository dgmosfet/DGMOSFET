#ifndef MOSFETPROBLEM_H
#define MOSFETPROBLEM_H

#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <sstream>
#include <limits>
#include <omp.h>
#include "lis.h"
#include "lis_config.h"
#include <ctime>
#include <sys/time.h>
#include <sys/sysinfo.h>
using namespace std;

typedef int idT;

// #define  enterfunc  cerr << "Called " << setw(32) << __func__ << ", line " << setw(4) << __LINE__ << ", file " << setw(32) << __FILE__ << "..."
#define  enterfunc  cerr << "Called " << setw(32) << __func__ << ", file " << setw(48) << __FILE__ << "..."
#define  exitfunc   cerr << "[ok]" << endl

/*************************************************************
 *              ENUMERATIVE TYPES (AND OTHER TYPES)          *
 *************************************************************/
enum BoundCondType    { DIRICHLET,  NEUMANN,  INOUT_EXPL,  PERIODICITY,  SYMPOINT,  REFLECTION, INTERNAL };
inline std::ostream& operator<<( std::ostream& os, const BoundCondType& bct )
{
  switch( bct )
    {
    case DIRICHLET      : os << "DIRICHLET"; break;
    case NEUMANN        : os << "NEUMANN"; break;
    case INOUT_EXPL     : os << "INOUT_EXPL"; break;
    case PERIODICITY    : os << "PERIODICITY"; break;
    case SYMPOINT       : os << "SYMPOINT"; break;
    case REFLECTION     : os << "REFLECTION"; break;
    case INTERNAL       : os << "INTERNAL"; break;
    }
  return os;
}

enum EigenProblem     { INITCOND=0, THERMEQUIL=1, POTENTIAL=2 };
inline std::ostream& operator<<( std::ostream& os, const EigenProblem& ep )
{
  switch( ep )
    {
    case INITCOND    : os << "INITCOND"; break;
    case THERMEQUIL  : os << "THERMEQUIL"; break;
    case POTENTIAL   : os << "POTENTIAL"; break;
    }
  return os;
}

enum FixedPointType   { NEWTON_RAPHSON=0, GUMMEL=2 };
inline std::ostream& operator<<( std::ostream& os, const FixedPointType& fpt )
{
  switch( fpt )
    {
    case NEWTON_RAPHSON : os << "NEWTON_RAPHSON"; break;
    case GUMMEL         : os << "GUMMEL";         break;
    }
  return os;
}

enum FixedPointTest   { RELATIVE=0, ABSOLUTE=1 };
inline std::ostream& operator<<( std::ostream& os, const FixedPointTest& fpt )
{
  switch( fpt )
    {
    case RELATIVE : os << "RELATIVE"; break;
    case ABSOLUTE : os << "ABSOLUTE"; break;
    }
  return os;
}

enum ScattType        { PHONON_ELASTIC, PHONON_FTYPE, PHONON_GTYPE, ROUGHNESS };
inline std::ostream& operator<<( std::ostream& os, const ScattType& st )
{
  switch( st )
    {
    case PHONON_ELASTIC : os << "PHONON_ELASTIC"; break;
    case PHONON_GTYPE   : os << "PHONON_GTYPE";   break;
    case PHONON_FTYPE   : os << "PHONON_FTYPE";   break;
    case ROUGHNESS      : os << "ROUGHNESS";      break;
    }
  return os;
}

enum SPPhase          { GIVEN=0, NEW=1, OLD=2, AUXFLAG=3 };
inline std::ostream& operator<<( std::ostream& os, const SPPhase& spp )
{
  switch( spp )
    {
    case GIVEN   : os << "GIVEN"; break;
    case NEW     : os << "NEW"; break;
    case OLD     : os << "OLD"; break;
    case AUXFLAG : os << "AUXFLAG"; break;
    }
  return os;
}

enum Variable         { Xdim=0, Ydim=1, Zdim, Wdim, PHIdim };
inline std::ostream& operator<<( std::ostream& os, const Variable& v )
{
  switch( v )
    {
    case Xdim   : os << "X"; break;
    case Ydim   : os << "Y"; break;
    case Zdim   : os << "Z"; break;
    case Wdim   : os << "W"; break;
    case PHIdim : os << "PHI"; break;
    }
  return os;
}


class BoundCond {
 public:
  BoundCondType condtype;
  double point_value;
  double derivative_value;
};

/************************************
 *         LAPACK routines          *
 ************************************/
extern "C"
{ 
  void   dgesv_ (int* N, int *NRHS, double* A, int* LDA, int* IPIV, double *B, int *LDB, int* INFO);
  void   dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);
  void   dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
  void   dgtsv_ (int *N, int *NRHS, double *DL, double *D, double *DU, double *B, int *LDB, int *INFO);
  double dlamch_( char * );
  void   dstegr_(char *JOBZ, char *RANGE, int *N, double *D, double *E, double *VL, double *VU, int *IL, int *IU, double *ABSTOL, int *M, double *W, double *Z, int *LDZ, int *ISUPPZ, double *WORK, int *LWORK, int *IWORK, int *LIWORK, int *INFO);
  void   dsteqr_(char *COMPZ,int *N,double *D,double *E,double *Z,int *LDZ,double *WORK,int *INFO);
  void   dstevx_( char *, char *, long int *, const double *, const double *, double *, double *, long int *, long int *, double *, long int *, double *, double *, long int *, double *, long int *, long int *, long int *, double );
  void dgeev_( char *, char *, int *, double *, int *, double *, double *, double *, int *, double *, int *, double *, int *, int * ); 
}


#include "solverparams.h"
#include "physdevice.h"
#include "physconsts.h"
#include "rescalingparams.h"
#include "adimparams.h"
#include "discrdim.h"
#include "discrmeshes.h"
#include "gridconfig.h"
#include "srjparams.h"

#include "cuda_mosfetproblem.h"

#include "errors_and_exceptions.h"




#endif
