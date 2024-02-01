#include "discrdim.h"

/*
 begin section: 'Meshes_02'
 written: 20230516
 last modified: 20230516
 author: Francesco Vecil

 description:
    Constructor for the discretization meshes.
*/
discrDim::discrDim(const int par_N, const double par_min, const double par_max, const meshType mt)
{
  dd_mt = mt;
  
  dd_N = par_N;
  dd_min = par_min;
  dd_max = par_max;
  
  switch( mt )
    {
    case PHI_mesh :
      dd_delta = (par_max - par_min) / par_N;
      break;

    default :
      dd_delta = (par_max - par_min) / (par_N - 1);
      break;
    }

  dd_delta_m1 = 1./dd_delta;
  dd_delta_m2 = dd_delta_m1 * dd_delta_m1;
}

discrDim::discrDim(const discrDim &dd2)
{
  dd_mt       = dd2.dd_mt;
  dd_N        = dd2.dd_N;
  dd_min      = dd2.dd_min;
  dd_max      = dd2.dd_max;
  dd_delta    = dd2.dd_delta;
  dd_delta_m1 = dd2.dd_delta_m1;
  dd_delta_m2 = dd2.dd_delta_m2;
}

void discrDim::printParams()
{
  cerr << "       dd_mt = " << dd_mt       << endl;
  cerr << "        dd_N = " << dd_N        << endl;
  cerr << "      dd_min = " << dd_min      << endl;
  cerr << "      dd_max = " << dd_max      << endl;
  cerr << "    dd_delta = " << dd_delta    << endl;
  cerr << " dd_delta_m1 = " << dd_delta_m1 << endl;
  cerr << " dd_delta_m1 = " << dd_delta_m2 << endl;

  cerr << " mesh = [ " << mesh(0);
  for(int i=1; i<dd_N; ++i)
    {
      cerr << ", " << mesh(i);
    }
  cerr << " ]" << endl;
  
  cerr << endl;

  return;
}
/*
 end section: 'Meshes_02'
*/


