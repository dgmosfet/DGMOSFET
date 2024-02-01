#include "mosfetproblem.h"
#include "cuda_constdata.h"

bool MOSFETProblemCuda::isgate( const int i,const int j )
{
  const double LSOURCE  = host_physdesc -> get_LSOURCE();
  const double LCHANNEL = host_physdesc -> get_LCHANNEL();
  const double LDRAIN   = host_physdesc -> get_LDRAIN();
  const double LGATE    = host_physdesc -> get_LGATE();
  const double XLENGTH  = host_physdesc -> compute_XLENGTH();

  const int NZ = host_dm->get_Z()->get_N();
  
  if( j==0 || j==NZ-1 )
    {
      const double xi = host_dm->get_X()->mesh(i);
      // if( 0.5*(LSOURCE+LCHANNEL+LDRAIN-LGATE)/XLENGTH <= x(i) && x(i) <= 0.5*(LSOURCE+LCHANNEL+LDRAIN+LGATE)/XLENGTH )
      if( 0.5*(LSOURCE+LCHANNEL+LDRAIN-LGATE)/XLENGTH <= xi && xi <= 0.5*(LSOURCE+LCHANNEL+LDRAIN+LGATE)/XLENGTH )
	return true;
      else
	return false;
    }
  else
    return false;
}

bool MOSFETProblemCuda::isoxide( const int i,const int j )
{
  const double ZWIDTH   = host_physdesc -> get_ZWIDTH();
  const double LSiO2  = host_physdesc -> get_LSiO2();

  const double zj = host_dm->get_Z()->mesh(j);
  
  bool aux = ( LSiO2/ZWIDTH<zj && zj<(ZWIDTH-LSiO2)/ZWIDTH ? false : true );
  return aux;
}

double MOSFETProblemCuda::s_chcoord( const int nu,const double wval )
{
  return sqrt(mass(Xdim,nu)*mass(Ydim,nu))*(1.+2.*_kane[nu]*host_rescpar->get_epsstar()*wval);
}
