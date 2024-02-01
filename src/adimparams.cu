#include "adimparams.h"

/*
 begin section: 'AdimPar05'
 written: 20230515
 last modified: 20230515
 author: Francesco Vecil

 description:
    Constructor of the class.
*/
adimParams::adimParams(const physDevice *pd, const physConsts *pc, const rescalingParams *rp)
{
  __cs1 = ((pc->__hbar*pc->__hbar)/(pd->get_ZWIDTH()*pd->get_ZWIDTH()*pc->__me*rp->get_epsstar()));
  __eta = (pd->get_ZWIDTH()/pd->compute_XLENGTH());
  __cp = ((pc->__poscharge*rp->get_nstar()*pd->get_ZWIDTH()*pd->get_ZWIDTH())/(rp->get_potstar()*pc->__eps0));
  __cg = ( pd->get_ZWIDTH()*pd->get_ZWIDTH() * pc->__poscharge*pc->__poscharge * rp->get_nstar() ) / ( pc->__eps0 * pc->__kb * pc->__latticetemp );
  // __csr = pd->compute_XLENGTH()*pow(rp->get_kstar(),3)*pow(pd->get_LENGTH_SR(),2)/(4.*M_PI);
};
/*
 end section: 'AdimPar05'
*/



/*
 begin section: 'AdimPar04'
 written: 20230515
 last modified: 20230515
 author: Francesco Vecil

 description:
    The method prints the content of all the data of this class.
*/
void adimParams::printParams()
{
  cerr << "  __cs1 = " << get_cs1()  << endl;
  cerr << "  __eta = " << get_eta()  << endl; 
  cerr << "   __cp = " << get_cp()   << endl; 
  cerr << "   __cg = " << get_cg()   << endl; 
  // cerr << "  __csr = " << get_csr()  << endl; 

  return;
}
/*
 end section: 'AdimPar04'
*/




