#include "rescalingparams.h"

/*
 begin section: 'AdimPar02'
 written: 20230515
 last modified: 20230515
 author: Francesco Vecil

 description:
    Constructor of the class.
*/
rescalingParams::rescalingParams(const physDevice *pd, const physConsts *pc)
{
  __xstar    = pd->compute_XLENGTH();
  __zstar    = pd->get_ZWIDTH();
  __tstar    = pd->compute_XLENGTH()*sqrt(pc->__me/(pc->__kb*pc->__latticetemp));
  __potstar  = pc->__kb*pc->__latticetemp/pc->__poscharge;
  __kstar    = sqrt(pc->__me*pc->__kb*pc->__latticetemp)/pc->__hbar;
  __epsstar  = pc->__kb*pc->__latticetemp;
  __vstar    = sqrt(pc->__kb*pc->__latticetemp/pc->__me);
  __maxwstar = (pc->__hbar*pc->__hbar)/(pc->__me*pc->__kb*pc->__latticetemp);
  __chistar  = 1./sqrt(pd->get_ZWIDTH());
  __nstar    = (pc->__me*pc->__kb*pc->__latticetemp)/(pc->__hbar*pc->__hbar*pd->get_ZWIDTH());
  __rhostar  = (pc->__me*pc->__kb*pc->__latticetemp)/(pc->__hbar*pc->__hbar);
  __jstar    = pc->__poscharge*sqrt(pc->__me)/(pc->__hbar*pc->__hbar)*(pc->__kb*pc->__latticetemp)*sqrt(pc->__kb*pc->__latticetemp);
  __scstar   = 1./__tstar;
  __astar    = (pc->__me*pc->__poscharge)/(pd->get_ZWIDTH()*pd->get_ZWIDTH()*pc->__hbar*pc->__hbar);
};
/*
 end section: 'AdimPar02'
*/



/*
 begin section: 'AdimPar03'
 written: 20230515
 last modified: 20230515
 author: Francesco Vecil

 description:
    The method prints the content of all the data of this class.
*/
void rescalingParams::printParams()
{
  cerr << "    __xstar = " << get_xstar()    << endl;
  cerr << "    __zstar = " << get_zstar()    << endl;
  cerr << "    __tstar = " << get_tstar()    << endl;
  cerr << "  __potstar = " << get_potstar()  << endl; 
  cerr << "    __kstar = " << get_kstar()    << endl; 
  cerr << "  __epsstar = " << get_epsstar()  << endl; 
  cerr << "    __vstar = " << get_vstar()    << endl; 
  cerr << " __maxwstar = " << get_maxwstar() << endl; 
  cerr << "  __chistar = " << get_chistar()  << endl; 
  cerr << "    __nstar = " << get_nstar()    << endl; 
  cerr << "  __rhostar = " << get_rhostar()  << endl; 
  cerr << "    __jstar = " << get_jstar()    << endl; 
  cerr << "   __scstar = " << get_scstar()   << endl; 
  cerr << "    __astar = " << get_astar()    << endl; 

  return;
}
/*
 end section: 'AdimPar03'
*/



