// file containing all the accesses to the data arrays
/* #define _SCHROED_MATRIX_SIZE      (_NZ-2) */
/* #define _SCHROED_MATRIX_SIZE_PAD  (32*idT( (_NZ+29)/(32) )) */

#define _SCHROED_MATRIX_SIZE      (NZ-2)
#define _SCHROED_MATRIX_SIZE_PAD  (32*idT( (NZ+29)/(32) ))
#define _SCHROED_ROW_SIZE         (2*_SCHROED_MATRIX_SIZE_PAD)

/*
  begin section  : Access_01
  author         : Francesco Vecil
  modified       : 2023/05/19
  notes          : The section has been completely re-written in order to factorize the
  nd-to-1d index mapping. Then, the access to _NX, _NZ, etc variables
  defined through directives to the pre-compiler are being removed.
*/

/*******************************************
 *           MAPPING FUNCTIONS             *
 *******************************************/
#define indx_dim_nu(dim,nu)   (idT)(dim)*_NVALLEYS+(nu)

__host__ inline idT host_dim_nu( const Variable dim, const idT nu )
{
  return indx_dim_nu(dim,nu);
}

// ---
#define indx_ep_r_s(ep,r,s) (idT)(ep)*NX*NZ*(2*NZ+1)+(r)*(2*NZ+1)+((s)+NZ-(r))

__host__ inline idT host_ep_r_s( const EigenProblem ep, const idT r, const idT s )
{
  const idT NX = host_dm->get_X() -> get_N();
  const idT NZ = host_dm->get_Z() -> get_N();
  
  return indx_ep_r_s(ep,r,s);
}

__host__ inline idT host_ep_r_s( const idT ep, const idT r, const idT s )
{
  const idT NX = host_dm->get_X() -> get_N();
  const idT NZ = host_dm->get_Z() -> get_N();
  
  /* return ep*NX*NZ*(2*NZ+1) + r*(2*NZ+1) + (s+NZ-r); */
  return indx_ep_r_s(ep,r,s);
}
// ---
#define indx_r_s(r,s)  (r)*(2*NZ+1)+((s)+NZ-(r))

__host__ inline idT host_r_s( const idT r, const idT s )
{
  const idT NZ = host_dm->get_Z() -> get_N();
  
  return indx_r_s(r,s);
}
// ---
#define indx_i_j(i,j) (i)*NZ+j

__host__ inline idT host_i_j( const idT i, const idT j )
{
  const idT NZ   = host_dm->get_Z()  ->get_N();
  
  return indx_i_j(i,j);
}
// ---
#define indx_i_j_jj(i,j,jj)   (i)*NZ*NZ+(j)*NZ+(jj)

__host__ inline idT host_i_j_jj( const idT i, const idT j, const idT jj )
{
  const idT NZ   = host_dm->get_Z()  ->get_N();
  
  return indx_i_j_jj(i,j,jj);
}
// ---
#define indx_i_j_ep(ep,i,j)  (i)*NZ*4+(j)*4+(idT)(ep)

__host__ inline idT host_i_j_ep( const EigenProblem ep,const idT i,const idT j )
{
  const idT NZ   = host_dm->get_Z()  ->get_N();
  
  return indx_i_j_ep(ep,i,j);
}

__host__ inline idT host_i_j_epidT( const idT ep,const idT i,const idT j )
{
  const idT NZ   = host_dm->get_Z()  ->get_N();

  return indx_i_j_ep(ep,i,j);
}
// ---
#define indx_i_nu_p(nu,p,i)  (i)*_NVALLEYS*NSBN+(nu)*NSBN+(p)

__host__ inline idT host_i_nu_p( const idT nu, const idT p, const idT i )
{
  const idT NSBN = host_dm->get_NSBN();

  return indx_i_nu_p(nu,p,i);
}
// ---
#define indx_i_j_nu_p(nu,p,i,j)  (i)*NZ*_NVALLEYS*NSBN+(j)* _NVALLEYS*NSBN+(nu)*NSBN+(p)

__host__ inline idT host_i_j_nu_p( const idT nu, const idT p, const idT i, const idT j )
{
  const idT NSBN = host_dm->get_NSBN();
  const idT NZ   = host_dm->get_Z()  ->get_N();

  return indx_i_j_nu_p(nu,p,i,j);
}
// ---
#define indx_i_nu_p_nup_pp(nu,p,nup,pp,i)   (i)*_NVALLEYS*NSBN*_NVALLEYS*NSBN+(nu)*NSBN*_NVALLEYS*NSBN+(p)* _NVALLEYS*NSBN+(nup)*NSBN+(pp)

__host__ inline idT host_i_nu_p_nup_pp( const idT nu, const idT p, const idT nup, const idT pp, const idT i )
{
  const idT NSBN = host_dm->get_NSBN();

  return indx_i_nu_p_nup_pp(nu,p,nup,pp,i);
}
// ---
#define indx_i_nu_p_pp(nu,p,pp,i)   (i)*_NVALLEYS*NSBN*NSBN+(nu)*NSBN*NSBN+(p)*NSBN+(pp)

__host__ inline idT host_i_nu_p_pp( const idT nu, const idT p, const idT pp, const idT i )
{
  const idT NSBN = host_dm->get_NSBN();

  return indx_i_nu_p_pp(nu,p,pp,i);
}
// ---
#define indx_i_jext(i,j)  (i)*NZEXT+(j)

__host__ inline idT host_i_jext( const idT i, const idT j )
{
  const idT NZEXT = host_dm -> get_ZEXT() -> get_N();
  return indx_i_jext(i,j);
}
// ---
#define indx_nu_i_p(nu,p,i)  (nu)*NX*NSBN+(i)*NSBN+(p)

__host__ inline idT host_nu_i_p( const idT nu, const idT p, const idT i )
{
  const idT NSBN = host_dm->get_NSBN();
  const idT NX   = host_dm->get_X()->get_N();

  return indx_nu_i_p(nu,p,i);
}
// ---
#define indx_nu_l(nu,l)  (nu)*NW+(l)

__host__ inline idT host_nu_l( const idT nu, const idT l )
{
  const idT NW   = host_dm->get_W()  ->get_N();

  return indx_nu_l(nu,l);
}
// ---
#define indx_nu_l_m(nu,l,m)     (nu)*NW*NPHI+(l)*NPHI+m

__host__ inline idT host_nu_l_m( const idT nu, const idT l, const idT m )
{
  const idT NW   = host_dm->get_W()  ->get_N();
  const idT NPHI = host_dm->get_PHI()->get_N();

  return indx_nu_l_m(nu,l,m);
}
// ---
#define indx_s_i_nu_p_l_m(nu,p,i,l,m,s)  (s)*NX*_NVALLEYS*NSBN*NW*NPHI+(i)*_NVALLEYS*NSBN*NW*NPHI+(nu)*NSBN*NW*NPHI+(p)*NW*NPHI+(l)*NPHI+(m)

__host__ inline idT host_s_i_nu_p_l_m( const idT nu, const idT p, const idT i, const idT l, const idT m, const idT s )
{
  const idT NSBN = host_dm->get_NSBN();
  const idT NX   = host_dm->get_X()  ->get_N();
  const idT NW   = host_dm->get_W()  ->get_N();
  const idT NPHI = host_dm->get_PHI()->get_N();

  return indx_s_i_nu_p_l_m(nu,p,i,l,m,s);
}
// ---
#define indx_i_j_dim_nu(dim,nu,i,j)  (i)*NZ*3*_NVALLEYS+(j)*3*_NVALLEYS+(idT)(dim)*_NVALLEYS+(nu)

__host__ inline idT host_i_j_dim_nu( const Variable dim,const idT nu,const idT i,const idT j )
{
  const idT NSBN = host_dm->get_NSBN();
  const idT NZ   = host_dm->get_Z()  ->get_N();

  return indx_i_j_dim_nu(dim,nu,i,j);
}
// ---
#define indx_i_nu_p_l(nu,p,i,l)  (i)*_NVALLEYS*NSBN*NW+(nu)*NSBN*NW+(p)*NW+(l)

__host__ inline idT host_i_nu_p_l( const idT nu, const idT p, const idT i, const idT l )
{
  const idT NSBN = host_dm->get_NSBN();
  const idT NX   = host_dm->get_X()  ->get_N();
  const idT NW   = host_dm->get_W()  ->get_N();

  return indx_i_nu_p_l(nu,p,i,l);
}
// ---
#define indx_i_nu_p_l_m(nu,p,i,l,m)  (i)*_NVALLEYS*NSBN*NW*NPHI+(nu)*NSBN*NW*NPHI+(p)*NW*NPHI+(l)*NPHI+(m)

__host__ inline idT host_i_nu_p_l_m( const idT nu, const idT p, const idT i, const idT l, const idT m )
{
  const idT NSBN = host_dm->get_NSBN();
  const idT NX   = host_dm->get_X()  ->get_N();
  const idT NW   = host_dm->get_W()  ->get_N();
  const idT NPHI = host_dm->get_PHI()->get_N();

  return indx_i_nu_p_l_m(nu,p,i,l,m);
}
// ---
#define indx_i_nu_p_jschr(nu,p,i,j)   (i)*_NVALLEYS*NSBN*_SCHROED_MATRIX_SIZE_PAD+(nu)*NSBN*_SCHROED_MATRIX_SIZE_PAD+(p)*_SCHROED_MATRIX_SIZE_PAD+(j-1)

__host__ inline idT host_i_nu_p_jschr( const idT nu, const idT p, const idT i, const idT j )
{
  const idT NSBN = host_dm->get_NSBN();
  const idT NZ   = host_dm->get_Z()->get_N();

  return indx_i_nu_p_jschr(nu,p,i,j);
}
// ---
#define indx_nu_l_m_mp(nu,l,m,mp)   (nu)*NW*NPHI*NPHI+(l)*NPHI*NPHI+(m)*NPHI+(mp)

__host__ inline idT host_nu_l_m_mp( const idT nu, const idT l, const idT m, const idT mp )
{
  const idT NW   = host_dm->get_W()  ->get_N();
  const idT NPHI = host_dm->get_PHI()->get_N();

  return indx_nu_l_m_mp(nu,l,m,mp);
}
// ---
#define indx_sc_nu_nup(sc,nu,nup)  (sc)*_NVALLEYS*_NVALLEYS+(nu)*_NVALLEYS+(nup)

__host__ inline idT host_sc_nu_nup( const idT sc, const idT nu, const idT nup )
{
  return indx_sc_nu_nup(sc,nu,nup);
}
// ---
#define indx_var_nu_l_m(var,nu,l,m)    (idT)(var)*_NVALLEYS*NW*NPHI+(nu)*NW*NPHI+(l)*NPHI+(m)

__host__ inline idT host_var_nu_l_m( const Variable var, const idT nu, const idT l, const idT m )
{
  const idT NW   = host_dm->get_W()  ->get_N();
  const idT NPHI = host_dm->get_PHI()->get_N();

  return indx_var_nu_l_m(var,nu,l,m);
}

/*
  end section  : Access_01
*/


/*******************************************
 *       ACCESS TO CPU-GPU DATA            *
 *******************************************/
#define                GPU_avgdriftvel(i)                                                                                                        _GPU_avgdriftvel[                    (i)             ]
__host__ inline double    &a1                    (         const idT nu,  const idT l,  const idT m                          )    {return                     _a1[ host_nu_l_m        (nu,l,m)        ];}
#define                GPU_a1(nu,l,m)                                                                                                                     _GPU_a1[ indx_nu_l_m        (nu,l,m)        ]
#define                GPU_a2(nu,p,i,l,m)                                                                                                                 _GPU_a2[ indx_i_nu_p_l_m    (nu,p,i,l,m)    ]
__host__ inline double    &a3                    (         const idT nu,  const idT p,  const idT i, const idT l, const idT m)    {return                     _a3[ host_i_nu_p_l_m    (nu,p,i,l,m)    ];}
#define                GPU_a3(nu,p,i,l,m)                                                                                                                 _GPU_a3[ indx_i_nu_p_l_m    (nu,p,i,l,m)    ]
__host__ inline double    &a3_const              (         const idT nu,  const idT l,  const idT m                          )    {return               _a3_const[ host_nu_l_m        (nu,l,m)        ];}
#define                GPU_a3_const(nu,l,m)                                                                                                         _GPU_a3_const[ indx_nu_l_m        (nu,l,m)        ]
__host__ inline BoundCond &bc_confined           (const EigenProblem ep,  const idT i,  const idT j                          )    {return            _bc_confined[ host_i_j_ep        (ep,i,j)        ];}
__host__ inline BoundCond &bc_confined           (         const idT ep,  const idT i,  const idT j                          )    {return            _bc_confined[ host_i_j_epidT     (ep,i,j)        ];}
__host__ inline double    &chi                   (         const idT nu,  const idT p,  const idT i, const idT j             )    {return                    _chi[ host_i_nu_p_jschr  (nu,p,i,j)      ];}
#define                GPU_chi(nu,p,i,j)                                                                                                                 _GPU_chi[ indx_i_nu_p_jschr  (nu,p,i,j)      ]
#define                GPU_cscatt(sc)                                                                                                                 _GPU_cscatt[                    (sc)            ]
__host__ inline double    &currdens              (         const idT nu,  const idT p,  const idT i                          )    {return               _currdens[ host_i_nu_p        (nu,p,i)        ];}
#define                GPU_currdens(nu,p,i)                                                                                                         _GPU_currdens[ indx_i_nu_p        (nu,p,i)        ]
__host__ inline double    &defpot                (         const idT s                                                       )    {return                 _defpot[                    (s)             ];}
__host__ inline double    &Deltapot_lower        (         const idT i,   const idT j                                        )    {return         _Deltapot_lower[ host_i_j           (i,j)           ];}
#define                GPU_Deltapot_lower(i,j)                                                                                                _GPU_Deltapot_lower[ indx_i_j           (i,j)           ]
__host__ inline double    &Deltapot_upper        (         const idT i,   const idT j                                        )    {return         _Deltapot_upper[ host_i_j           (i,j)           ];}
#define                GPU_Deltapot_upper(i,j)                                                                                                _GPU_Deltapot_upper[ indx_i_j           (i,j)           ]
__host__ inline double    &denom_SR              (         const idT nu,  const idT l,  const idT m, const idT mp            )    {return               _denom_SR[ host_nu_l_m_mp     (nu,l,m,mp)     ];}
#define                GPU_denom_SR(nu,l,m,mp)                                                                                                      _GPU_denom_SR[ indx_nu_l_m_mp     (nu,l,m,mp)     ]
__host__ inline double    &deps_dx               (         const idT nu,  const idT p,  const idT i                          )    {return                _deps_dx[ host_i_nu_p        (nu,p,i)        ];}
#define                GPU_deps_dx(nu,p,i)                                                                                                           _GPU_deps_dx[ indx_i_nu_p        (nu,p,i)        ]
__host__ inline double    &effmass               (    const Variable dim, const idT nu, const idT i, const idT j             )    {return                _effmass[ host_i_j_dim_nu    (dim,nu,i,j)    ];}
__host__ inline double    &eps                   (         const idT nu,  const idT p,  const idT i                          )    {return                    _eps[ host_i_nu_p        (nu,p,i)        ];}
#define                GPU_eps(nu,p,i)                                                                                                                   _GPU_eps[ indx_i_nu_p        (nu,p,i)        ]
__host__ inline double    &eps_diff_m1           (         const idT nu,  const idT p,  const idT pp, const idT i            )    {return            _eps_diff_m1[ host_i_nu_p_pp     (nu,p,pp,i)     ];}
#define                GPU_eps_diff_m1(nu,p,pp,i)                                                                                                _GPU_eps_diff_m1[ indx_i_nu_p_pp     (nu,p,pp,i)     ]
__host__ inline double    &epsr_ext              (         const idT i,   const idT j                                        )    {return               _epsr_ext[ host_i_jext        (i,j)           ];}
#define                GPU_eps_inf(nu,p,i)                                                                                                           _GPU_eps_inf[ indx_i_nu_p        (nu,p,i)        ]
__host__ inline double    &epskin                (         const idT nu,  const idT l                                        )    {return                 _epskin[ host_nu_l          (nu,l)          ];}
__host__ inline double    &integrateddenom_SR    (         const idT nu,  const idT l,  const idT m                          )    {return     _integrateddenom_SR[ host_nu_l_m        (nu,l,m)        ];}
#define                GPU_integrateddenom_SR(nu,l,m)                                                                                     _GPU_integrateddenom_SR[ indx_nu_l_m        (nu,l,m)        ]
__host__ inline double    &integrated_pdf_energy (         const idT nu,  const idT p,  const idT i, const idT l             )    {return  _integrated_pdf_energy[ host_i_nu_p_l      (nu,p,i,l)      ];}
#define                GPU_integrated_pdf_energy(nu,p,i,l)                                                                             _GPU_integrated_pdf_energy[ indx_i_nu_p_l      (nu,p,i,l)      ]
__host__ inline bool      &ischannel             (         const idT i,   const idT j                                        )    {return              _ischannel[ host_i_j           (i,j)           ];}
__host__ inline bool      &isdrain               (         const idT i,   const idT j                                        )    {return                _isdrain[ host_i_j           (i,j)           ];}
__host__ inline bool      &issource              (         const idT i,   const idT j                                        )    {return               _issource[ host_i_j           (i,j)           ];}
__host__ inline bool      &isdrain_ext           (         const idT i,   const idT j                                        )    {return            _isdrain_ext[ host_i_jext        (i,j)           ];}
__host__ inline bool      &isoxide_ext           (         const idT i,   const idT j                                        )    {return            _isoxide_ext[ host_i_jext        (i,j)           ];}
__host__ inline bool      &issource_ext          (         const idT i,   const idT j                                        )    {return           _issource_ext[ host_i_jext        (i,j)           ];}
/* __host__ inline double    &I_SR                  (         const idT nu,  const idT p,  const idT i                          )    {return                   _I_SR[ host_nu_i_p        (nu,p,i)        ];} */
/* #define                GPU_I_SR(nu,p,i)                                                                                                                 _GPU_I_SR[ indx_nu_i_p        (nu,p,i)        ] */
__host__ inline double    &I_SR                  (         const idT nu,  const idT p,  const idT i                          )    {return                   _I_SR[ host_i_nu_p        (nu,p,i)        ];}
#define                GPU_I_SR(nu,p,i)                                                                                                                 _GPU_I_SR[ indx_i_nu_p        (nu,p,i)        ]
__host__ inline double    &frechet               (         const idT i,   const idT j,  const idT jj                         )    {return                _frechet[ host_i_j_jj        (i,j,jj)        ];}
#define                GPU_frechet(i,j,jj)                                                                                                           _GPU_frechet[ indx_i_j_jj        (i,j,jj)        ]
#define                GPU_kane(nu)                                                                                                                     _GPU_kane[                    (nu)            ]
__host__ inline double    &mass                  (    const Variable dim, const idT nu                                       )    {return                   _mass[ host_dim_nu        (dim,nu)        ];}
#define                GPU_mass(dim,nu)                                                                                                                 _GPU_mass[ indx_dim_nu        (dim,nu)        ]
__host__ inline double    &matrix_2dconst        (const EigenProblem ep,  const idT r,  const idT s                          )    {return         _matrix_2dconst[ host_ep_r_s        (ep,r,s)        ];}
__host__ inline double    &matrix_2d             (         const idT r,   const idT s                                        )    {return              _matrix_2d[ host_r_s           (r,s)           ];}
#define                GPU_matrix_2d(r,s)                                                                                                          _GPU_matrix_2d[ indx_r_s           (r,s)           ]
__host__ inline double    &matrix_2dconst        (         const idT ep,  const idT r,  const idT s                          )    {return         _matrix_2dconst[ host_ep_r_s        (ep,r,s)        ];}
__host__ inline double    &matrix_2dconst_ext    (         const idT r,   const idT s                                        )    {return     _matrix_2dconst_ext[ r*(2*host_dm -> get_ZEXT() -> get_N()+1) + s+host_dm -> get_ZEXT() -> get_N()-r       ];}
/* __host__ inline double    &matrix_2dconst_ext_inv(         const idT i,   const idT j                                        )    {return _matrix_2dconst_ext_inv[ i + j*host_dm->get_X() -> get_N()*host_dm -> get_ZEXT() -> get_N() ];} */
#define                    matrix_2dconst_ext_inv(r,s)                                                                                    _matrix_2dconst_ext_inv[ (r) + (s)*NX*NZEXT                 ]
#define                GPU_matrix_2dconst_ext_inv(r,s)                                                                                _GPU_matrix_2dconst_ext_inv[ (r) + (s)*NX*NZEXT                 ]
__host__ inline double    &maxw                  (         const idT nu,  const idT l                                        )    {return                   _maxw[ host_nu_l          (nu,l)          ];}
#define                GPU_maxw(nu,l)                                                                                                                   _GPU_maxw[ indx_nu_l          (nu,l)          ]
__host__ inline double    &nd                    (         const idT i,   const idT j                                        )    {return                     _nd[ host_i_j           (i,j)           ];}
#define                GPU_nd(i,j)                                                                                                                        _GPU_nd[ indx_i_j           (i,j)           ]
__host__ inline double    &nd_ext                (         const idT i,   const idT j                                        )    {return                 _nd_ext[ host_i_jext        (i,j)           ];}
__host__ inline double    &occupations           (         const idT sc,  const idT nu, const idT nup                        )    {return            _occupations[ host_sc_nu_nup     (sc,nu,nup)     ];}
#define                GPU_occupations(sc,nu,nup)                                                                                                _GPU_occupations[ indx_sc_nu_nup     (sc,nu,nup)     ]
#define                GPU_omega(sc)                                                                                                                   _GPU_omega[                    (sc)            ]
__host__ inline double    &pdf                   (         const idT nu,  const idT p,  const idT i, const idT l, const idT m, const idT s) {return          _pdf[ host_s_i_nu_p_l_m  (nu,p,i,l,m,s)  ];}
#define                GPU_pdf(nu,p,i,l,m,s)                                                                                                             _GPU_pdf[ indx_s_i_nu_p_l_m  (nu,p,i,l,m,s)  ]
__host__ inline double    &phtemp                (         const idT s                                                       )    {return                 _phtemp[                    (s)             ];}
__host__ inline double    &pot                   (         const idT i,   const idT j                                        )    {return                    _pot[ host_i_j           (i,j)           ];}
#define                GPU_pot(i,j)                                                                                                                      _GPU_pot[ indx_i_j           (i,j)           ]
__host__ inline double    &pot_b_ext             (         const idT j                                                       )    {return              _pot_b_ext[                    (j)             ];}
__host__ inline double    &pot_ext               (         const idT i,   const idT j                                        )    {return                _pot_ext[ host_i_jext        (i,j)           ];}
#define                GPU_pot_ext(i,j)                                                                                                              _GPU_pot_ext[ indx_i_jext        (i,j)           ]
__host__ inline double    &pot_OLD               (         const idT i,   const idT j                                        )    {return                _pot_OLD[ host_i_j           (i,j)           ];}
#define                GPU_pot_OLD(i,j)                                                                                                              _GPU_pot_OLD[ indx_i_j           (i,j)           ]
__host__ inline double    &result_ext            (         const idT i,   const idT j                                        )    {return             _result_ext[ host_i_jext        (i,j)           ];}
__host__ inline double    &result_ext            (         const idT r                                                       )    {return             _result_ext[                    (r)             ];}
__host__ inline double    &rhs_pdf               (         const idT nu,  const idT p,  const idT i, const idT l, const idT m)    {return                _rhs_pdf[ host_i_nu_p_l_m    (nu,p,i,l,m)    ];}
#define                GPU_rhs_pdf(nu,p,i,l,m)                                                                                                       _GPU_rhs_pdf[ indx_i_nu_p_l_m    (nu,p,i,l,m)    ]
#define                GPU_rhs_pdf_gain(nu,p,i,l)                                                                                               _GPU_rhs_pdf_gain[ indx_i_nu_p_l      (nu,p,i,l)      ]
__host__ inline double    &test_gain             (         const idT nu,  const idT p,  const idT i, const idT l, const idT m)    {return              _test_gain[ host_i_nu_p_l_m    (nu,p,i,l,m)    ];}
#define                GPU_test_gain(nu,p,i,l,m)                                                                                                   _GPU_test_gain[ indx_i_nu_p_l_m    (nu,p,i,l,m)    ]
__host__ inline double    &test_loss             (         const idT nu,  const idT p,  const idT i, const idT l, const idT m)    {return              _test_loss[ host_i_nu_p_l_m    (nu,p,i,l,m)    ];}
#define                GPU_test_loss(nu,p,i,l,m)                                                                                                   _GPU_test_loss[ indx_i_nu_p_l_m    (nu,p,i,l,m)    ]
__host__ inline double    &rhs_pdf_gain          (         const idT nu,  const idT p,  const idT i, const idT l             )    {return           _rhs_pdf_gain[ host_i_nu_p_l      (nu,p,i,l)      ];}
__host__ inline double    &righthandside_ext     (         const idT i,   const idT j                                        )    {return      _righthandside_ext[ host_i_jext        (i,j)           ];}
__host__ inline double    &righthandside_ext     (         const idT r                                                       )    {return      _righthandside_ext[                    (r)             ];}
__host__ inline ScattType &scatttype             (         const idT s                                                       )    {return              _scatttype[                    (s)             ];}
#define                GPU_sigma_sign(nu,p,i,k)                                                                                                   _GPU_sigma_sign[ (i)*_NVALLEYS*NSBN*(_NMULTI-1) + (nu)*NSBN*(_NMULTI-1) + (p)*(_NMULTI-1) + (k) ]
__host__ inline double    &sqrtmass              (    const Variable dim, const idT nu                                       )    {return               _sqrtmass[ host_dim_nu        (dim,nu)        ];}
#define                GPU_sqrtmass(dim,nu)                                                                                                         _GPU_sqrtmass[ indx_dim_nu        (dim,nu)        ]
__host__ inline double    &sqrtmassXY            (         const idT nu                                                      )    {return             _sqrtmassXY[                    (nu)            ];}
#define                GPU_sqrtmassXY(nu)                                                                                                         _GPU_sqrtmassXY[                    (nu)            ]
__host__ inline double    &surfdens              (         const idT nu,  const idT p,  const idT i                          )    {return               _surfdens[ host_i_nu_p        (nu,p,i)        ];}
#define                GPU_surfdens(nu,p,i)                                                                                                         _GPU_surfdens[ indx_i_nu_p        (nu,p,i)        ]
__host__ inline double    &surfdens_eq           (         const idT nu,  const idT p,  const idT i                          )    {return            _surfdens_eq[ host_i_nu_p        (nu,p,i)        ];}
#define                GPU_surfdens_eq(nu,p,i)                                                                                                   _GPU_surfdens_eq[ indx_i_nu_p        (nu,p,i)        ]
#define                GPU_totcurrdens(i)                                                                                                        _GPU_totcurrdens[                    (i)             ]
#define                GPU_totsurfdens(i)                                                                                                        _GPU_totsurfdens[                    (i)             ]
__host__ inline double    &totvoldens            (         const idT i,   const idT j                                        )    {return             _totvoldens[ host_i_j           (i,j)           ];}
#define                GPU_totvoldens(i,j)                                                                                                        _GPU_totvoldens[ indx_i_j           (i,j)           ]
__host__ inline double    &totvoldens_ext        (         const idT i,   const idT j                                        )    {return         _totvoldens_ext[ host_i_jext        (i,j)           ];}
#define                GPU_totvoldens_ext(i,j)                                                                                                _GPU_totvoldens_ext[ indx_i_jext        (i,j)           ]
__host__ inline double    &totvoldens_OLD        (         const idT i,   const idT j                                        )    {return         _totvoldens_OLD[ host_i_j           (i,j)           ];}
#define                GPU_totvoldens_OLD(i,j)                                                                                                _GPU_totvoldens_OLD[ indx_i_j           (i,j)           ]
__host__ inline double    &vconf                 (         const idT i,   const idT j                                        )    {return                  _vconf[ host_i_j           (i,j)           ];}
__host__ inline double    &vel                   (    const Variable var, const idT nu, const idT l, const idT m             )    {return                    _vel[ host_var_nu_l_m    (var,nu,l,m)    ];}
#define                GPU_vel( var,nu,l,m )                                                                                                             _GPU_vel[ indx_var_nu_l_m    (var,nu,l,m)    ]
__host__ inline double    &vgate                 (         const idT i,   const idT j                                        )    {return                  _vgate[ host_i_j           (i,j)           ];}
__host__ inline double    &voldens               (         const idT nu,  const idT p,  const idT i, const idT j             )    {return                _voldens[ host_i_j_nu_p      (nu,p,i,j)      ];}
#define                GPU_voldens(nu,p,i,j)                                                                                                         _GPU_voldens[ indx_i_j_nu_p      (nu,p,i,j)      ]
__host__ inline double    &Wm1                   (         const idT nu,  const idT p,  const idT nup, const idT pp, const idT i) {return                    _Wm1[ host_i_nu_p_nup_pp (nu,p,nup,pp,i) ];}
#define                GPU_Wm1(nu,p,nup,pp,i)                                                                                                            _GPU_Wm1[ indx_i_nu_p_nup_pp (nu,p,nup,pp,i) ]
