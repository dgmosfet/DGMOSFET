#ifndef __CUDACC__
void CPU_perform_RK_1_3( const double DT );
#endif
void perform_RK_1_3( const double DT );



#ifndef __CUDACC__
void CPU_perform_RK_2_3( const double DT );
#endif
void perform_RK_2_3( const double DT );



#ifndef __CUDACC__
void CPU_perform_RK_3_3( const double DT );
#endif
void perform_RK_3_3( const double DT );


int perform_RK( const double DT, const int s );


#ifndef __CUDACC__
void CPU_set_fluxes_a3();
#endif
void set_fluxes_a3();


/* #ifndef __CUDACC__ */
/* void CPU_set_fluxes_a2(); */
/* #endif */
void set_fluxes_a2();



void CPU_set_rhs_to_zero();
void set_rhs_to_zero();
