/* void CPU_spstep(); */
/* void GPU_spstep(); */
/* void _spstep(); */
/* void spstep(); */
void spstep_CPU_Jacobi( const double omega, const FixedPointType = NEWTON_RAPHSON );
void spstep_CPU_Jacobi_vectorial( const double omega, const FixedPointType = NEWTON_RAPHSON );
void spstep_CPU_LIS( const FixedPointType = NEWTON_RAPHSON );
void spstep_CPU_SRJ( const int M, const double omega, const FixedPointType = NEWTON_RAPHSON );
void spstep_GPU_Jacobi( const double omega, const FixedPointType = NEWTON_RAPHSON );
