#ifndef __CUDACC__
void CPU_pdftilde( const int STAGE );
void CPU_integrate_phitilde();
#endif

void dens( const int STAGE );
#ifdef __CUDACC__
void GPU_dens( const int STAGE );
#else
void CPU_dens( const int STAGE );
#endif
