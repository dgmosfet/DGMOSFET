// CONFIGURATION AND MESHES
void config();
void config_bandstruct();
void config_scatterings();
void allocate_memory();
void config_meshes();
void config_constdata();
void config_datastructs();
void config_spsolvers();
void config_rhs();
void config_scintegrate();
void config_mapping();
void compute_initcond();
#ifndef __CUDACC__
void CPU_initialize_pdf();
#endif

inline double x( const int i ) const    { return i*_dx;         }
inline double z( const int j ) const    { return j*_dz;         }
inline double w( const int l )          { return (l+0.5)*_dw;   }
inline double phi( const int m )        { return m*_dphi;       }
