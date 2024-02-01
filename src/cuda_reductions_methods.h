void CPU_voldens_totvoldens(const SPPhase spphase);
/* void GPU_voldens_totvoldens(const SPPhase spphase); */
void voldens_totvoldens(const SPPhase spphase);
void CPU_compute_surfdens();


void CPU_currdens_voldens( const int STAGE );
void currdens_voldens( const int STAGE );


void macro( const int s );

enum IntegrType { TOPLEFT_RECTANGLES, TOPRIGHT_RECTANGLES, TRAPEZOIDS, PHISTYLE };
double integrate_R1( double *f, int n1, double d1, IntegrType it1);
double integrate_R2( double *f, int n1, double d1,  IntegrType it1, int n2, double d2,IntegrType it2);

double norm_L1_device( double *data, int N );
double norm_L2_device( double *data, int N );
double norm_Linf_device( double *data, int N );
double sum_device( double *data, int N );
