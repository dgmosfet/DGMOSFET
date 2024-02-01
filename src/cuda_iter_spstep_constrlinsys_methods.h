void CPU_constr_linsys_LIS( const FixedPointType );
void CPU_constr_linsys_SRJ( const FixedPointType );
void GPU_constr_linsys_SRJ( const FixedPointType );
void constr_linsys( const FixedPointType );



/*
  name          : 'check_constr_linsys_input', 'check_constr_linsys_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'constr_linsys' method.
                  As input  : (i) frechet, (ii) tovoldens_OLD, (iii) pot_OLD
		  As output : (iv) matrix_2d, (v) rhs
 */
#define  _ERR_CHECK_CONSTR_LINSYS_INPUT_OUTPUT   163
void check_constr_linsys_input();
void check_constr_linsys_output();
