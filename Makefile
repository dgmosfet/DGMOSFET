CUDAROOT    = /usr/local/cuda-10.2
LISROOT     = /home/jmmantas/lis-1.7.33
OPENMPIH    = /usr/include/openmpi-x86_64
OPENMPILIB  = /usr/lib64/openmpi/lib
# LAPACKLIB   = lapack-3.7.0
# BLASLIB     = BLAS-3.7.0

SRC_DIR     = src
CC          = nvcc

CFLAGS = -I$(CUDAROOT)/include \
	-I/home/jmmantas/NVIDIA_CUDA-10.1_Samples/common/inc/ \
	-I$(LISROOT)/include/ \
	-I$(OPENMPIH)/ \
	-Xcompiler \
	-fopenmp \
	--compiler-options \
	-DUNIX \
	-O3 \
	-m64 \
	-g \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_86,code=sm_86 \
	-Xptxas -v \
	-Xptxas -warn-lmem-usage \
	-Xptxas -warn-spills \
	-Xptxas -dlcm=ca \
	-lineinfo \
	-use_fast_math \
	# -D _ITER_DEBUG

CCLINKLIBS = -L$(CUDAROOT)/lib64/ \
	-L$(OPENMPILIB)/ \
	-L$(LISROOT)/src/.libs/ \
	-lcudart \
	-lm \
	-lgfortran \
	-ldl \
	-lnsl \
	-lutil \
	-lm \
	-llapack \
	-lblas \
	-lgomp \
	$(LISROOT)/src/.libs/liblis.a \

NAME = detmosfet

HEADERS = \
	$(SRC_DIR)/adimparams.h \
	$(SRC_DIR)/cuda_bte_methods.h \
	$(SRC_DIR)/cuda_bte_rk_kernels.h \
	$(SRC_DIR)/cuda_bte_rk_methods.h \
	$(SRC_DIR)/cuda_bte_scatterings_methods.h \
	$(SRC_DIR)/cuda_bte_scatterings_phonons_kernels.h \
	$(SRC_DIR)/cuda_bte_scatterings_phonons_methods.h \
	$(SRC_DIR)/cuda_bte_scatterings_roughness_kernels.h \
	$(SRC_DIR)/cuda_bte_scatterings_roughness_methods.h \
	$(SRC_DIR)/cuda_bte_weno_kernels.h \
	$(SRC_DIR)/cuda_bte_weno_methods.h \
	$(SRC_DIR)/cuda_comptime.h \
	$(SRC_DIR)/cuda_config.h \
	$(SRC_DIR)/cuda_config_kernels.h \
	$(SRC_DIR)/cuda_config_methods.h \
	$(SRC_DIR)/cuda_constdata.h \
	$(SRC_DIR)/cuda_datastructs.h \
	$(SRC_DIR)/debug_flags.h \
	$(SRC_DIR)/cuda_dens_kernels.h \
	$(SRC_DIR)/cuda_dens_methods.h \
	$(SRC_DIR)/cuda_filestorage.h \
	$(SRC_DIR)/cuda_iter_eigenstates_kernels.h \
	$(SRC_DIR)/cuda_iter_eigenstates_methods.h \
	$(SRC_DIR)/cuda_iter_eigenstates_eigenvalues_kernels.h \
	$(SRC_DIR)/cuda_iter_eigenstates_eigenvalues_methods.h \
	$(SRC_DIR)/cuda_iter_eigenstates_eigenvectors_kernels.h \
	$(SRC_DIR)/cuda_iter_eigenstates_eigenvectors_methods.h \
	$(SRC_DIR)/cuda_iter_kernels.h \
	$(SRC_DIR)/cuda_iter_methods.h \
	$(SRC_DIR)/cuda_iter_spstep_constrlinsys_kernels.h \
	$(SRC_DIR)/cuda_iter_spstep_constrlinsys_methods.h \
	$(SRC_DIR)/cuda_iter_spstep_frechet_methods.h \
	$(SRC_DIR)/cuda_iter_spstep_frechet_kernels.h \
	$(SRC_DIR)/cuda_iter_spstep_kernels.h \
	$(SRC_DIR)/cuda_iter_spstep_methods.h \
	$(SRC_DIR)/cuda_iter_spstep_solvelinsys_kernels.h \
	$(SRC_DIR)/cuda_iter_spstep_solvelinsys_methods.h \
	$(SRC_DIR)/cuda_kernels.h \
	$(SRC_DIR)/cuda_mappings.h \
	$(SRC_DIR)/cuda_mosfetproblem.h \
	$(SRC_DIR)/cuda_reductions.h \
	$(SRC_DIR)/cuda_reductions_kernels.h \
	$(SRC_DIR)/cuda_reductions_methods.h \
	$(SRC_DIR)/cuda_testing.h \
	$(SRC_DIR)/cuda_time_integration_methods.h \
	$(SRC_DIR)/discrdim.h \
	$(SRC_DIR)/discrmeshes.h \
	$(SRC_DIR)/errors_and_exceptions.h \
	$(SRC_DIR)/itmethparams.h \
	$(SRC_DIR)/gridconfig.h \
	$(SRC_DIR)/kernelconfig.h \
	$(SRC_DIR)/mosfetproblem.h \
	$(SRC_DIR)/meshparams.h \
	$(SRC_DIR)/physconsts.h \
	$(SRC_DIR)/physdevice.h \
	$(SRC_DIR)/rescalingparams.h \
	$(SRC_DIR)/scattparams.h \
	$(SRC_DIR)/solverparams.h \
	$(SRC_DIR)/srjparams.h \

MODULES = \
	$(SRC_DIR)/adimparams.o \
	$(SRC_DIR)/cuda_bte.o \
	$(SRC_DIR)/cuda_bte_rk.o \
	$(SRC_DIR)/cuda_bte_scatterings.o \
	$(SRC_DIR)/cuda_bte_scatterings_phonons.o \
	$(SRC_DIR)/cuda_bte_scatterings_roughness.o \
	$(SRC_DIR)/cuda_bte_scatterings_roughness_config.o \
	$(SRC_DIR)/cuda_bte_scatterings_roughness_overlap.o \
	$(SRC_DIR)/cuda_bte_weno.o \
	$(SRC_DIR)/cuda_comptime.o \
	$(SRC_DIR)/cuda_config.o \
	$(SRC_DIR)/cuda_constdata.o \
	$(SRC_DIR)/cuda_dens.o \
	$(SRC_DIR)/cuda_filestorage.o \
	$(SRC_DIR)/cuda_iter.o \
	$(SRC_DIR)/cuda_iter_eigenstates.o \
	$(SRC_DIR)/cuda_iter_eigenstates_eigenvalues.o \
	$(SRC_DIR)/cuda_iter_eigenstates_eigenvectors.o \
	$(SRC_DIR)/cuda_iter_spstep.o \
	$(SRC_DIR)/cuda_iter_spstep_constrlinsys.o \
	$(SRC_DIR)/cuda_iter_spstep_frechet.o \
	$(SRC_DIR)/cuda_iter_spstep_solvelinsys.o \
	$(SRC_DIR)/cuda_mosfetproblem.o \
	$(SRC_DIR)/cuda_reductions.o \
	$(SRC_DIR)/cuda_solve.o \
	$(SRC_DIR)/cuda_testing.o \
	$(SRC_DIR)/cuda_time_integration.o \
	$(SRC_DIR)/discrdim.o \
	$(SRC_DIR)/discrmeshes.o \
	$(SRC_DIR)/gridconfig.o \
	$(SRC_DIR)/kernelconfig.o \
	$(SRC_DIR)/itmethparams.o \
	$(SRC_DIR)/main.o \
	$(SRC_DIR)/meshparams.o \
	$(SRC_DIR)/mosfetproblem.o \
	$(SRC_DIR)/physconsts.o \
	$(SRC_DIR)/physdevice.o \
	$(SRC_DIR)/rescalingparams.o \
	$(SRC_DIR)/scattparams.o \
	$(SRC_DIR)/solverparams.o \
	$(SRC_DIR)/srjparams.o \

all : $(NAME)

#******************************************
#             CLASSES COMPILATION         *
#******************************************
$(SRC_DIR)/adimparams.o         : $(SRC_DIR)/adimparams.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/adimparams.cu -o $(SRC_DIR)/adimparams.o

$(SRC_DIR)/cuda_bte.o         : $(SRC_DIR)/cuda_bte.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_bte.cu -o $(SRC_DIR)/cuda_bte.o

$(SRC_DIR)/cuda_bte_rk.o         : $(SRC_DIR)/cuda_bte_rk.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_bte_rk.cu -o $(SRC_DIR)/cuda_bte_rk.o

$(SRC_DIR)/cuda_bte_scatterings.o         : $(SRC_DIR)/cuda_bte_scatterings.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_bte_scatterings.cu -o $(SRC_DIR)/cuda_bte_scatterings.o

$(SRC_DIR)/cuda_bte_scatterings_phonons.o         : $(SRC_DIR)/cuda_bte_scatterings_phonons.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_bte_scatterings_phonons.cu -o $(SRC_DIR)/cuda_bte_scatterings_phonons.o

$(SRC_DIR)/cuda_bte_scatterings_roughness.o         : $(SRC_DIR)/cuda_bte_scatterings_roughness.cu $(HEADERS)   $(SRC_DIR)/cuda_bte_scatterings_roughness_config.cu   $(SRC_DIR)/cuda_bte_scatterings_roughness_overlap.cu 
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_bte_scatterings_roughness.cu -o $(SRC_DIR)/cuda_bte_scatterings_roughness.o

$(SRC_DIR)/cuda_bte_scatterings_roughness_config.o         : $(SRC_DIR)/cuda_bte_scatterings_roughness_config.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_bte_scatterings_roughness_config.cu -o $(SRC_DIR)/cuda_bte_scatterings_roughness_config.o

$(SRC_DIR)/cuda_bte_scatterings_roughness_overlap.o         : $(SRC_DIR)/cuda_bte_scatterings_roughness_overlap.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_bte_scatterings_roughness_overlap.cu -o $(SRC_DIR)/cuda_bte_scatterings_roughness_overlap.o

$(SRC_DIR)/cuda_bte_weno.o           : $(SRC_DIR)/cuda_bte_weno.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_bte_weno.cu -o $(SRC_DIR)/cuda_bte_weno.o

$(SRC_DIR)/cuda_comptime.o           : $(SRC_DIR)/cuda_comptime.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_comptime.cu -o $(SRC_DIR)/cuda_comptime.o

$(SRC_DIR)/cuda_config.o           : $(SRC_DIR)/cuda_config.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_config.cu -o $(SRC_DIR)/cuda_config.o

$(SRC_DIR)/cuda_constdata.o           : $(SRC_DIR)/cuda_constdata.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_constdata.cu -o $(SRC_DIR)/cuda_constdata.o

$(SRC_DIR)/cuda_dens.o           : $(SRC_DIR)/cuda_dens.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_dens.cu -o $(SRC_DIR)/cuda_dens.o

$(SRC_DIR)/cuda_filestorage.o         : $(SRC_DIR)/cuda_filestorage.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_filestorage.cu -o $(SRC_DIR)/cuda_filestorage.o

$(SRC_DIR)/cuda_iter.o           : $(SRC_DIR)/cuda_iter.cu $(HEADERS) $(SRC_DIR)/cuda_iter_eigenstates.cu $(SRC_DIR)/cuda_reductions.cu
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_iter.cu -o $(SRC_DIR)/cuda_iter.o

$(SRC_DIR)/cuda_iter_eigenstates.o           : $(SRC_DIR)/cuda_iter_eigenstates.cu $(HEADERS) $(SRC_DIR)/cuda_iter_eigenstates_methods.h $(SRC_DIR)/cuda_iter_eigenstates_kernels.h
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_iter_eigenstates.cu -o $(SRC_DIR)/cuda_iter_eigenstates.o

$(SRC_DIR)/cuda_iter_eigenstates_eigenvalues.o           : $(SRC_DIR)/cuda_iter_eigenstates_eigenvalues.cu $(HEADERS) $(SRC_DIR)/cuda_iter_eigenstates_eigenvalues_methods.h $(SRC_DIR)/cuda_iter_eigenstates_eigenvalues_kernels.h
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_iter_eigenstates_eigenvalues.cu -o $(SRC_DIR)/cuda_iter_eigenstates_eigenvalues.o

$(SRC_DIR)/cuda_iter_eigenstates_eigenvectors.o           : $(SRC_DIR)/cuda_iter_eigenstates_eigenvectors.cu $(HEADERS) $(SRC_DIR)/cuda_iter_eigenstates_eigenvectors_methods.h $(SRC_DIR)/cuda_iter_eigenstates_eigenvectors_kernels.h
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_iter_eigenstates_eigenvectors.cu -o $(SRC_DIR)/cuda_iter_eigenstates_eigenvectors.o

$(SRC_DIR)/cuda_iter_spstep.o           : $(SRC_DIR)/cuda_iter_spstep.cu $(HEADERS) $(SRC_DIR)/cuda_iter_spstep_frechet.cu $(SRC_DIR)/cuda_iter_spstep_constrlinsys.cu 
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_iter_spstep.cu -o $(SRC_DIR)/cuda_iter_spstep.o

$(SRC_DIR)/cuda_iter_spstep_constrlinsys.o       : $(SRC_DIR)/cuda_iter_spstep_constrlinsys.cu $(HEADERS) 
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_iter_spstep_constrlinsys.cu -o $(SRC_DIR)/cuda_iter_spstep_constrlinsys.o

$(SRC_DIR)/cuda_iter_spstep_frechet.o       : $(SRC_DIR)/cuda_iter_spstep_frechet.cu $(HEADERS) 
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_iter_spstep_frechet.cu -o $(SRC_DIR)/cuda_iter_spstep_frechet.o

$(SRC_DIR)/cuda_iter_spstep_solvelinsys.o       : $(SRC_DIR)/cuda_iter_spstep_solvelinsys.cu $(HEADERS) 
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_iter_spstep_solvelinsys.cu -o $(SRC_DIR)/cuda_iter_spstep_solvelinsys.o

$(SRC_DIR)/discrdim.o         : $(SRC_DIR)/discrdim.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/discrdim.cu -o $(SRC_DIR)/discrdim.o

$(SRC_DIR)/discrmeshes.o         : $(SRC_DIR)/discrmeshes.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/discrmeshes.cu -o $(SRC_DIR)/discrmeshes.o

$(SRC_DIR)/gridconfig.o         : $(SRC_DIR)/gridconfig.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/gridconfig.cu -o $(SRC_DIR)/gridconfig.o

$(SRC_DIR)/kernelconfig.o         : $(SRC_DIR)/kernelconfig.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/kernelconfig.cu -o $(SRC_DIR)/kernelconfig.o

$(SRC_DIR)/itmethparams.o         : $(SRC_DIR)/itmethparams.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/itmethparams.cu -o $(SRC_DIR)/itmethparams.o

$(SRC_DIR)/cuda_mosfetproblem.o       : $(SRC_DIR)/cuda_mosfetproblem.cu $(HEADERS) $(SRC_DIR)/cuda_reductions.cu 
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_mosfetproblem.cu -o $(SRC_DIR)/cuda_mosfetproblem.o

$(SRC_DIR)/cuda_reductions.o         : $(SRC_DIR)/cuda_reductions.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_reductions.cu -o $(SRC_DIR)/cuda_reductions.o

$(SRC_DIR)/physconsts.o         : $(SRC_DIR)/physconsts.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/physconsts.cu -o $(SRC_DIR)/physconsts.o

$(SRC_DIR)/physdevice.o         : $(SRC_DIR)/physdevice.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/physdevice.cu -o $(SRC_DIR)/physdevice.o

$(SRC_DIR)/rescalingparams.o         : $(SRC_DIR)/rescalingparams.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/rescalingparams.cu -o $(SRC_DIR)/rescalingparams.o

$(SRC_DIR)/scattparams.o         : $(SRC_DIR)/scattparams.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/scattparams.cu -o $(SRC_DIR)/scattparams.o

$(SRC_DIR)/solverparams.o         : $(SRC_DIR)/solverparams.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/solverparams.cu -o $(SRC_DIR)/solverparams.o

$(SRC_DIR)/srjparams.o         : $(SRC_DIR)/srjparams.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/srjparams.cu -o $(SRC_DIR)/srjparams.o

$(SRC_DIR)/cuda_solve.o           : $(SRC_DIR)/cuda_solve.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_solve.cu -o $(SRC_DIR)/cuda_solve.o

$(SRC_DIR)/cuda_testing.o           : $(SRC_DIR)/cuda_testing.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_testing.cu -o $(SRC_DIR)/cuda_testing.o

$(SRC_DIR)/cuda_time_integration.o          : $(SRC_DIR)/cuda_time_integration.cu $(HEADERS) $(SRC_DIR)/cuda_bte.cu $(SRC_DIR)/cuda_bte_weno.cu $(SRC_DIR)/cuda_bte_rk.cu $(SRC_DIR)/cuda_bte_scatterings_phonons.cu $(SRC_DIR)/cuda_reductions.cu  $(SRC_DIR)/cuda_bte_scatterings_roughness.cu  
	$(CC) $(CFLAGS) -D CUDA_CODE -c $(SRC_DIR)/cuda_time_integration.cu -o $(SRC_DIR)/cuda_time_integration.o

$(SRC_DIR)/main.o                : $(SRC_DIR)/main.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CODE -c $(SRC_DIR)/main.cu -o $(SRC_DIR)/main.o

$(SRC_DIR)/meshparams.o                : $(SRC_DIR)/meshparams.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CODE -c $(SRC_DIR)/meshparams.cu -o $(SRC_DIR)/meshparams.o

$(SRC_DIR)/mosfetproblem.o                : $(SRC_DIR)/mosfetproblem.cu $(HEADERS)
	$(CC) $(CFLAGS) -D CODE -c $(SRC_DIR)/mosfetproblem.cu -o $(SRC_DIR)/mosfetproblem.o

auxiliary_codes/cuda_lee_metricas.o       : auxiliary_codes/cuda_lee_metricas.C
	g++ -c auxiliary_codes/cuda_lee_metricas.C -o auxiliary_codes/cuda_lee_metricas.o

auxiliary_codes/cuda_postprocessing.o       : auxiliary_codes/cuda_postprocessing.cu 
	$(CC) -c auxiliary_codes/cuda_postprocessing.cu -o auxiliary_codes/cuda_postprocessing.o

#******************************************
#                  MAIN                   *
#******************************************
detmosfet : $(MODULES)
	 $(CC) $(MODULES) -o $(NAME) $(CCLINKLIBS)

clean: 
	rm -rfv                *~ $(SRC_DIR)/*~ detmosfet *.txt *.dat *.gp *.eps *.jpg *.avi *.sh PI* POTENTIAL/ BORDER/ THERMEQUIL/ INITCOND/       results/ scripts/ films_and_graphs/ comptimes_analysis/ adimparams*                  lee_metricas postprocessing report*

cleanall:    
	rm -rfv $(SRC_DIR)/*.o *~ $(SRC_DIR)/*~ detmosfet *.txt *.dat *.gp *.eps *.jpg *.avi *.sh PI* POTENTIAL/ BORDER/ THERMEQUIL/ INITCOND/ *.txt results/ scripts/ films_and_graphs/ comptimes_analysis/ adimparams* *.GP *.DAT *.pdf lee_metricas postprocessing report*

auxiliary_codes: auxiliary_codes/lee_metricas.o auxiliary_codes/postprocessing.o
	g++ auxiliary_codes/lee_metricas.o -o lee_metricas
	$(CC) auxiliary_codes/postprocessing.o -o postprocessing

