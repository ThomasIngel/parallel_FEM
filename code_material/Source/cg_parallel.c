// cg_parallel

#include "hpc.h"
#include "mesh_trans.h"
#include <mpi.h>
#include "blas_level1.h"

double* make_global_r0(index* c,double* r,double* rhs_glob,index nlocal){
  for(int i=0;i<nlocal;i++){
    rhs_glob[c[i]] = r[i];
  }
  return rhs_glob;
}

void make_local_r0(index* c,double* rhs_glob,double* r,index nlocal){
  for(int i=0;i<nlocal;i++){
     r[i] = rhs_glob[c[i]];
  }
}

void accum_vec_r0(index* c, double* vec, double* accum, index nloc, index n){
    // Akkumuliert vec in accum (beide DIM nloc)
    double* vec_loc = calloc(n,sizeof(double));
    double accum_buff[n];
    MPI_Allreduce(
        make_global_r0(c,vec,vec_loc,nloc),
        accum_buff,
        n,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD);
    free(vec_loc);
    make_local_r0(c,accum_buff,accum,nloc);
}

double* ddot_local(double* w, double* r, index nloc, double* ddot_loc){
  ddot_loc[0] = 0;
  for(int i=0;i<nloc;i++){
    ddot_loc[0] += w[i]*r[i];
  }
  return ddot_loc;
}

double get_sigma(double* w, double* r, index nloc){
  double ddot[1];
  double ddot_loc[1];
  MPI_Allreduce(
    ddot_local(w,r,nloc,ddot_loc),
    ddot,
    1,
    MPI_DOUBLE,
    MPI_SUM,
    MPI_COMM_WORLD);
  return ddot[0];
}

void cg_parallel(const sed *A, const double *b, double *u, double tol, 
		double (*f_dir)(double *), mesh_trans* mesh_loc, MPI_Comm comm) {
        // A   - Part of the stiffness matrix (sed Format!)
        // b   - Part of the righthand side
        // u   - Part of the inital guess for the solution
        // tol - Toleranz (stopping criteria)
        
        int myid;
        MPI_Comm_rank(comm,&myid);

        // gather variables for readability
		index nfixed = mesh_loc->nfixed_loc;
		index* fixed = mesh_loc->fixed_loc;
		double* coord = mesh_loc->domcoord;
	
		// calculate dirichlet bcs
		double dir[nfixed];
		double x[2];
		for (index i = 0; i < nfixed; ++i){
			x[0] = coord[2 * fixed[i]];
			x[1] = coord[2 * fixed[i] + 1];
			dir[i] = f_dir(x);
		}

        index n = A->n ;                                //Matrix Dim

        // SWITCH VON ACCUM ÜBER R0 (1) ODER RED/BLACK (0)
        index r0 = 1;
        
        // set nodes at dirichlet to 0 because of homogenization
        inc_dir_r(u, fixed, nfixed);

        double r[n];
        blasl1_dcopy(b, r, n, 1.0);                     //kopiert b in r (also r=b)

        // r = b - A*u    r = r-A*u
        sed_spmv_adapt(A, u, r, -1.0);                  //Ergebnis stet in r
        
       	// residuum is 0 at dirichlet bcs
       	inc_dir_r(r, fixed, nfixed);

        // w = Akkumulation (Summe über Prozessoren)
        double w[n];                                                            // Dimension??
        double t0 = walltime();
        double t1;
        if(r0 == 1){
            accum_vec_r0(mesh_loc->c,r,w,n,mesh_loc->ncoord_glo);
            t1 = walltime()-t0;
            printf("PROCESS %d: %fs for rank0 Akkum\n",myid,t1);
        }else{
            accum_vec(mesh_loc, r, w, comm);
            t1 = walltime()-t0;
            printf("PROCESS %d: %fs for parallel Akkum\n",myid,t1);
        }

        // sigma = w'*r (Skalarprodukt)
        double sigma_0;
        t0 = walltime();
        if(r0 == 1){
            sigma_0 = get_sigma(w,r,n);
            t1 = walltime()-t0;
            printf("PROCESS %d: %fs for rank0 ddot\n",myid,t1);
        }else{
            sigma_0 = ddot_parallel(w, r, n, comm);
            t1 = walltime()-t0;
            printf("PROCESS %d: %fs for parallel ddot\n",myid,t1);
        }     
}
