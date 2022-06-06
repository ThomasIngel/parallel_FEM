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

        int r0 = 1;
        if(myid==1){
            if(r0==0){
                printf("\nAKKUMULATION/DDOT PARALLEL!\n");
            }else{
                printf("\nAKKUMULATION/DDOT RANK0\n");
            }
        }

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
        if(r0==0){
            accum_vec(mesh_loc, r, w, comm);
        }else{
            accum_vec_r0(mesh_loc->c,r,w,n,mesh_loc->ncoord_glo);
        }

        // sigma = w'*r (Skalarprodukt)

        double sigma_0;
        if(r0==0){
            sigma_0 = ddot_parallel(w,r,n,comm);
        }else{
            sigma_0 = get_sigma(w,r,n);
        }
        double sigma = sigma_0;

        if(myid==0) printf("SIGMA_0 = %f\n",sigma_0);

        // d = w
        double d[n];
        blasl1_dcopy(w, d, n, 1.0);

        // Speicher allokieren für ad
        double *ad = calloc(n, sizeof(double));         // ad mit 0en initiieren calloc
        if (ad == NULL) {
                printf("Error! memory not allocated.");
                exit(0);
        }

        double sigma_neu;
        size_t k = 0;
         do {
                k++;

                // ad = A*d (damit nur 1x Matrixprodukt)
                if (k>0) {                              // ad mit 0en initiieren
                    for (index i=0; i<n; i++) {
                        ad[i] = 0;
                    }
                }
                // ad = A*d (damit nur 1x Matrixprodukt)
                sed_spmv_adapt(A, d, ad, 1.0);

                // alpha = sigma/(d*ad)
                double dad = ddot_parallel(d, ad, n, comm);
                double alpha = sigma / dad;

                // Update: u = u + alpha*d
                blasl1_daxpy(u, d, n, alpha, 1.0);
                
				// set dirichlet nodes to 0
				inc_dir_r(u, fixed, nfixed);

                // r = r - alpha*ad
                blasl1_daxpy(r, ad, n, -alpha, 1.0);
                
                // residuum is 0 at dirichlet bcs
                inc_dir_r(r, fixed, nfixed);

                // w = Akkumulation (Summe über Prozessoren (C*r))
                if(r0==0){
                    accum_vec(mesh_loc, r, w, comm);
                }else{
                    accum_vec_r0(mesh_loc->c,r,w,n,mesh_loc->ncoord_glo);
                }

                // sigma_neu = w' * r
                if(r0==0){ 
                    sigma_neu = ddot_parallel(w,r,n,comm);
                }else{
                    sigma_neu = get_sigma(w,r,n);
                }

                // d = (sigma_neu/sigma)*d + w
                blasl1_daxpy(d, w, n, 1.0, sigma_neu / sigma);

                // sigma = sigma_neu
                sigma = sigma_neu;

                // printf("k = %d \t norm = %10g\n", k, sqrt(sigma));

        } while (sqrt(sigma) > tol);

        if(myid==0){
            printf("ITERATIONS FOR SOLVING: %d\n",k);
        }
        free(ad);
        
        // write dirichlet data at right position in solution vector
        inc_dir_u(u, dir, fixed, nfixed);
}
