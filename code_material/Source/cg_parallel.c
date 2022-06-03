// cg_parallel

#include "hpc.h"
#include "mesh_trans.h"
#include <mpi.h>
#include "blas_level1.h"

void cg_parallel(const sed *A, const double *b, double *u, double tol, 
		double (*f_dir)(double *), mesh_trans* mesh_loc, MPI_Comm comm) {
        // A   - Part of the stiffness matrix (sed Format!)
        // b   - Part of the righthand side
        // u   - Part of the inital guess for the solution
        // tol - Toleranz (stopping criteria)
        
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
        
        // incoporate dirichlet bcs in u0
        inc_dir_u(u, dir, fixed, nfixed);

        double r[n];
        blasl1_dcopy(b, r, n, 1.0);                     //kopiert b in r (also r=b)

        // r = b - A*u    r = r-A*u
        sed_spmv_adapt(A, u, r, -1.0);                  //Ergebnis stet in r
        
       	// residuum is 0 at dirichlet bcs
       	inc_dir_r(r, fixed, nfixed);

        // w = Akkumulation (Summe über Prozessoren)
        double w[n];                                                            // Dimension??
        accum_vec(mesh_loc, r, w, comm);

        // sigma = w'*r (Skalarprodukt)
        double sigma_0 = ddot_parallel(w, r, n, comm);
        double sigma = sigma_0;

        // d = w
        double d[n];
        blasl1_dcopy(w, d, n, 1.0);

        // Speicher allokieren für ad
        double *ad = calloc(n, sizeof(double));         // ad mit 0en initiieren calloc
        if (ad == NULL) {
                printf("Error! memory not allocated.");
                exit(0);
        }

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

                // r = r - alpha*ad
                blasl1_daxpy(r, ad, n, -alpha, 1.0);
                
                // residuum is 0 at dirichlet bcs
                inc_dir_r(r, fixed, nfixed);

                // w = Akkumulation (Summe über Prozessoren (C*r))
                accum_vec(mesh_loc, r, w, comm);              // Passt das so??

                // sigma_neu = w' * r
                double sigma_neu = ddot_parallel(w, r, n, comm);

                // d = (sigma_neu/sigma)*d + w
                blasl1_daxpy(d, w, n, 1.0, sigma_neu / sigma);

                // sigma = sigma_neu
                sigma = sigma_neu;

                // printf("k = %d \t norm = %10g\n", k, sqrt(sigma));

        } while (sqrt(sigma) > tol);

        free(ad);
}
