#include "hpc.h"
#include "mesh_trans.h"
#include <mpi.h>
#include "blas_level1.h"

// Paralleler omega_jacobi solver

void omega_jacobi(size_t n, const sed *A, const double *b, double *u, double omega, double tol, double (*f_dir)(double *), mesh_trans *mesh_loc, MPI_Comm comm) {
    // n     - Amount of columns of A (also length of most vectors in the algorithm)
    // A     - Part of a stiffness matrix (sed Format!)
    // b     - Part of the righthand side
    // u     - Part of the inital guess for the solution
    // omega - often 2/3
    // tol   - Toleranz (stopping criteria)
    // the rest is not important to know inside of this parallel solver
	
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
	
    double *Ax = A->x; // data of A
    
    // dirichlet nodes 0 because of homogenization
    inc_dir_r(u, fixed, nfixed);

    // Alg. 6.6, line 1: d := diag(A)
    double diag_inv[n];
    blasl1_dcopy(Ax,diag_inv,(index) n,1.); // the diagonal of A is now in vector diag_inv

    // Alg. 6.6, line 2: d := sum(C^T_s * d_s), s = 1,...,P
    // accumulated version of the diagonal
    // P is the amount of processors on which we split our problem
    // C is an incidence vector
    double diag_buff[n];
    accum_vec(mesh_loc, diag_inv, diag_buff, comm);
    
    // Alg. 6.6, line 3: d_inv := {1/d_i}, i = 1,...,n
    for(index i = 0; i < n; i++){
        diag_inv[i] = omega / diag_buff[i]; // compute D^-1 and save it in diag_inv
    }

    // Alg. 6.6, line 5: r := b - A * u
    // calculating the residuum locally
    double r[n];
    blasl1_dcopy(b,r,(index) n,1.);         //copy b into r (r=b)
    sed_spmv_adapt(A,u,r,-1.0);             //Solution in vector r
    
    // set residuum to 0 at dirichlet bcs
    inc_dir_r(r, fixed, nfixed);

    // Alg. 6.6, line 6: w := sum(C^T_s * r_s), s = 1,...,P
    // accumulated version of the residuum
    // P is the amount of processors on which we split our problem
    // C is an incidence vector
    double w[n];
    accum_vec(mesh_loc, r, w, comm);

    // Alg. 6.6, line 7: sigma := sigma_0 := <w,r>
    // computing the scalar product of w and r
    double sigma_0 = ddot_parallel(w,r,n,comm);
    double sigma = sigma_0;

    // initializing the loop variable
    size_t k = 0;

    do {
        
        // increasing loop variable
        k++;

        // Alg. 6.6, line 11: u_k := u_k-1 + omega * diag(A)^-1 .* w
        // .* is an element-wise multiplication
        for(index i = 0; i < n; i++){
            u[i] += diag_inv[i] * w[i]; // compute diag_inv .* w and save it in w
        }        
        //blasl1_daxpy(u,w,n,omega,1.0); // u <- u + w * omega

		// dirichlet nodes 0 
		inc_dir_r(u, fixed, nfixed);

        // Alg. 6.6, line 12: r := b - A * u
        // calculating the residuum locally
        blasl1_dcopy(b,r,(index) n,1.);  //copy b in r (r=b)
        sed_spmv_adapt(A,u,r,-1.0);
        
        // set residuum to 0 at dirichlet bcs
        inc_dir_r(r, fixed, nfixed);

        // Alg. 6.6, line 13: w := sum(C^T_s * r_s), s = 1,...,P
        // accumulated version of the residuum
        // P is the amount of processors on which we split our problem
        // C is an incidence vector
        accum_vec(mesh_loc, r, w, comm);
     
        // Alg. 6.6, line 14: sigma := sigma_0 := <w,r>
        // computing the scalar product of w and r   
        sigma = ddot_parallel(w,r,n,comm);
        // printf("k = %d \t norm = %10g\n", k, sqrt(sigma));

    } while (sqrt(sigma) > tol);
    
    // write dirichlet data in solution vector
    inc_dir_u(u, dir, fixed, nfixed);
}
