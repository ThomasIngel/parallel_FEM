#include "hpc.h"
#include "blas_level1.h"
#include "mesh_trans.h"

void
omega_jacobi_seriell(size_t n,
           const sed *A, const double *b, double *u, double omega, double tol) {
        // A     - stiffness matrix (sed Format!)
        // b     - righthand side
        // u     - inital guess for solution
        // omega - often 2/3
        // tol   - Toleranz (stopping criteria)

        double *Ax = A->x; // data of the matrix A

        double diag[n];
        blasl1_dcopy(Ax,diag,(index) n,1.); // the diagonal of A is now in vector diag_inv
        
        // For later calculation purposes
        double tmp[n];

        double r[n];
        blasl1_dcopy(b,r,(index) n,1.);         //copy b in r (r=b)

        // r = b - A*u , calculating the residuum
        sed_spmv_adapt(A,u,r,-1.0);             //Solution in vector r

        // sigma = r'*r = sigma_0 , computing the scalarproduct
        double sigma_0 = blasl1_ddot(r,r,n);
        double sigma = sigma_0;

        // initializing the loop variable
        size_t k = 0;

        do {
                // increasing loop variable
                k++;

                // u_k := u_k-1 + omega * diag(A)^-1 * r
                for(index i =0;i<n;i++){
                        r[i] = r[i] / diag[i]; // compute D^-1 * r and save it in r
                }
                
                blasl1_daxpy(u,r,n,omega,1.0); // u <- u + r * omega


                blasl1_dcopy(b,r,(index) n,1.);  //copy b in r (r=b)

                // r = b - A*u , calculating the residuum
                sed_spmv_adapt(A,u,r,-1.0);

                // sigma = r' * r , computing the scalarproduct with the new residuum
                sigma = blasl1_ddot(r,r,n);

        } while (sqrt(sigma) > tol);
}