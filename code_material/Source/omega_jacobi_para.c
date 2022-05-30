// omega Jacobi algorithm parallel

// TODO: Zeile 26, 46, 51, 79, 84 :)

#include "hpc.h"

void
omega_jacobi(size_t n, const sed *A, const double *b, double *u, double omega, double tol) {
    // n     - Amount of columns of A (also length of most vectors in the algorithm)
    // A     - Part of a stiffness matrix (sed Format!)
    // b     - Part of the righthand side
    // u     - Part of the inital guess for the solution
    // omega - often 2/3
    // tol   - Toleranz (stopping criteria)

    double *Ax = A->x; // data of A

    // Alg. 6.6, line 1: d := diag(A)
    double diag_inv[n];
    blasl1_dcopy(Ax,diag_inv,(index) n,1.); // the diagonal of A is now in vector diag_inv

    // Alg. 6.6, line 2: d := sum(C^T_s * d_s), s = 1,...,P
    // accumulated version of the diagonal
    // P is the amount of processors on which we split our problem
    // C is an incidence matrix
    // TODO: OLIS AKKUMULATIONS FUNKTION, input: diag_inv
    // bitte in diag_inv speichern (bzw wie und wo wird gespeichert?)

    // Alg. 6.6, line 3: d_inv := {1/d_i}, i = 1,...,n
    for(index i = 0; i < n; i++){
        diag_inv[i] = 1 / diag_inv[i]; // compute D^-1 and save it in diag_inv
    }

    // Alg. 6.6, line 5: r := b - A * u
    // calculating the residuum locally
    double r[n];
    blasl1_dcopy(b,r,(index) n,1.);         //copy b into r (r=b)
    sed_spmv_adapt(A,u,r,-1.0);             //Solution in vector r

    // Alg. 6.6, line 6: w := sum(C^T_s * r_s), s = 1,...,P
    // accumulated version of the residuum
    // P is the amount of processors on which we split our problem
    // C is an incidence matrix
    double w[n];
    blasl1_dcopy(r,w,(index) n,1.);         // copy r into w (w=r)
    // TODO: OLIS AKKUMULATIONS FUNKTION, input: w
    // bitte in w speichern :)

    // Alg. 6.6, line 7: sigma := sigma_0 := <w,r>
    // computing the scalar product of w and r
    double sigma_0 = // OLIS SKALARPRODUKT FUNKTION
    double sigma = sigma_0;

    // initializing the loop variable
    size_t k = 0;

    do {
        
        // increasing loop variable
        k++;

        // Alg. 6.6, line 11: u_k := u_k-1 + omega * diag(A)^-1 .* w
        // .* is an element-wise multiplication
        for(index i = 0; i < n; i++){
            w[i] = diag_inv[i] * w[i]; // compute diag_inv .* w and save it in w
        }        
        blasl1_daxpy(u,w,n,omega,1.0); // u <- u + w * omega

        // Alg. 6.6, line 12: r := b - A * u
        // calculating the residuum locally
        blasl1_dcopy(b,r,(index) n,1.);  //copy b in r (r=b)
        sed_spmv_adapt(A,u,r,-1.0);

        // Alg. 6.6, line 13: w := sum(C^T_s * r_s), s = 1,...,P
        // accumulated version of the residuum
        // P is the amount of processors on which we split our problem
        // C is an incidence matrix
        blasl1_dcopy(r,w,(index) n,1.);         // copy r into w (w=r)
        // TODO: OLIS AKKUMULATIONS FUNKTION, input: w
        // bitte in w speichern :)

        // Alg. 6.6, line 14: sigma := sigma_0 := <w,r>
        // computing the scalar product of w and r
        sigma = // OLIS SKALARPRODUKT FUNKTION
               
        printf("k = %d \t norm = %10g\n", k, sqrt(sigma));

    } while (sqrt(sigma) > tol);
        
}