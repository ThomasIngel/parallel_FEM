#include "hpc.h"
#include "mesh_trans.h"

void get_dirichlet(mesh_trans* mesh_loc, double (*f_dir)(double *), 
		double* dir){
	// gather variables for readability
	index nfixed = mesh_loc->nfixed_loc;
	index* fixed = mesh_loc->fixed_loc;
	double* coord = mesh_loc->domcoord;
	
	// calculate dirichlet bcs
	double x[2];
	for (index i = 0; i < nfixed; ++i){
		x[0] = coord[2 * fixed[i]];
		x[1] = coord[2 * fixed[i] + 1];
		dir[i] = f_dir(x);
	}
}
