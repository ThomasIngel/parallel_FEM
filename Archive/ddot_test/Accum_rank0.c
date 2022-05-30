#include "hpc.h"
#include "mesh_trans.h"
#include <mpi.h>
#include "blas_level1.h"

double* make_global(index* c,double* r,double* rhs_loc,index nlocal){
  for(int i=0;i<nlocal;i++){
    rhs_loc[c[i]] = r[i];
  }
  return rhs_loc;
}

double* accum_vec(index* c, double* vec, double* accum, double* vec_loc, index nloc, index n){
  MPI_Allreduce(
    make_global(c,vec,vec_loc,nloc),
    accum,
    n,
    MPI_DOUBLE,
    MPI_SUM,
    MPI_COMM_WORLD);
  return accum;
}

double* ddot_local(index* c, double* r, double* rhs_glob, index nloc, double* ddot_loc){
  ddot_loc[0] = 0;
  for(int i=0;i<nloc;i++){
    ddot_loc[0] += rhs_glob[c[i]]*r[i];
  }
  return ddot_loc;
}

double get_sigma(index* c, double* r, index nloc, index n){
  double rhs_loc[n];
  double rhs_glob[n];
  double vec_loc[nloc];
  /*printf("\nrhs_loc = ");
  for(int i=0;i<n;i++) printf("%f ", rhs_loc[i]);
  printf("\nrhs_glob = ");
  for(int i=0;i<n;i++) printf("%f ", rhs_glob[i]);*/
  double ddot[1];
  double ddot_loc[1];
  MPI_Allreduce(
    ddot_local(c,r,accum_vec(c,r,rhs_glob,vec_loc,nloc,n),nloc,ddot_loc),
    ddot,
    1,
    MPI_DOUBLE,
    MPI_SUM,
    MPI_COMM_WORLD);
  free(rhs_loc);
  free(rhs_glob);
  free(vec_loc);
  free(ddot_loc);
  return ddot[0];
}

int main(int argc, char *argv[]) {

  int numprocs;
	int myid;
	MPI_Status stat;

  MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs); /* find out how big the SPMD world is */
	MPI_Comm_rank(MPI_COMM_WORLD,&myid); /* and this processes' rank is */

  index n = 2*numprocs+1;
  index nloc = 3;
  index c[3] = { myid*2, myid*2+1, myid*2+2 };
  double r[3] = { 1, 1, 1 };

  sleep(myid*2);

  printf("\n---Process %d---\n",myid);
  printf("\nc: ");
  for(int i=0;i<3;i++) printf("%d ",c[i]);
  printf("\nr: ");
  for(int i=0;i<3;i++) printf("%f ",r[i]);
  printf("\n");

  double sigma = get_sigma(c,r,nloc,n);
  if(myid==0){
  printf("\nsigma = %f",sigma);
  printf("\n");
  }

  MPI_Finalize();
  return 0;
}
  


