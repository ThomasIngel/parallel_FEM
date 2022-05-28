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

void* get_rhs_global(index* c, double* r, index nloc, double* rhs_loc, double* rhs_glob, index n){
  MPI_Allreduce(
    make_global(c,r,rhs_loc,nloc),
    rhs_glob,
    n,
    MPI_DOUBLE,
    MPI_SUM,
    MPI_COMM_WORLD);
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
  get_rhs_global(c,r,nloc,rhs_loc,rhs_glob,n);
  printf("\nrhs_loc = ");
  for(int i=0;i<n;i++) printf("%f ", rhs_loc[i]);
  printf("\nrhs_glob = ");
  for(int i=0;i<n;i++) printf("%f ", rhs_glob[i]);
  double ddot[1];
  double ddot_loc[1];
  MPI_Allreduce(
    ddot_local(c,r,rhs_glob,nloc,ddot_loc),
    ddot,
    1,
    MPI_DOUBLE,
    MPI_SUM,
    MPI_COMM_WORLD);
  return ddot[0];
}

double get_sigma2(index* c, double* r, index nloc, index n){
  double rhs_loc[n];
  double rhs_glob[n];
  get_rhs_global(c,r,nloc,rhs_loc,rhs_glob,n);
  printf("\nrhs_loc = ");
  for(int i=0;i<n;i++) printf("%f ", rhs_loc[i]);
  printf("\nrhs_glob = ");
  for(int i=0;i<n;i++) printf("%f ", rhs_glob[i]);
  double ddot = blasl1_ddot(rhs_glob,rhs_glob,n);
  return ddot;
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

  sleep(myid);

  printf("\n---Process %d---\n",myid);
  printf("\nc: ");
  for(int i=0;i<3;i++) printf("%d ",c[i]);
  printf("\nr: ");
  for(int i=0;i<3;i++) printf("%f ",r[i]);
  printf("\n");

  double sigma = get_sigma2(c,r,nloc,n);
  printf("\nsigma = %f",sigma);
  printf("\n");

  MPI_Finalize();
  return 0;
}
  


