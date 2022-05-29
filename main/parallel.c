#include "hpc.h"
#include "mesh_trans.h"
#include <mpi.h>

void* make_global(index *c,index nlocal,double *rhs_loc,double *rhs_glob){
  for(int i=0;i<nlocal;i++){
    rhs_glob[c[i]] = rhs_loc[i];
  }
}

void* make_local(index *c,index nlocal,double *rhs_glob,double *rhs_loc){
  for(int i=0;i<nlocal;i++){
    rhs_loc[i] = rhs_glob[c[i]] ;
  }
}

int main(int argc, char *argv[]) {

  int numprocs;
	int myid;
	int i;
  const int root=0;
	MPI_Status stat;

  MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs); /* find out how big the SPMD world is */
	MPI_Comm_rank(MPI_COMM_WORLD,&myid); /* and this processes' rank is */

  mesh_trans **metra;
  index anz_dom = 2;
  index ncoord = 9;
  if (myid == 0){
    
    mesh* H = get_refined_mesh(1);

    double* b1 = calloc(ncoord, sizeof(double));  
    mesh_RHS(H, b1, F_vol, g_Neu); 
    printf("\nProcessor %d rhs full mesh:\n", myid);
    for(int i=0;i<ncoord;i++){
      printf("%lg\n",b1[i]);
    }

    metra = malloc ( (anz_dom) * sizeof(mesh_trans));

    for(size_t i=0;i<anz_dom;i++){
      metra[i]=alloc_mesh_trans(anz_dom,ncoord);
    }

    meshsplit(H, metra, anz_dom);
  }
  mesh_trans* mesh_loc =  scatter_meshes(metra,MPI_COMM_WORLD,anz_dom,ncoord);

  sed* S;
  S = malloc (sizeof(sed));
  S = sed_sm_build(mesh_loc);
  double* b = calloc(mesh_loc->ncoord_loc, sizeof(double));
  mesh_trans_rhs(mesh_loc,b,F_vol, g_Neu);

  sleep(myid);

  printf("\nProcessor %d rhs: ", myid);
  for(int i=0;i<mesh_loc->ncoord_loc;i++) printf("%f ",b[i]);
  printf("\nProcessor %d neighbours: ", myid);
  for(int i=0;i<4;i++) printf("%f ",mesh_loc->neighbours[i]);

  MPI_Finalize();
  return 0;
}
  


