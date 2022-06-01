#include "hpc.h"
#include "mesh_trans.h"
#include <mpi.h>
#include <unistd.h>

double* make_global(index* c,double* r,double* rhs_loc,index nlocal){
  for(int i=0;i<nlocal;i++){
    rhs_loc[c[i]] = r[i];
  }
  return rhs_loc;
}

double kappa( double x[2], index typ )
{
  return ( 1.0 );
}

double F_vol( double x[2], index typ )
{
  return ( 4.0 );
}

double g_Neu( double x[2], index typ )
{
  // return ( x[0] * x[1] );
  return 0.0;
}

double u_D( double x[2])
{
//  return ( 0.0 );
  return ( x[0] * x[1] );
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
 
  double omega = 2.0/3.0;
  double tol = 1e-10;
  mesh_trans **metra;
  index anz_dom = numprocs;
  index ncoords;  

  if (myid == 0){
    mesh* H = get_refined_mesh(1);
    ncoords = H->ncoord;
    
    // double* b1 = calloc(ncoords, sizeof(double));  
    // mesh_RHS(H, b1, F_vol, g_Neu); 
    /*printf("\nProcessor %d rhs full mesh:\n", myid);
    for(i=0;i<ncoords;i++){
      printf("%lg\n",b1[i]);
    }*/

    metra = malloc ( (anz_dom) * sizeof(mesh_trans));

    for(size_t i=0;i<anz_dom;i++){
      metra[i]=alloc_mesh_trans(anz_dom,ncoords);
    }

    meshsplit(H, metra, anz_dom);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&ncoords,1,MPI_INT,0,MPI_COMM_WORLD);

  mesh_trans* mesh_loc =  scatter_meshes(metra,MPI_COMM_WORLD,anz_dom,ncoords);
  sed* S;
  S = malloc (sizeof(sed));
  S = sed_sm_build(mesh_loc);
  double* b = calloc(mesh_loc->ncoord_loc, sizeof(double));
  mesh_trans_rhs(mesh_loc,b,F_vol, g_Neu);

  sleep(myid);
  /*
  printf("\nProcessor %d S: ", myid);
  sed_print(S,1);*/
    
  printf("\nProcessor %d rhs: ", myid);
  for(i=0;i<mesh_loc->ncoord_loc;i++) printf("%f ",b[i]);
  printf("\nProcessor %d c: ", myid);
  for(i=0;i<mesh_loc->ncoord_loc;i++) printf("%d ",mesh_loc->c[i]);
  // printf("\nProcessor %d neighbours: ", myid);
  // for(i=0;i<4;i++) printf("%d ",mesh_loc->neighbours[i]);
  // printf("\nProcessor %d nedgenodes: %d", myid,mesh_loc->nedgenodes);
  // printf("\nProcessor %d n_single_bdry: ", myid);
  // for(i=0;i<4;i++) printf("%d ",mesh_loc->n_single_bdry[i]);
  // printf("\nProcessor %d black: %d", myid,mesh_loc->black);
  printf("\n");
  
  MPI_Barrier(MPI_COMM_WORLD);

  double* u = calloc(mesh_loc->ncoord_loc, sizeof(double));
  
  int change = 0;
  if(change==0){
    omega_jacobi(mesh_loc->ncoord_loc, S, b, u, omega, tol, mesh_loc, MPI_COMM_WORLD);
  }
  if(change==1){
    cg_parallel(S, b, u, tol, mesh_loc, MPI_COMM_WORLD);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  sleep(myid);
  printf("\nProcessor %d lokales Ergebnis: ", myid);
  for(i=0;i<mesh_loc->ncoord_loc;i++) printf("%f ",u[i]);
  printf("\n");

  double vec_loc[ncoords];
  double accum[ncoords];
  MPI_Reduce(
    make_global(mesh_loc->c,u,vec_loc,mesh_loc->ncoord_loc),
    accum,
    ncoords,
    MPI_DOUBLE,
    MPI_SUM,
    0,
    MPI_COMM_WORLD);

  if(myid == 0){
    sleep(1);
    printf("\nProcessor %d globales Ergebnis: ", myid);
    for(i=0;i<ncoords;i++) printf("%f ",accum[i]);
    printf("\n"); 
  }

  MPI_Finalize();
  return 0;
}
  


