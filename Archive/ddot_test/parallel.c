#include "hpc.h"
#include "mesh_trans.h"
#include <mpi.h>
#include <unistd.h>

double kappa( double x[2], index typ )
{
  return ( 1.0 );
}

double F_vol( double x[2], index typ )
{
  return ( 0.0 );
}

double g_Neu( double x[2], index typ )
{
  return ( x[0] * x[1] );
}

double u_D( double x[2])
{
//  return ( 0.0 );
  return ( x[0] * x[1] );
}

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
  // sed* S;
  // S = malloc (sizeof(sed));
  // S = sed_sm_build(mesh_loc);
  double* b = calloc(mesh_loc->ncoord_loc, sizeof(double));
    // mesh_trans_rhs(mesh_loc,b,F_vol, g_Neu);

  // EASIER VERGLEICHEN 
  for(i=0;i<mesh_loc->ncoord_loc;i++) b[i] = 1;
    
  printf("\nProcessor %d rhs: ", myid);
  for(i=0;i<mesh_loc->ncoord_loc;i++) printf("%f ",b[i]);
  printf("\nProcessor %d neighbours: ", myid);
  for(i=0;i<4;i++) printf("%d ",mesh_loc->neighbours[i]);
  printf("\nProcessor %d nedgenodes: %d", myid,mesh_loc->nedgenodes);
  printf("\nProcessor %d n_single_bdry: ", myid);
  for(i=0;i<4;i++) printf("%d ",mesh_loc->n_single_bdry[i]);
  printf("\nProcessor %d black: %d", myid,mesh_loc->black);
  printf("\n");
  
  double m_i[mesh_loc->ncoord_loc];
  MPI_Barrier(MPI_COMM_WORLD);
  accum_vec(mesh_loc, b, m_i, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  sleep(myid);
  printf("\nProcessor %d rhs: ", myid);
  for(i=0;i<mesh_loc->ncoord_loc;i++) printf("%f ",m_i[i]);
  printf("\n");

  MPI_Barrier(MPI_COMM_WORLD);
  double ddot = ddot_parallel(m_i,b,mesh_loc->ncoord_loc,MPI_COMM_WORLD);
  printf("\nDDOT = %f",ddot);

  MPI_Finalize();
  return 0;
}
  


