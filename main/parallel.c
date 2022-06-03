#include "hpc.h"
#include "mesh_trans.h"
#include <mpi.h>
#include <unistd.h>

// PARAMETER INPUT
#include <errno.h>  // for errno
#include <limits.h> // for INT_MIN and INT_MAX
#include <string.h>  // for strlen

// WALLTIME
#include <sys/times.h> 

/* return real time in seconds since start of the process */ 
double walltime() { 
    static clock_t ticks_per_second = 0; 
    if (!ticks_per_second) { 
        ticks_per_second = sysconf(_SC_CLK_TCK); 
    }
    struct tms timebuf;
    /* times returns the number of real time ticks passed since start */
    return (double) times(&timebuf) / ticks_per_second;
}

void print_time(double t0){
  double t1 = walltime() - t0;
  printf("\nTIME PASSED: %f",t1);
}

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
  return ( 1.0 );
}

double g_Neu( double x[2], index typ )
{
  // return ( x[0] * x[1] );
  return 0.0;
}

double u_D( double x[2])
{
  return ( 2.0 );
  // return ( x[0] * x[1] );
}

int main(int argc, char *argv[]) {

  int numprocs;
	int myid;
  int i;

  // GET NOREFINE FROM INPUT PARAMETER
  if (strlen(argv[1]) == 0) {
  printf("ERROR WITH REFINEMENT INPUT! ABORTING...\n");
      return 1; // empty string
  }
  char* p;
  errno = 0; // not 'int errno', because the '#include' already defined it
  long arg = strtol(argv[1], &p, 10);
  if (*p != '\0' || errno != 0) {
  printf("ERROR WITH REFINEMENT INPUT! ABORTING...\n");
      return 1; // In main(), returning non-zero means failure
  }

  if (arg < INT_MIN || arg > INT_MAX) {
  printf("ERROR WITH REFINEMENT INPUT! ABORTING...\n");
      return 1;
  }
  int norefine = arg;

  // INITIALIZE MPI
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

  double t0;

  if (myid == 0){
    printf("Starting program with %d mesh refinement(s) on %d processes!\n", norefine,numprocs);

    // START-TIME
    t0 = walltime();

    // CREATE GLOBAL MESH
    mesh* H = get_refined_mesh(norefine);
    ncoords = H->ncoord;

    metra = malloc ( (anz_dom) * sizeof(mesh_trans));

    for(i=0;i<anz_dom;i++){
      metra[i]=alloc_mesh_trans(anz_dom,ncoords);
    }

    // SPLIT MESH
    meshsplit(H, metra, anz_dom);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&ncoords,1,MPI_INT,0,MPI_COMM_WORLD);

  // SCATTER MESH
  mesh_trans* mesh_loc =  scatter_meshes(metra,MPI_COMM_WORLD,anz_dom,ncoords);

  // LOCAL S
  sed* S;
  S = malloc (sizeof(sed));
  S = sed_sm_build(mesh_loc);

  // LOCAL RHS
  double* b = calloc(mesh_loc->ncoord_loc, sizeof(double));
  mesh_trans_rhs(mesh_loc,b,F_vol, g_Neu);
  
  MPI_Barrier(MPI_COMM_WORLD);

  double* u = calloc(mesh_loc->ncoord_loc, sizeof(double));
  
  // HOMOGENITIZE RHS
  double dir_val[mesh_loc->nfixed_loc];
  get_dirichlet(mesh_loc, u_D, dir_val);
  
  inc_dir_u(u, dir_val, mesh_loc->fixed_loc, mesh_loc->nfixed_loc);
  
  sed_spmv_adapt(S, u, b, -1.0);
  
  // SOLVE PROBLEM
  int change = 1;
  if(change==0){
    omega_jacobi(mesh_loc->ncoord_loc, S, b, u, omega, tol, u_D, mesh_loc, MPI_COMM_WORLD);
  }
  if(change==1){
    cg_parallel(S, b, u, tol, u_D, mesh_loc, MPI_COMM_WORLD);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  // PRINT TIME
  if(myid==0){
    print_time(t0);
  }
  sed_free(S);
  free(b);
  
  /*
  sleep(myid);
  printf("\nProcessor %d lokales Ergebnis: ", myid);
  for(i=0;i<mesh_loc->ncoord_loc;i++) printf("%f ",u[i]);
  printf("\n");*/

  // Globalen Lösungsvektor auf rank 0 zusammenstellen
  double* u_loc = calloc(ncoords, sizeof(double));
  make_global(mesh_loc-> c, u, u_loc, mesh_loc->ncoord_loc);
  free(u);
  accum_result(u_loc, ncoords, myid, numprocs, MPI_COMM_WORLD);
  
  // Finale globale Lösung printen
  if(myid == 0){
    printf("\nProcessor %d globales Ergebnis: ", myid);
    for(i=0;i<ncoords;i++) printf("%f ",u_loc[i]);
    printf("\n"); 

    for(i=0;i<anz_dom;i++){
      free_mesh_trans(metra[i]);
    }
    free(metra);
  }
  
  free(u_loc);
  MPI_Finalize();
  return 0;
}
  


