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
  
  MPI_Barrier(MPI_COMM_WORLD);

  // ------------ GET LOCAL m_i ------------- 
  double east[2];
  double west[2];
  int length;
  int pos;
  int max_edgenodes = 0;
  for(i=0;i<4;i++){
    if(mesh_loc->n_single_bdry[i]>max_edgenodes){
      max_edgenodes = mesh_loc->n_single_bdry[i];
    }
  }
  double data[2+max_edgenodes];
  int k;
  // RED SENDET CROSSPOINT-INFO AN NACHBARN EAST/WEST, SETZT CROSSPOINT AUF 0
  // RED SENDET EDGENODE-INFO AN ALLE NACHBARN, SETZT EDGENODE AUF 0
  // BLACK ADDIERT CROSSPOINTS AUF
  // BLACK ADDIERT EDGENODES AUF
  if(mesh_loc->black){    //BLACK RECEIVED VON ALLEN RED EDGENODES
    for(i=0;i<4;i++){
      if(mesh_loc->neighbours[i] > -1){
        length = 2 + mesh_loc->n_single_bdry[i];
        MPI_Recv(data, length, MPI_DOUBLE, mesh_loc->neighbours[i], MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // CROSSPOINTS EAST
        if(i==1){
          b[1] += data[0];
          b[2] += data[1];
        }
        // CROSSPOINTS WEST
        if(i==3){
          b[0] += data[0];
          b[3] += data[1];
        }
        // EDGENODES
        pos = 4;
        for(k=0;k<i;k++) pos += mesh_loc->n_single_bdry[k];
        for(k=0;k<mesh_loc->n_single_bdry[i];k++){
          b[pos+k] += data[2+k];
        }  
      }
    }
  }else{    //RED SENDET EDGENODES AN ALLE NACHBARN UND CROSSPOINTS AN EAST/WEST
    // EDGENODES
    for(i=0;i<4;i++){
      if(mesh_loc->neighbours[i] > -1){
        length = 2 + mesh_loc->n_single_bdry[i];
        // CROSSPOINT EAST
        if(i==1){
          data[0] = b[1]; data[1] = b[2];
          b[1] = 0; b[2] = 0;
        }
        // CROSSPOINT WEST
        if(i==3){
          data[0] = b[0]; data[1] = b[3];
          b[0] = 0; b[3] = 0;
        }
        // EDGENODES
        pos = 4;
        for(int k=0;k<i;k++) pos += mesh_loc->n_single_bdry[k];
        for(int k=0;k<mesh_loc->n_single_bdry[i];k++){
          data[2+k] = b[pos+k];
          b[pos+k] = 0;
        } 
        MPI_Send(data, length, MPI_DOUBLE, mesh_loc->neighbours[i], 0, MPI_COMM_WORLD);
      }
    }
  }

  // WAIT FOR BLACK
  /*
  sleep(myid);
  printf("\nProcessor %d rhs: ", myid);
  for(i=0;i<mesh_loc->ncoord_loc;i++) printf("%f ",b[i]);
  printf("\n");
  */
  
  // REDS AKKUM MIT BLACK NACHBARN
  if(!mesh_loc->black){           // REDS RECEIVEN DATA VON ALLEN BLACK NACHBARGEBIETEN
    for(i=0;i<4;i++){
      if(mesh_loc->neighbours[i] > -1){
        length = 2 + mesh_loc->n_single_bdry[i];
        // double* data =  (double*) malloc(length*sizeof(double));      
        // DATA RECEIVE [CROSSP CROSSP (EDGENODE) (...)]
        // CROSSPOINTS VON LINKS NACH RECHTS ODER OBEN NACH UNTEN
        MPI_Recv(data, length, MPI_DOUBLE, mesh_loc->neighbours[i], MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // DATEN ANPASSEN
        if(i==0){                 // AUS SOUTH
          // CROSSPOINTS
          b[0] += data[0];
          b[1] += data[1];
        }else if(i==1){           // AUS EAST
          // CROSSPOINTS
          b[2] += data[0];
          b[1] += data[1];
        }else if(i==2){           // AUS NORTH
          // CROSSPOINTS
          b[3] += data[0];
          b[2] += data[1];
        }else if(i==3){           // AUS WEST
          // CROSSPOINTS
          b[3] += data[0];
          b[0] += data[1];
        }
        // EDGENODES
        pos = 4;
        for(k=0;k<i;k++) pos += mesh_loc->n_single_bdry[k];
        for(k=0;k<mesh_loc->n_single_bdry[i];k++){
          b[pos+k] += data[2+k];
        }  
      }
    }
  }else{                        // BLACKS SENDEN DATA AN ALLE RED NACHBARN
    for(i=0;i<4;i++){
      if(mesh_loc->neighbours[i] > -1){ 
        length = 2 + mesh_loc->n_single_bdry[i];
        // double* data = (double*)  malloc(length*sizeof(double));
        if(i==0){                 // AN SOUTH
          // CROSSPOINTS
          data[0] = b[0];
          data[1] = b[1];
        }else if(i==1){           // AN EAST
          // CROSSPOINTS
          data[0] = b[2];
          data[1] = b[1];
        }else if(i==2){           // AN NORTH
          // CROSSPOINTS
          data[0] = b[3];
          data[1] = b[2];
        }else if(i==3){           // AN WEST
          // CROSSPOINTS
          data[0] = b[3];
          data[1] = b[0];
        }
        pos = 4;
        for(k=0;k<i;k++) pos += mesh_loc->n_single_bdry[k];
        for(k=0;k<mesh_loc->n_single_bdry[i];k++){
          data[2+k] = b[pos+k];
        } 
        MPI_Send(data, length, MPI_DOUBLE, mesh_loc->neighbours[i], 0, MPI_COMM_WORLD);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  double rand;
  // RED SCHICKT CROSSPOINTS AN NACHBARN EAST/WEST
  // WENN KEIN NACHBAR WEST/EAST, DANN ZUSÄTZLICH DIE RANDPUNKTE NACH NORTH/SOUTH
  if(mesh_loc->black){    // BLACK RECEIVED
    // EAST
    if(mesh_loc->neighbours[1] > -1){ 
      // printf("\nMYID = %d, BEKOMME EAST VON %d",myid,mesh_loc->neighbours[1]);
      MPI_Recv(data, 2, MPI_DOUBLE, mesh_loc->neighbours[1], MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      b[2] = data[0]; b[1] = data[1];
    }else{  // KEINE DOMAIN IN EAST
      // RECV VON SOUTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[0] > -1){
        // printf("\nMYID = %d, WARTE FÜR SOUTH VON %d",myid,mesh_loc->neighbours[0]);
        MPI_Recv(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[0], MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        b[1] = rand;
      }
      
      // RECV VON NORTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[2] > -1){
        // printf("\nMYID = %d, WARTE FÜR NORTH VON %d",myid,mesh_loc->neighbours[2]);
        MPI_Recv(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[2], MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        b[2] = rand;
      }
    }
    // WEST
    if(mesh_loc->neighbours[3] > -1){ 
      // printf("\nMYID = %d, BEKOMME WEST VON %d",myid,mesh_loc->neighbours[3]);
      MPI_Recv(data, 2, MPI_DOUBLE, mesh_loc->neighbours[3], MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      b[3] = data[0]; b[0] = data[1];
    } else{  // KEINE DOMAIN IN WEST
      // RECV VON SOUTH, FALLS VORHANDEN
      // THIS -----------------------------------
      if(mesh_loc->neighbours[0] > -1){
        // printf("\nMYID = %d, WARTE FÜR SOUTH VON %d",myid,mesh_loc->neighbours[0]);
        MPI_Recv(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[0], MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        b[0] = rand;
      }
      // ----------------
      // RECV VON NORTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[2] > -1){
        // printf("\nMYID = %d, WARTE FÜR NORTH VON %d",myid,mesh_loc->neighbours[2]);
        MPI_Recv(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[2], MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        b[3] = rand;
      }
    }
  }else{                  // ROT SENDET
    // EAST
    if(mesh_loc->neighbours[1] > -1){ 
      data[0] = b[2]; data[1] = b[1];
      // printf("\nMYID = %d, SCHICKE EAST AN %d",myid,mesh_loc->neighbours[1]);
      MPI_Send(data, 2, MPI_DOUBLE, mesh_loc->neighbours[1], 0, MPI_COMM_WORLD);
    }else{  // KEINE DOMAIN IN EAST
      // SCHICK SOUTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[0] > -1){
        rand = b[1];
        // printf("\nMYID = %d, SCHICKE SOUTH AN %d",myid,mesh_loc->neighbours[0]);
        MPI_Send(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[0], 0, MPI_COMM_WORLD);
      }
      // SCHICK NORTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[2] > -1){
        rand = b[2];
        // printf("\nMYID = %d, SCHICKE NORTH AN %d",myid,mesh_loc->neighbours[2]);
        MPI_Send(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[2], 0, MPI_COMM_WORLD);
      }
    }
    // WEST
    if(mesh_loc->neighbours[3] > -1){ 
      data[0] = b[3]; data[1] = b[0];
      // printf("\nMYID = %d, SCHICKE WEST AN %d",myid,mesh_loc->neighbours[3]);
      MPI_Send(data, 2, MPI_DOUBLE, mesh_loc->neighbours[3], 0, MPI_COMM_WORLD);
    }else{  // KEINE DOMAIN IN WEST
      // SCHICK SOUTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[0] > -1){
        rand = b[0];
        // printf("\nMYID = %d, SCHICKE SOUTH AN %d",myid,mesh_loc->neighbours[0]);
        MPI_Send(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[0], 0, MPI_COMM_WORLD);
      }
      // THIS-----------------------------------
      // SCHICK NORTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[2] > -1){
        rand = b[3];
        // printf("\nMYID = %d, SCHICKE SOUTH AN %d",myid,mesh_loc->neighbours[2]);
        MPI_Send(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[2], 0, MPI_COMM_WORLD);
      }
      // ----------------------------------------
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  sleep(myid);
  printf("\nProcessor %d rhs: ", myid);
  for(i=0;i<mesh_loc->ncoord_loc;i++) printf("%f ",b[i]);
  printf("\n");


  MPI_Finalize();
  return 0;
}
  


