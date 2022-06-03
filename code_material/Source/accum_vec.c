#include "hpc.h"
#include "mesh_trans.h"
#include <mpi.h>
#include <unistd.h>

void accum_vec(mesh_trans* mesh_loc, double* r_loc, double* m_i, MPI_Comm comm) {
  
  MPI_Barrier(comm);

	int myid;
  int i;
	MPI_Status stat;
	MPI_Comm_rank(comm,&myid); /* and this processes' rank is */
 
  for(i=0;i<mesh_loc->ncoord_loc;i++) m_i[i] = r_loc[i];
  
  MPI_Barrier(comm);

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
        MPI_Recv(data, length, MPI_DOUBLE, mesh_loc->neighbours[i], MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
        // CROSSPOINTS EAST
        if(i==1){
          m_i[1] += data[0];
          m_i[2] += data[1];
        }
        // CROSSPOINTS WEST
        if(i==3){
          m_i[0] += data[0];
          m_i[3] += data[1];
        }
        // EDGENODES
        pos = 4;
        for(k=0;k<i;k++) pos += mesh_loc->n_single_bdry[k];
        for(k=0;k<mesh_loc->n_single_bdry[i];k++){
          m_i[pos+k] += data[2+k];
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
          data[0] = m_i[1]; data[1] = m_i[2];
          m_i[1] = 0; m_i[2] = 0;
        }
        // CROSSPOINT WEST
        if(i==3){
          data[0] = m_i[0]; data[1] = m_i[3];
          m_i[0] = 0; m_i[3] = 0;
        }
        // EDGENODES
        pos = 4;
        for(int k=0;k<i;k++) pos += mesh_loc->n_single_bdry[k];
        for(int k=0;k<mesh_loc->n_single_bdry[i];k++){
          data[2+k] = m_i[pos+k];
          m_i[pos+k] = 0;
        } 
        MPI_Send(data, length, MPI_DOUBLE, mesh_loc->neighbours[i], 0, comm);
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
        MPI_Recv(data, length, MPI_DOUBLE, mesh_loc->neighbours[i], MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
        // DATEN ANPASSEN
        if(i==0){                 // AUS SOUTH
          // CROSSPOINTS
          m_i[0] += data[0];
          m_i[1] += data[1];
        }else if(i==1){           // AUS EAST
          // CROSSPOINTS
          m_i[2] += data[0];
          m_i[1] += data[1];
        }else if(i==2){           // AUS NORTH
          // CROSSPOINTS
          m_i[3] += data[0];
          m_i[2] += data[1];
        }else if(i==3){           // AUS WEST
          // CROSSPOINTS
          m_i[3] += data[0];
          m_i[0] += data[1];
        }
        // EDGENODES
        pos = 4;
        for(k=0;k<i;k++) pos += mesh_loc->n_single_bdry[k];
        for(k=0;k<mesh_loc->n_single_bdry[i];k++){
          m_i[pos+k] += data[2+k];
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
          data[0] = m_i[0];
          data[1] = m_i[1];
        }else if(i==1){           // AN EAST
          // CROSSPOINTS
          data[0] = m_i[2];
          data[1] = m_i[1];
        }else if(i==2){           // AN NORTH
          // CROSSPOINTS
          data[0] = m_i[3];
          data[1] = m_i[2];
        }else if(i==3){           // AN WEST
          // CROSSPOINTS
          data[0] = m_i[3];
          data[1] = m_i[0];
        }
        pos = 4;
        for(k=0;k<i;k++) pos += mesh_loc->n_single_bdry[k];
        for(k=0;k<mesh_loc->n_single_bdry[i];k++){
          data[2+k] = m_i[pos+k];
        } 
        MPI_Send(data, length, MPI_DOUBLE, mesh_loc->neighbours[i], 0, comm);
      }
    }
  }

  MPI_Barrier(comm);
  
  double rand;
  // RED SCHICKT CROSSPOINTS AN NACHBARN EAST/WEST
  // WENN KEIN NACHBAR WEST/EAST, DANN ZUSÄTZLICH DIE RANDPUNKTE NACH NORTH/SOUTH
  if(mesh_loc->black){    // BLACK RECEIVED
    // EAST
    if(mesh_loc->neighbours[1] > -1){ 
      // printf("\nMYID = %d, BEKOMME EAST VON %d",myid,mesh_loc->neighbours[1]);
      MPI_Recv(data, 2, MPI_DOUBLE, mesh_loc->neighbours[1], MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
      m_i[2] = data[0]; m_i[1] = data[1];
    }else{  // KEINE DOMAIN IN EAST
      // RECV VON SOUTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[0] > -1){
        // printf("\nMYID = %d, WARTE FÜR SOUTH VON %d",myid,mesh_loc->neighbours[0]);
        MPI_Recv(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[0], MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
        m_i[1] = rand;
      }
      
      // RECV VON NORTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[2] > -1){
        // printf("\nMYID = %d, WARTE FÜR NORTH VON %d",myid,mesh_loc->neighbours[2]);
        MPI_Recv(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[2], MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
        m_i[2] = rand;
      }
    }
    // WEST
    if(mesh_loc->neighbours[3] > -1){ 
      // printf("\nMYID = %d, BEKOMME WEST VON %d",myid,mesh_loc->neighbours[3]);
      MPI_Recv(data, 2, MPI_DOUBLE, mesh_loc->neighbours[3], MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
      m_i[3] = data[0]; m_i[0] = data[1];
    } else{  // KEINE DOMAIN IN WEST
      // RECV VON SOUTH, FALLS VORHANDEN
      // THIS -----------------------------------
      if(mesh_loc->neighbours[0] > -1){
        // printf("\nMYID = %d, WARTE FÜR SOUTH VON %d",myid,mesh_loc->neighbours[0]);
        MPI_Recv(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[0], MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
        m_i[0] = rand;
      }
      // ----------------
      // RECV VON NORTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[2] > -1){
        // printf("\nMYID = %d, WARTE FÜR NORTH VON %d",myid,mesh_loc->neighbours[2]);
        MPI_Recv(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[2], MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
        m_i[3] = rand;
      }
    }
  }else{                  // ROT SENDET
    // EAST
    if(mesh_loc->neighbours[1] > -1){ 
      data[0] = m_i[2]; data[1] = m_i[1];
      // printf("\nMYID = %d, SCHICKE EAST AN %d",myid,mesh_loc->neighbours[1]);
      MPI_Send(data, 2, MPI_DOUBLE, mesh_loc->neighbours[1], 0, comm);
    }else{  // KEINE DOMAIN IN EAST
      // SCHICK SOUTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[0] > -1){
        rand = m_i[1];
        // printf("\nMYID = %d, SCHICKE SOUTH AN %d",myid,mesh_loc->neighbours[0]);
        MPI_Send(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[0], 0, comm);
      }
      // SCHICK NORTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[2] > -1){
        rand = m_i[2];
        // printf("\nMYID = %d, SCHICKE NORTH AN %d",myid,mesh_loc->neighbours[2]);
        MPI_Send(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[2], 0, comm);
      }
    }
    // WEST
    if(mesh_loc->neighbours[3] > -1){ 
      data[0] = m_i[3]; data[1] = m_i[0];
      // printf("\nMYID = %d, SCHICKE WEST AN %d",myid,mesh_loc->neighbours[3]);
      MPI_Send(data, 2, MPI_DOUBLE, mesh_loc->neighbours[3], 0, comm);
    }else{  // KEINE DOMAIN IN WEST
      // SCHICK SOUTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[0] > -1){
        rand = m_i[0];
        // printf("\nMYID = %d, SCHICKE SOUTH AN %d",myid,mesh_loc->neighbours[0]);
        MPI_Send(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[0], 0, comm);
      }
      // THIS-----------------------------------
      // SCHICK NORTH, FALLS VORHANDEN
      if(mesh_loc->neighbours[2] > -1){
        rand = m_i[3];
        // printf("\nMYID = %d, SCHICKE SOUTH AN %d",myid,mesh_loc->neighbours[2]);
        MPI_Send(&rand, 1, MPI_DOUBLE, mesh_loc->neighbours[2], 0, comm);
      }
      // ----------------------------------------
    }
  }
  
  MPI_Barrier(comm);
}