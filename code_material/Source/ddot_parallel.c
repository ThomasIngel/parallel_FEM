#include "hpc.h"
#include "mesh_trans.h"
#include <mpi.h>
#include <unistd.h>

double* get_local_ddot(double* m_i, double* r_i, index nloc, double* local_ddot){
  // Berechnet Skalarprodukt aus m_i und r_i
  // Return als Pointer für einfacheres MPI_Allreduce handling
  local_ddot[0] = 0;
  for(int i=0;i<nloc;i++){
    local_ddot[0] += m_i[i]*r_i[i];
  }
  return local_ddot;
}

double ddot_parallel(double* m_i, double* r_i, index nloc, MPI_Comm comm){
  // Berechnet akkumuliertes Skalarprodukt über alle Prozesse und teilt es mit allen
  double ddot[1];
  double local_ddot[1];
  MPI_Allreduce(
    get_local_ddot(m_i, r_i, nloc,local_ddot),
    ddot,
    1,
    MPI_DOUBLE,
    MPI_SUM,
    comm);
  return ddot[0];
}