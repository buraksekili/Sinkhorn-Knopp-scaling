// Burak Sekili

//#include "scale.h"
#include <math.h> /* fabs */
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

int ntInt;

int parallel_sk(int* xadj, int* adj, int* txadj, int* tadj, double* rv, double* cv, int nov,
                int iter) {
  int adjlen = xadj[nov];

  double start = omp_get_wtime();
#pragma omp parallel for schedule(static) num_threads(ntInt)
  for (int i = 0; i < nov; i++) {
    rv[i] = 1;
    cv[i] = 1;
  }

  omp_set_num_threads(ntInt);
  for (int x = 0; x < iter; x++) {
    int starti, endi;
#pragma omp parallel for schedule(guided) private(starti, endi)
    for (int i = 0; i < nov; i += 1) {
      starti = xadj[i];
      endi = xadj[i + 1];
      double rsum = 0;
      for (int j = starti; j < endi; j++) {
        rsum += 1 * cv[adj[j]];
      }
      rv[i] = 1 / rsum;
    }

    int startj, endj;
#pragma omp parallel for schedule(guided) private(startj, endj)
    for (int j = 0; j < nov; j++) {
      startj = txadj[j];
      endj = txadj[j + 1];
      double csum = 0;
      for (int i = startj; i < endj; i++) {
        csum += 1 * rv[tadj[i]];
      }
      cv[j] = 1 / csum;
    }
    double maxerr = 0;

#pragma omp parallel for schedule(guided) private(starti, endi)
    for (int i = 0; i < nov; i++) {
      starti = xadj[i];
      endi = xadj[i + 1];
      double err = 0;
      double res = 0;
      for (int j = starti; j < endi; j++) {
        res = rv[i] * cv[adj[j]];
        err += res;
      }
      double finalErr = fabs(1 - err);
      if (finalErr > maxerr) {
        maxerr = finalErr;
      }
    }
    std::cout << "iter " << x << " - error: " << maxerr << std::endl;
  }

  double end = omp_get_wtime();
  std::cout << ntInt << " Threads  --  "
            << "Time: " << end - start << " s." << std::endl;

  return 1;
}

void read_mtxbin(std::string bin_name, int iter) {
  const char* fname = bin_name.c_str();
  FILE* bp;
  bp = fopen(fname, "rb");

  int* nov = new int;
  int* nnz = new int;

  fread(nov, sizeof(int), 1, bp);
  fread(nnz, sizeof(int), 1, bp);

  int* adj = new int[*nnz];
  int* xadj = new int[*nov];
  int* tadj = new int[*nnz];
  int* txadj = new int[*nov];

  fread(adj, sizeof(int), *nnz, bp);
  fread(xadj, sizeof(int), *nov + 1, bp);

  fread(tadj, sizeof(int), *nnz, bp);
  fread(txadj, sizeof(int), *nov + 1, bp);

  int inov = *nov + 1;

  double* rv = new double[inov];
  double* cv = new double[inov];

  parallel_sk(xadj, adj, txadj, tadj, rv, cv, *nov, iter);  // or no_col
}

int main(int argc, char* argv[]) {
  char* fname = argv[1];
  char* siter = argv[2];
  char* nt = argv[3];

  std::cout << "fname: " << fname << std::endl;

  ntInt = atoi(nt);
  int iter = atoi(siter);

  read_mtxbin(fname, iter);
  return 0;
}
