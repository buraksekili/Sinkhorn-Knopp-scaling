#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#define DEBUG 0

void wrapper(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int* nov, int* nnz,
             int siter);

void* read_mtxbin(std::string bin_name, int*& adj, int*& xadj, int*& tadj, int*& txadj, double*& rv,
                  double*& cv, int*& nov, int*& nnz) {
  const char* fname = bin_name.c_str();
  FILE* bp;
  bp = fopen(fname, "rb");

  nov = new int;
  nnz = new int;

  fread(nov, sizeof(int), 1, bp);
  fread(nnz, sizeof(int), 1, bp);

  std::cout << "READ-nov: " << nov << "\t*nov: " << *nov << "\t&nov: " << &nov << std::endl;
  std::cout << "READ-nnz: " << nnz << "\t*nnz: " << *nnz << "\t&nnz: " << &nnz << std::endl;

  adj = new int[*nnz];
  xadj = new int[*nov];
  tadj = new int[*nnz];
  txadj = new int[*nov];

  fread(adj, sizeof(int), *nnz, bp);
  fread(xadj, sizeof(int), *nov + 1, bp);

  fread(tadj, sizeof(int), *nnz, bp);
  fread(txadj, sizeof(int), *nov + 1, bp);

  std::cout << "Read the binary file" << std::endl;

  rv = (double*)malloc(*nov * sizeof(double));
  cv = (double*)malloc(*nov * sizeof(double));

  if (DEBUG) {
    std::cout << "##################" << std::endl;
    std::cout << "Binary Read Report" << std::endl;
    std::cout << "nov: " << *nov << std::endl;
    std::cout << "nnz: " << *nnz << std::endl;

    for (int i = 0; i < *nov + 1; i++) {
      std::cout << "i: " << i << "  xadj[i]: " << xadj[i] << std::endl;
    }

    for (int i = 0; i < *nnz; i++) {
      std::cout << "i: " << i << "  adj[i]: " << adj[i] << std::endl;
    }

    std::cout << "Binary Read Report" << std::endl;
    std::cout << "##################" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  std::string fname = argv[1];
  std::cout << "fname: " << fname << std::endl;
  int siter = atoi(argv[2]);

  int* adj;
  int* xadj;
  int* tadj;
  int* txadj;
  double* rv;
  double* cv;
  int* nov;
  int* nnz;

  read_mtxbin(fname, adj, xadj, tadj, txadj, rv, cv, nov, nnz);
  wrapper(adj, xadj, tadj, txadj, rv, cv, nov, nnz, siter);
}
