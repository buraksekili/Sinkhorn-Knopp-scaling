#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <math.h>       /* fabsf */
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define DEBUG 0

//Error check-----
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}
//Error check-----
//This is a very good idea to wrap your calls with that function.. Otherwise you will not be able to see what is the error.
//Moreover, you may also want to look at how to use cuda-memcheck and cuda-gdb for debugging.

__global__ void calculateError(int* xadj, int* adj, double* rv, double* cv, double* maxErr, int maxOperation) {
	// Get idx for each thread
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < maxOperation) {
		int starti = xadj[i], endi = xadj[i+1];
		double err = 0;
		for (int j = starti; j < endi; j++) err += rv[i] * cv[adj[j]];

		err = fabs(1-err);
		if (err > *maxErr) *maxErr = err;
	} 
}

__global__ void scaleskRV(int* xadj, int* adj, double* rv, double* cv, int maxOperation) {
	// Get idx for each thread
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < maxOperation) {
		int starti = xadj[i], endi = xadj[i+1];
		double rowSum = 0;
		for (int j = starti; j < endi; j++) rowSum += cv[adj[j]];
		rv[i] = 1 / rowSum;	
	}
}


__global__ void scaleskCV(int* txadj, int* tadj, double* rv, double* cv, int maxOperation) {
	// Get idx for each thread
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < maxOperation) {
		int	starti = txadj[i], endi = txadj[i+1];
		double colSum = 0;
		for (int j = starti; j < endi; j++) colSum += rv[tadj[j]];
		cv[i] = 1 / colSum;
	} 
}

void wrapper(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int* nov, int* nnz, int siter){
	// Initialize rv and cv
	for (int i = 0; i < *nov; i++) rv[i] = cv[i] = 1;

	// Transfer data from host to device.
	int NO_THREADS = 1024;
	int NO_BLOCKS = (*nov + NO_THREADS - 1)/NO_THREADS;

	int maxOperation = (*nov) - 1;
	std::cout << "NO_BLOCKS " << NO_BLOCKS << std::endl;
	std::cout << "NO_THREADS " << NO_THREADS << std::endl;
	std::cout << "NO_THREADS * NO_BLOCKS " << NO_THREADS * NO_BLOCKS << std::endl;
	std::cout << "maxOperation " << maxOperation << std::endl;
	std::cout << "no of one " << xadj[*nov] << std::endl;

	int* adj_d, *xadj_d, *tadj_d, *txadj_d;
	gpuErrchk(cudaMalloc( (void**) &adj_d, (*nnz) * sizeof(int)));
	gpuErrchk(cudaMemcpy(adj_d, adj, (*nnz) * sizeof(int), cudaMemcpyHostToDevice ));

	gpuErrchk(cudaMalloc( (void**) &xadj_d, (*nov) * sizeof(int)));
	gpuErrchk(cudaMemcpy(xadj_d, xadj, (*nov) * sizeof(int), cudaMemcpyHostToDevice ));

	gpuErrchk(cudaMalloc( (void**) &tadj_d, (*nnz) * sizeof(int)));
	gpuErrchk(cudaMemcpy(tadj_d, tadj,(*nnz) * sizeof(int), cudaMemcpyHostToDevice ));

	gpuErrchk(cudaMalloc( (void**) &txadj_d, (*nov) * sizeof(int)));
	gpuErrchk(cudaMemcpy(txadj_d, txadj,(*nov) * sizeof(int), cudaMemcpyHostToDevice ));

	double* rv_d, *cv_d;
	gpuErrchk(cudaMalloc( (void**) &rv_d, (*nov) * sizeof(double)));
	gpuErrchk(cudaMemcpy(rv_d, rv, (*nov) * sizeof(double), cudaMemcpyHostToDevice ));

	gpuErrchk(cudaMalloc( (void**) &cv_d, (*nov) * sizeof(double)));
	gpuErrchk(cudaMemcpy(cv_d, cv, (*nov) * sizeof(double), cudaMemcpyHostToDevice ));

	double* err_d;
	double* err = new double(0);

	cudaEvent_t start, stop;
  	cudaEventCreate(&start);
  	cudaEventRecord(start, 0);

	for (int i = 0; i < siter; i++) {
		// Fill rv 
		scaleskRV<<<NO_BLOCKS, NO_THREADS>>>(xadj_d, adj_d, rv_d, cv_d, maxOperation);
		gpuErrchk(cudaPeekAtLastError());

		// Fill cv 
		scaleskCV<<<NO_BLOCKS, NO_THREADS>>>(txadj_d, tadj_d, rv_d, cv_d, maxOperation);
		gpuErrchk(cudaPeekAtLastError());

		// calculate error
		gpuErrchk(cudaMalloc((void**) &err_d, sizeof(double)));
		calculateError<<<NO_BLOCKS, NO_THREADS>>>(xadj_d, adj_d, rv_d, cv_d, err_d, maxOperation);

		// get error from device
		gpuErrchk(cudaMemcpy(err, err_d, sizeof(double), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaFree(err_d));

		std::cout << "iter " << i << " - error: " << *err << std::endl;
	}
  
  	cudaEventCreate(&stop);
  	cudaEventRecord(stop, 0);
  	cudaEventSynchronize(stop);
  
	float elapsedTime;
 	cudaEventElapsedTime(&elapsedTime, start, stop);
 	printf("GPU scale took: %f s\n", elapsedTime/1000);
    
	gpuErrchk(cudaFree(xadj_d));
	gpuErrchk(cudaFree(adj_d));
	gpuErrchk(cudaFree(txadj_d));
	gpuErrchk(cudaFree(tadj_d));
	gpuErrchk(cudaFree(rv_d));
	gpuErrchk(cudaFree(cv_d));
  
}

