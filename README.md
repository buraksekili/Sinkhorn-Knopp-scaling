Parallel Sinkhorn-Knopp scaling algorithm is used to convert
a given matrix to the doubly stochastic form.

## Installation
```shell
git clone https://github.com/buraksekili/Sinkhorn-Knopp-scaling.git
```


The required version of the `gcc` is `8.2.0` for the CPU code `main.cpp`. 
You must change the version of gcc by running; 
```shell
$ module load gcc/8.2.0
```
For GPU code, `cuda/10.0` and `gcc/7.5.0` is required. 
You must change the version of gcc and CUDA by running; 
```shell
$ module load gcc/7.5.0
$ module load cuda/10.0
```

### GPU 
```shell
$ cd ./gpu
$ nvcc kernel.cu main.cpp -O3 -Xcompiler -fopenmp
$ ./a.out ./cage15.mtxbin 5 4
```
where `5` is number of iterations and `4` is number of threads.

### CPU 
```shell
$ cd ./cpu
$ g++ main.cpp -fopenmp -O3
$ ./a.out ./cage15.mtxbin 5
```
where `5` is number of iterations. 