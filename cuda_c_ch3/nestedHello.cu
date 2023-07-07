//
// Created by 林庚 on 2021/6/1.
//
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "../tool.h"

__global__ void nestedHelloWorld(int const iSize,int iDepth){
    int tid=threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n",iDepth,tid,blockIdx.x);

    if (iSize==1){
        return;
    }

    int nthreads=iSize>>1;
    if (tid+blockIdx.x*blockDim.x==0 && nthreads>0){
        nestedHelloWorld<<<2,nthreads>>>(nthreads,++iDepth);
        printf("--------> nested execution depth: %d\n",iDepth);
    }
}
int main(int argc,char* argv[]){
    int dev=0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s starting .... at ",argv[0]);
    printf("device %d: %s",dev,deviceProp.name);
    cudaSetDevice(dev);

    int father_size=8;
    int blocksize=8;
    int gridsize=1;

    if (argc>1){
        gridsize=atoi(argv[1]);
        father_size=gridsize*blocksize;
    }
    dim3 block(blocksize,1);
    dim3 grid((father_size+block.x-1)/block.x,1);
    printf("Execution Coniguration: grid %d block %d",grid.x,block.x);
    nestedHelloWorld<<<grid,block>>>(block.x,0);

    check(cudaGetLastError());
    check(cudaDeviceReset());

    return 0;
}
