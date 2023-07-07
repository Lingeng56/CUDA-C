//
// Created by 林庚 on 2021/8/23.
//
#include <stdio.h>
#include <cuda_runtime.h>
#include "../tool.h"

__global__ void mathKernel1(float* c){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    float a,b;
    a=b=0.0f;
    if (tid%2==0){
        a=100.0f;
    } else{
        b=200.0f;
    }
    c[tid]=a+b;
}

__global__ void mathKernel2(float* c){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    float a,b;
    a=b=0.0f;
    if ((tid/warpSize)%2==0){
        a=100.0f;
    } else{
        b=200.0f;
    }
    c[tid]=a+b;
}

int main(int argc,char** argv){
    int dev=0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s using Device %d: %s\n",argv[0],dev,deviceProp.name);

    int size=64;
    int blocksize=64;
    if (argc>1) blocksize=atoi(argv[1]);
    if (argc>2) size=atoi(argv[2]);
    printf("Data size %d ",size);

    dim3 block(blocksize,1);
    dim3 grid((size+block.x-1)/block.x,1);
    printf("Execution Configure (block %d grid %d)\n",block.x,grid.x);

    float *d_C;
    size_t nBytes=size* sizeof(float );
    cudaMalloc((float **)d_C,nBytes);
    double iStart,iElaps;
    GET_TIME(iStart)
    mathKernel1<<<grid,block>>>(d_C);
    GET_TIME(iElaps)
    iElaps=iElaps-iStart;
    printf("mathKernel1<<<%4d,%4d>>> cost %f sec\n",grid.x,block.x,iElaps);
    GET_TIME(iStart)
    mathKernel2<<<grid,block>>>(d_C);
    GET_TIME(iElaps)
    iElaps=iElaps-iStart;
    printf("mathKernel2<<<%4d,%4d>>> cost %f sec\n",grid.x,block.x,iElaps);


}
