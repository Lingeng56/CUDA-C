//
// Created by 林庚 on 2021/5/27.
//
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <sys/time.h>
#include "tool.h"
#include <string.h>
void checkResult(float *hostRef,float *gpuRef,const int N){
    int match=1;
    double epsilon=1.0E-8;
    for (int i=0;i<N;i++){
        if (std::abs(hostRef[i]-gpuRef[i])>epsilon){
            match=0;
            printf("not match happened at index %d ,in hostRef is %f and in gpuRef is %f\n",i,hostRef[i],gpuRef[i]);
            break;
        }
    }
    if (match){
        printf("Array Match!\n");
    }
}

__global__ void sumArraysOnDevice(float *A,float *B,float *C,const int N){
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if (i<N) {
        C[i] = A[i] + B[i];
    }
}

void sumArraysOnHost(float *A,float *B,float *C,const int N){
    for (int idx=0;idx<N;idx++){
        C[idx]=A[idx]+B[idx];
    }
}

void initialData(float *ip,int size){
    time_t t;
    srand((unsigned int)time(&t));

    for (int i=0;i<size;i++){
        ip[i]=(float )(rand() & 0xFF)/10.0f;
    }
}



int main(int argc,char **argv){
    printf("%s Starting...\n",argv[0]);

    int dev=0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n",dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nElem=1<<24;
    printf("Vector size %d\n",nElem);

    size_t nBytes=nElem*sizeof(float );

    float *h_A,*h_B,*hostRef,*gpuRef;
    h_A=(float *)malloc(nBytes);
    h_B=(float *)malloc(nBytes);
    hostRef=(float *)malloc(nBytes);
    gpuRef=(float *)malloc(nBytes);

    double iStart,iElaps,cputime;

    GET_TIME(iStart);
    initialData(h_A,nElem);
    initialData(h_B,nElem);
    GET_TIME(iElaps);
    iElaps-=iStart;

    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    GET_TIME(iStart);
    sumArraysOnHost(h_A,h_B,hostRef,nElem);
    GET_TIME(iElaps);
    iElaps-=iStart;
    printf("sumArraysOnHost Time elaped %f sec\n",iElaps);
    cputime=iElaps;

    float *d_A,*d_B,*d_C;
    cudaMalloc((float**)&d_A,nBytes);
    cudaMalloc((float**)&d_B,nBytes);
    cudaMalloc((float**)&d_C,nBytes);

    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);

    int iLen=256;
    dim3 block(iLen);
    dim3 grid((nElem+block.x-1)/block.x);

    GET_TIME(iStart);
    sumArraysOnDevice<<<grid,block>>>(d_A,d_B,d_C,nElem);
    cudaDeviceSynchronize();
    GET_TIME(iElaps);
    iElaps-=iStart;
    printf("sumArraysOnDevice<<<%d, %d>>> Time elaped %f sec\n",grid.x,block.x,iElaps);
    printf("GPU is %f times Faster than CPU\n",cputime/iElaps);

    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
    checkResult(hostRef,gpuRef,nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
