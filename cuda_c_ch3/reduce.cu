//
// Created by 林庚 on 2021/5/29.
//
#include <stdio.h>
#include "../tool.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

int recursiveReduce(int *data,int const size){

    if (size==1){
        return data[0];
    }
    int const stride =size/2;

    for (int i=0; i<stride;i++)
    {
        data[i]+=data[i+stride];
    }

    return recursiveReduce(data,stride);
}

__global__ void reduceNeighbored(int *g_idata,int *g_odata,unsigned int n)
{
    unsigned int tid=threadIdx.x;
    unsigned int idx=blockIdx.x * blockDim.x+threadIdx.x;

    int *idata=g_idata+blockIdx.x*blockDim.x;

    if (idx >= n){
        return ;
    }

    for (int stride=1;stride<blockDim.x;stride*=2)
    {
        if ((tid%(2*stride))==0)
        {
            idata[tid]+=idata[tid+stride];
        }
        __syncthreads();
    }

    if (tid==0){
        g_odata[blockIdx.x]=idata[0];
    }
}

__global__ void reduceNeighboredLess(int *g_idata,int *g_odata,unsigned int n)
{
    unsigned int tid=threadIdx.x;
    unsigned int idx=blockIdx.x*blockDim.x+threadIdx.x;

    int *idata=g_idata+blockIdx.x*blockDim.x;

    if (idx>n){
        return ;
    }
    for (int stride=1;stride<blockDim.x;stride*=2){
        int index=2*tid*stride;
        if (index<blockDim.x){
            idata[index]+=idata[index+stride];
        }
        __syncthreads();
    }

    if (tid==0){
        g_odata[blockIdx.x]=idata[0];
    }
}

int main(int argc,char* argv[]){
    int dev=0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s starting reduction at ",argv[0]);
    printf("device %d: %s \n",dev,deviceProp.name);
    cudaSetDevice(dev);



    int size=1<<12;
    printf("    with array size %d    ",size);

    int blocksize=512;
    if (argc>1){
        blocksize=atoi(argv[1]);
    }
    dim3 block(blocksize,1);
    dim3 grid((size+block.x-1)/block.x,1);
    printf("grid %d block %d\n",grid.x,block.x);

    size_t nBytes=size*sizeof(int);
    int *h_idata=(int *)malloc(nBytes);
    int *h_odata=(int *)malloc(grid.x*sizeof(int));
    int *tmp=(int*)malloc(nBytes);

    for (int i=0;i<size;i++){
        h_idata[i]=(int)(rand()&0xFF);
    }
    memcpy(tmp,h_idata,nBytes);

    double iStart,iFinish,iElapsed;
    int gpu_sum=0;

    int *d_idata=NULL;
    int *d_odata=NULL;
    cudaMalloc((void **) &d_idata,nBytes);
    cudaMalloc((void **) &d_odata,grid.x*sizeof(int));

    GET_TIME(iStart);
    int cpu_sum=recursiveReduce(tmp,size);
    GET_TIME(iFinish);
    iElapsed=iFinish-iStart;
    printf("cpu reduce      elapsed %lf sc cpu_sum: %d\n",iElapsed,cpu_sum);

    cudaMemcpy(d_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    GET_TIME(iStart);
    reduceNeighbored<<<grid,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    GET_TIME(iFinish);
    iElapsed=iFinish-iStart;
    cudaMemcpy(h_odata,d_odata,grid.x*sizeof(int),cudaMemcpyDeviceToHost);
    for (int i=0;i<grid.x;i++){
        gpu_sum+=h_odata[i];
    }
    printf("gpu Neighbored      elapsed %lf sc gpu_sum: %d\n",iElapsed,gpu_sum);



    cudaMemcpy(d_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gpu_sum=0;
    GET_TIME(iStart);
    reduceNeighboredLess<<<grid,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    GET_TIME(iFinish);
    iElapsed=iFinish-iStart;
    cudaMemcpy(h_odata,d_odata,grid.x*sizeof(int),cudaMemcpyDeviceToHost);
    for (int i=0;i<grid.x;i++){
        gpu_sum+=h_odata[i];
    }
    printf("gpu NeighboredLess      elapsed %lf sc gpu_sum: %d\n",iElapsed,gpu_sum);


    free(h_odata);
    free(h_idata);
    cudaFree(d_idata);
    cudaFree(d_odata);
}

