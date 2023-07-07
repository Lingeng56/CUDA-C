//
// Created by 林庚 on 2021/6/1.
//
#include <stdio.h>
#include <cuda_runtime.h>
#include "../tool.h"
int cpuRecursiveReduce(int *data,const int size){
    if (size==1){
        return data[0];
    }
    int const stride =size/2;

    for (int i=0; i<stride;i++)
    {
        data[i]+=data[i+stride];
    }

    return cpuRecursiveReduce(data,stride);
}

__global__ void gpuRecursiveReduce(int *g_idata,int *g_odata,const int iSize){

    unsigned int tid=threadIdx.x;
    int *idata=g_idata+blockIdx.x*blockDim.x;
    int *odata=g_odata+blockIdx.x;

    if (iSize==2 && tid==0){
        g_odata[blockIdx.x]=idata[0]+idata[1];
        return;
    }
    int iStride=iSize>>1;
    if (iStride>1 && tid<iStride) {
        idata[tid] += idata[tid + iStride];
    }



    __syncthreads();

    if (tid==0){
        gpuRecursiveReduce<<<1,iStride>>>(idata,odata,iStride);

        cudaDeviceSynchronize();
    }

    __syncthreads();

}

int main(int argc,char* argv[]){
    int dev=0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s starting .... at ",argv[0]);
    printf("device %d: %s\n",dev,deviceProp.name);
    cudaSetDevice(dev);

    int size=32;
    int blocksize=32;
    int gridsize=(size+blocksize-1)/blocksize;
    size_t nBytes=size*sizeof(int);
    int *hData,*diData,*doData,*gpuRef,cpuSum=0.0,gpuSum=0.0;
    double start,finish,elapsed;
    if (argc>1){
        gridsize=atoi(argv[1]);
        blocksize=(size+gridsize-1)/gridsize;
    }

    dim3 block(blocksize,1);
    dim3 grid(gridsize,1);

    hData=(int *)malloc(nBytes);
    gpuRef=(int *)malloc(grid.x*sizeof(int));
    check(cudaMalloc((void**)&diData,nBytes));
    check(cudaMalloc((void**)&doData,grid.x*sizeof(int)));
    for (int i=0;i<size;i++){
        *(hData)=i;
    }
    check(cudaMemcpy(diData,hData,nBytes,cudaMemcpyHostToDevice));

    start=cpu_time();
    cpuSum=cpuRecursiveReduce(hData,size);
    finish=cpu_time();
    elapsed=finish-start;
    printf("calculate %d numbers,cpu cost %lf sc. result is %d\n",size,elapsed,cpuSum);

    start=cpu_time();
    gpuRecursiveReduce<<<grid,block>>>(diData,doData,block.x);
    check(cudaDeviceSynchronize());
    check(cudaGetLastError());
    finish=cpu_time();
    elapsed=finish-start;
    check(cudaMemcpy(gpuRef,doData,grid.x*sizeof(int),cudaMemcpyDeviceToHost));
    for (int i=0;i<grid.x;i++){
        gpuSum+=gpuRef[i];
    }

    printf("calculate %d numbers,gpu cost %lf sc. result is %d\n",size,elapsed,gpuSum);
    return 0;
}
