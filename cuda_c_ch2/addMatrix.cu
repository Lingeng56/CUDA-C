//
// Created by 林庚 on 2021/5/27.
//
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "tool.h"
#include "sys/time.h"
#include <string.h>

/*-------------------------*
 *     Local Function      *
 *-------------------------*/
void sumMatrixOnHost(float *h_A,float *h_B,float *h_C,const int nx,const int ny);
__global__ void sumMatrixOnDevice(float *d_A,float *d_B,float *d_C,const int nx,const int ny);
void initData(float *h_A,int nxy);
void checkResult(float *hostRef,float *gpuRef,int nxy);

/*-------------------------------------------------------------------*/
int main(int argc,char* argv[]){
    printf("%s starting...\n",argv[0]);
    int dev=0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using Device %d: %s\n",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx=1<<13;
    int ny=1<<13;
    int nxy=nx*ny;
    size_t nBytes=nxy*sizeof(float);

    double start,finish,elapsed,cpuTime;

    float *h_A,*h_B,*hostRef,*gpuRef;
    h_A=(float*)malloc(nBytes);
    h_B=(float*)malloc(nBytes);
    hostRef=(float*)malloc(nBytes);
    gpuRef=(float*)malloc(nBytes);

    initData(h_A,nxy);
    initData(h_B,nxy);

    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    GET_TIME(start);
    sumMatrixOnHost(h_A,h_B,hostRef,nx,ny);
    GET_TIME(finish);
    elapsed=finish-start;
    cpuTime=elapsed;
    printf("On CPU, Calculate Cost %f sec\n",elapsed);

    float *d_A,*d_B,*d_C;
    cudaMalloc((void **)&d_A,nBytes);
    cudaMalloc((void **)&d_B,nBytes);
    cudaMalloc((void **)&d_C,nBytes);
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);


    dim3 block(32,1);
    dim3 grid((nx+block.x-1)/block.x);

    GET_TIME(start);
    sumMatrixOnDevice<<<grid,block>>>(d_A,d_B,d_C,nx,ny);
    cudaDeviceSynchronize();
    GET_TIME(finish);
    elapsed=finish-start;
    printf("On Gpu<<<%d, %d>>>, Calculate Cost %f sec\n",grid.x,block.x,elapsed);
    printf("-----------------------------------------\n");
    printf("-----------------------------------------\n");
    printf("GPU Is %f Faster Than CPU\n",cpuTime/elapsed);

    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);

    printf("Checking result....\n");
    checkResult(hostRef,gpuRef,nxy);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
}

/*-------------------------------------------------------------------
 * Function:
 * Purpose:
 * Input args:
 * In/out args:
 */
void sumMatrixOnHost(float *h_A,float *h_B,float *h_C,const int nx,const int ny){
    for (int ix=0;ix<nx;ix++){
        for (int iy=0;iy<ny;iy++){
            h_C[ix*nx+iy]=h_A[ix*nx+iy]+h_B[ix*nx+iy];
        }
    }
}


/*-------------------------------------------------------------------
 * Function:
 * Purpose:
 * Input args:
 * In/out args:
 */
void initData(float *h_A,int nxy){
    for (int i=0;i<nxy;i++){
        h_A[i]=i;
    }
}

/*-------------------------------------------------------------------
 * Function:
 * Purpose:
 * Input args:
 * In/out args:
 */
void checkResult(float *hostRef,float *gpuRef,int nxy){
    bool match= true;
    for (int i=0;i<nxy;i++){
        if (hostRef[i]!=gpuRef[i]){
            match= false;
            printf("Result Is Wrong...\nthe result of CPU at index=%d is %f ,"
                   "which is not same with result of GPU at same index=%f\n",i,hostRef[i],gpuRef[i]);
            break;
        }
    }
    if (match){
        printf("Result is right!\n");
    }
}

/*-------------------------------------------------------------------
 * Function:
 * Purpose:
 * Input args:
 * In/out args:
 */
__global__ void sumMatrixOnDevice(float *d_A,float *d_B,float *d_C,const int nx,const int ny){
    unsigned int my_x=blockIdx.x*blockDim.x+threadIdx.x;
    for (int my_y=0;my_y<ny;my_y++) {
        unsigned int idx = my_y * nx + my_x;
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}