//
// Created by 林庚 on 2021/8/27.
//
#include <cuda_runtime.h>
#include <stdio.h>
#include "../tool.h"
#include <sys/time.h>

double cpu_time(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double )tp.tv_sec+(double )tp.tv_usec*1.e-6);
}

void sumArraysOnHost(float* A,float *B,float *C,const int N){
    for (int idx=0;idx<N;idx++){
        C[idx]=A[idx]+B[idx];
    }
}
void initialData(float *ip,int size){
    time_t t;
    srand((unsigned ) time(&t));
    for (int i=0;i<size;i++){
    ip[i]=(float)(rand() & 0xFF)/10.0f;
    }
}
void checkResult(float *hostRef,float *gpuRef,const int N){
    double epsilon=1.0E-8;
    bool match=1;
    for (int i=0;i<N;i++){
        if (abs(hostRef[i]-gpuRef[i])>epsilon){
            match=0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
            break;
        }
    }
    if (match){
        printf("Arrays match.\n\n");
    }
}




__global__ void sumArraysZeroCopy(float *A,float *B,float *C,const int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N){
        C[i]=A[i]+B[i];
    }
}

int main(int argc,char** argv){
    int dev=0;
    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);

    if (!deviceProp.canMapHostMemory){
        printf("Device %d does not support mapping CPU host memory\n",dev);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
    printf("Using Device %d : %s",dev,deviceProp.name);

    int ipower=10;
    if (argc>1){
        ipower=atoi(argv[1]);
    }
    int nElem=1<<ipower;
    size_t nBytes=nElem*sizeof(float );
    if (ipower<18){
        printf("Vector sie %d power %d nbytes %3.0f KB\n",nElem,ipower,(float)nBytes/1024.0f);
    } else{
        printf("Vector sie %d power %d nbytes %3.0f KB\n",nElem,ipower,(float)nBytes/(1024.0f*1024.0f));

    }

    float *h_A,*h_B,*hostRef,*gpuRef;
    double start,elapsed;
    h_A=(float *)malloc(nBytes);
    h_B=(float *)malloc(nBytes);
    hostRef=(float *)malloc(nBytes);
    gpuRef=(float *)malloc(nBytes);

    initialData(h_A,nElem);
    initialData(h_B,nElem);
    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);
    start=cpu_time();
    sumArraysOnHost(h_A,h_B,hostRef,nElem);
    elapsed=cpu_time()-start;
    printf("sumArraysOnHost cost %f secs\n",elapsed);

    float *d_A,*d_B,*d_C;
    cudaMalloc((float **)&d_A,nBytes);
    cudaMalloc((float **)&d_B,nBytes);
    cudaMalloc((float **)&d_C,nBytes);

    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);

    int iLen=512;
    dim3 block(iLen);
    dim3 grid((nElem+block.x-1)/block.x);

    start=cpu_time();
    sumArraysZeroCopy<<<grid,block>>>(d_A,d_B,d_C,nElem);
    elapsed=cpu_time()-start;
    printf("sumArraysOnGPU cost %f secs\n",elapsed);
    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
    checkResult(hostRef,gpuRef,nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    unsigned int flags=cudaHostAllocMapped;
    cudaHostAlloc((void**)&h_A,nBytes,flags);
    cudaHostAlloc((void**)&h_B,nBytes,flags);

    initialData(h_A,nElem);
    initialData(h_B,nElem);
    memset(hostRef,0,nBytes);
    memset(gpuRef,0,nBytes);

    cudaHostGetDevicePointer((void**)&d_A,(void*)h_A,0);
    cudaHostGetDevicePointer((void**)&d_B,(void*)h_B,0);


    start=cpu_time();
    sumArraysOnHost(h_A,h_B,hostRef,nElem);
    elapsed=cpu_time()-start;
    printf("sumArraysOnHost cost %f secs\n",elapsed);


    start=cpu_time();
    sumArraysZeroCopy<<<grid,block>>>(d_A,d_B,d_C,nElem);
    elapsed=cpu_time()-start;
    printf("sumArraysZeroCopy cost %f secs\n",elapsed);
    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
    checkResult(hostRef,gpuRef,nElem);

    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);

    free(gpuRef);
    free(hostRef);

    cudaDeviceReset();
    return EXIT_SUCCESS;


}

