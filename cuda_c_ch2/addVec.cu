//
// Created by 林庚 on 2021/5/26.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


__global__ void sumArraysOnDevice(float *A,float *B,float *C){
    int i=threadIdx.x;
    C[i]=A[i]+B[i];
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

void checkResult(float *hostRef,float *gpuRef,const int N){
    int match=1;
    for (int i=0;i<N;i++){
        if (hostRef[i]!=gpuRef[i]){
            match=0;
            printf("Not Match at index %d!\n",i);
            break;
        }
    }
    if (match){
        printf("Array Match!\n");
    }
}

int main(int argc,char* argv[]){
    int nElem=4;
    size_t nBytes=nElem*sizeof (float );
    float *h_A,*h_B,*h_C,*Ref;
    h_C=(float *)malloc(nBytes);
    Ref=(float *)malloc(nBytes);
    h_A=(float *)malloc(nBytes);
    h_B=(float *)malloc(nBytes);

    initialData(h_A,nElem);
    initialData(h_B,nElem);

    sumArraysOnHost(h_A,h_B,h_C,nBytes);

    float *d_A,*d_B,*d_C;
    cudaMalloc((void **)&d_A,nBytes);
    cudaMalloc((void **)&d_B,nBytes);
    cudaMalloc((void **)&d_C,nBytes);
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);


    sumArraysOnDevice<<<1,nElem>>>(d_A,d_B,d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(Ref,d_C,nBytes,cudaMemcpyDeviceToHost);
    checkResult(h_C,Ref,nElem);


    free(h_A);
    free(h_B);
    free(h_C);
    free(Ref);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);



    cudaDeviceReset();

    return 0;
}

