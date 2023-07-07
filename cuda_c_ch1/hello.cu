//
// Created by 林庚 on 2021/5/26.
//
#include <stdio.h>

__global__ void helloFromGPU(){
    printf("Hello WorLd from GPU thread %d!\n",threadIdx.x);
}
int main(){
    printf("Hello World from CPU !\n");
    helloFromGPU<<<1,10>>>();
    cudaDeviceSynchronize();
}
