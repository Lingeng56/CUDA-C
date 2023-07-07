#include <stdio.h>
#include <sys/time.h>

#ifndef CUDA_WORK_CHECK_H
#define CUDA_WORK_CHECK_H

#endif //CUDA_WORK_CHECK_H


#define CHECK(call){ \
            const cudaError_t error=call; \
            if(error!=cudaSuccess)        \
            {        \
                printf("Error: %s:%d, ",__FILE__,__LINE__); \
                printf("code:%d, reason: %s\n",error,cudaGetErrorString(error)); \
                exit(1);\
            }\
}


#define GET_TIME(time){ \
    struct timeval tp;  \
    gettimeofday(&tp,NULL);\
    time=((double)tp.tv_sec+(double)tp.tv_usec*1.e-6);\
}


