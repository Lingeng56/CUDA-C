CMAKE_MINIMUM_REQUIRED(VERSION 3.5.1)
# 项目名称
PROJECT(cuda_work)
# cmake寻找cuda，这个要现在系统里面装好cuda，设置好cuda的环境参数啥的
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.1/)
FIND_PACKAGE(CUDA REQUIRED)
# C++和CUDA的编译参数，可选。

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

# 头文件路径，按需

# 库文件路径，按需

# 主要就是这个，教cmake去找nvcc来编译这些东西
CUDA_ADD_EXECUTABLE(ad-census cuda_c_ch1/main.cu)
cuda_add_executable(nestedHello cuda_c_ch3/nestedHello.cu)
CUDA_ADD_EXECUTABLE(checkThreadIndex cuda_c_ch2/checkThreadIndex.cu)
# 链接外部库，按需




cuda_add_executable(checkDevice cuda_c_ch2/checkDevice.cu)
cuda_add_executable(sumArraysOnGPU-timer cuda_c_ch2/sumArraysOnGPU-timer.cu)
cuda_add_executable(simpleDivergence cuda_c_ch3/simpleDivergence.cu)
cuda_add_executable(sumArraysZeroCopy cuda_c_ch4/sumArraysZeroCopy.cu)
add_executable(tool tool.cpp)