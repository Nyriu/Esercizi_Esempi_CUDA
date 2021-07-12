#!/bin/sh
rm -rf build main
mkdir build
cd build
cmake -D CMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc ..
make
cd ..
