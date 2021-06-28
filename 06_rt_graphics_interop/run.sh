#!/bin/sh

if [ "$1" == "-b" ]; then
  rm -rf ./build ./main
fi

if [ -d "./build" ]; then
  cd build
  make
  cd ..
fi

if [ ! -f "./main" ]; then
  echo "Building..."
  rm -rf build main
  mkdir build
  cd build
  cmake -D CMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc ..
  #cmake -D CUDAToolkit_ROOT=/opt/cuda/ -D CMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc ..
  #cmake -DCMAKE_CUDA_FLAGS=”-arch=sm_50” ..
  #cmake -DCMAKE_CUDA_FLAGS=”-gencode arch=compute_50,code=sm_50” ..

  make
  cd ..
fi
./main
