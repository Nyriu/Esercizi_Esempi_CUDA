#!/bin/sh

if [ "$1" == "-b" ]; then
  rm -rf ./build ./main
fi

if [ -d "./build" ]; then
  rm -rf main
  cd build
  make
  cd ..
elif [ ! -f "./main" ]; then
  echo "Building..."
  rm -rf build main
  mkdir build
  cd build
  cmake -D CMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc ..
  make
  cd ..
fi
./main
