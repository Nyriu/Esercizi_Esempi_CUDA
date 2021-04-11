#!/bin/sh

if [ "$1" == "-b" ]; then
  rm -rf ./build ./main ./particle_test
fi

if [ -d "./build" ]; then
  cd build
  make
  cd ..
fi

if [ ! -f "./main" ]; then
  echo "Building..."
  rm -rf build main particle_test
  mkdir build
  cd build
  cmake ..
  #cmake -DCMAKE_CUDA_FLAGS=”-arch=sm_50” ..
  #cmake -DCMAKE_CUDA_FLAGS=”-gencode arch=compute_50,code=sm_50” ..

  make
  cd ..
fi
./particle_test
