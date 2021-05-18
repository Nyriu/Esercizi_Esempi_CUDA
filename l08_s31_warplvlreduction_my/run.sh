#!/bin/sh

# remove comment when CMake works
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
  cmake ..
  make
  cd ..
fi
./main



## ## delete below when CMake works
## if [ "$1" == "-b" ]; then
##   rm -rf ./main
## fi
## make
## ./main
