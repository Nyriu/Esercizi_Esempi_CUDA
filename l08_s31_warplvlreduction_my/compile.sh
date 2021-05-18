#!/bin/sh
rm -rf build main
mkdir build
cd build
cmake ..
make
cd ..
