cmake_minimum_required(VERSION 3.9)
project(CUDA_Test LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")

add_compile_options(-Wall -Wextra) # -pedantic) # -Werror)
add_compile_options(-Wno-deprecated-gpu-targets)

#include_directories(include)
file(GLOB SOURCES "*.cu")

add_executable(main ${SOURCES})

# build with -std=c++11
target_compile_features(main PUBLIC cxx_std_11)

# add -gencode arch=compute_50,code=sm_50
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 50)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)



## More or less equivalent to
# nvcc -std=c++11 main.cu -gencode arch=compute_50,code=sm_50 -o main -Wno-deprecated-gpu-targets
