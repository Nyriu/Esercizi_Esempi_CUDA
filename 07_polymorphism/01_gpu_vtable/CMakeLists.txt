cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(Esempio LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

 
include(CTest)
 
find_package(CUDA REQUIRED)


add_compile_options(-g -Wall -Wextra ) # -pedantic) # -Werror)
add_compile_options(-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored)

include_directories(include)
file(GLOB SOURCES "src/*.c*")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")

add_executable(main ${SOURCES})
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 75)
set_target_properties( main
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
  )
