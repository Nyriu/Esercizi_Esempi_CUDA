cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(RT_Interop LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
 
include(CTest)
 
find_package(CUDA REQUIRED)
find_package(glfw3 REQUIRED)
#set(OpenGL_GL_PREFERENCE "LEGACY")
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)

add_compile_options(-g -Wall -Wextra ) # -pedantic) # -Werror)

#add_library(particles STATIC
#  randomize.cpp
#  randomize.h
#  particle.cu
#  particle.h
#  v3.cu
#  v3.h
#  )


include_directories(include)

file(GLOB SOURCES "src/*.c*")
#file(GLOB SOURCES "main.cu")

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
#target_compile_features(particles PUBLIC cxx_std_11)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
#set_target_properties( particles
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")

add_executable(main ${SOURCES})
#set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 50 75)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 75)
set_target_properties( main
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main glfw GL)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50") # works but bad solution
# TODO make work the one below and remove the one above
#target_compile_options(particle_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:"-gencode arch=compute_50,code=sm_50">) 

 
#set_property(TARGET particle_test 
#             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(particle_test PRIVATE particles)


 
if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  #set_property(TARGET particle_test 
  #             PROPERTY
  #             BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
