cmake_minimum_required(VERSION 3.0)

project(Simple_SphereTracer)

set(CMAKE_CXX_FLAGS "-g")
#set(CMAKE_CXX_FLAGS "-g -O3 -Wextra -std=c++11")

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include_directories(include)

#file(GLOB SOURCES "src/*.c*")
file(GLOB SOURCES "main.c")
#set_source_files_properties(main.c PROPERTIES LANGUAGE CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")

add_executable(main ${SOURCES})

target_link_libraries(main
  -lcuda
  #-lcudart
  #-lpthread
  )
