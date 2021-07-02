#ifndef COMMON_H
#define COMMON_H

#include <iostream>

#include <glm/glm.hpp>
//#include <glm/common.hpp>
#include <glm/vec3.hpp>
//#include <glm/fwd.hpp>
//#include <glm/geometric.hpp>


static void HandleError( cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cout << "Error Name: " << cudaGetErrorName( err ) << std::endl;
    std::cout << cudaGetErrorString( err ) << " in " << file << " line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))


constexpr float infinity = std::numeric_limits<float>::max();


// Aliases
using vec3   = glm::vec3;
using point3 = glm::vec3;
using color  = glm::vec3;

#endif
