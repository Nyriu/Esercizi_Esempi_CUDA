#include <__clang_cuda_device_functions.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mallocTest() {
  size_t size = 123;
  char *ptr = (char*) malloc(size);
  memset(ptr, 0, size);
  printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
  free(ptr);
}


int main( void ) {
  // Set heap size of 120Mb
  // Must be done before any kernel si launched
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
  mallocTest<<<1,5>>>();
  cudaDeviceSynchronize();

  return 0;
}
