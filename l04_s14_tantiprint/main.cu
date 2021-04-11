// file esempio tantiprint
#include "stdio.h"

__global__ void miokernel(void){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Sono il thread %d!\n", tid);
}

int main() {
  //miokernel<<<2,32>>>();
  miokernel<<<1,8>>>();
  printf("Hello World!\n");
  cudaDeviceSynchronize();
  return 0;
}
