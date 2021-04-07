// file esempio tantiprint
#include "stdio.h"

__global__ void miokernel(void){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Sono il thread %d!\n", tid);
}

int main() {
  miokernel<<<2,32>>>();
  printf("Hello World!\n");
  return 0;
}
