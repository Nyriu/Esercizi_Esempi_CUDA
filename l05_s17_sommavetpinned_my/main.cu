// vector sum with pinned memory (page-locked)
#include "stdio.h"

#define N 100 // 10 // 32 // 100
#define NumThPerBlock 64 // 32 //256
#define NumBlocks 2 // 1

static void HandleError( cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))


__global__ void add(int *d_a, int *d_b, int *d_c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N)
    d_c[tid] = d_a[tid] + d_b[tid];
  else
    printf("I'm a padding thread tid=%d\n", tid);
}


int main( void ) {
  int *a, *b, *c;     // host variables containing host pointers
  int *dev_a, *dev_b, *dev_c; // host variables containing device pointers

  // pinned memory on host
  cudaSetDeviceFlags(cudaDeviceMapHost);
  HANDLE_ERROR(cudaHostAlloc(&a, N*sizeof(int), cudaHostAllocMapped));
  HANDLE_ERROR(cudaHostAlloc(&b, N*sizeof(int), cudaHostAllocMapped));
  HANDLE_ERROR(cudaHostAlloc(&c, N*sizeof(int), cudaHostAllocMapped));

  // static allocation on device memory
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(int)));

  // host initializes arrays
  for (int i=0; i<N; i++) {
    a[i] = -i;
    b[i] = i * i;
    c[i] = 0;
  }

  // copy arrays from host to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));

  add<<<NumBlocks, NumThPerBlock>>>(dev_a, dev_b, dev_c);

  // wait threads completion
  HANDLE_ERROR(cudaDeviceSynchronize());

  //retrieve the result from device dev_c into c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));

  //show results
  for (int i=0; i<N; i++) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  //free device memory
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}
