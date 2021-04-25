// Esercizio
// Come il precedente ma 2D
// Dato array 2D di NxM interi e un valore intero x conta quanti sono gli elementi di uguali a x
//
// x e N forniti da linea di comando // TODO

// Variante
// Usare grid 2D // TODO

#include <locale>
#include <stdlib.h>
#include <iostream>
#include <experimental/random>
#include <time.h>

#define N 60 // 3 // Rows
#define M 70 // 4 // Cols
#define THREADS_PER_BLOCK 256
#define MAX_VAL 5 // 50

static void HandleError( cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString( err ) << " in " << file << " line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))

void init_vec(int v[N][M]) {
  for (int i=0; i<N; i++) {
    for (int j=0; j<M; j++) {
      v[i][j] = std::experimental::randint(0,MAX_VAL);
      //std::cout << "i = " << i << "\tj = " << j << std::endl;
      //std::cout << "v[i][j] = "<< v[i][j] << std::endl;
      //v[i][j] = 0;
      //v[i][j] = i*M + j;
      //std::cout << "v[i][j] = "<< v[i][j] << "\n" << std::endl;
    }
  }
}

void show_vec(int v[N][M]) {
  std::cout << "[\n";
  for (int i=0; i<N; i++) {
    for (int j=0; j<M; j++) {
      std::cout << v[i][j] << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "]\n" << std::endl;
}

int cpu_count(int v[N][M], const int x) {
  int c=0;
  for (int i=0; i<N; i++) {
    for (int j=0; j<M; j++) {
      if (v[i][j] == x)
        c++;
    }
  }
  return c;
}

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}


int compute_num_blocks(const int n, const int m) {
  int b = div_ceil(n*m, THREADS_PER_BLOCK);
  return b;
}


__device__ int d_g_count = 0;
__global__ void gpu_count(int *d_v, const int x) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N*M)
    return;


  if (d_v[tid] == x) {
    atomicAdd(&d_g_count, 1);
    printf("%d\t%d - Increment\n", tid, d_v[tid]);
  } else {
    printf("%d\t%d\n", tid, d_v[tid]);
  }
}

int main( void ) {
  size_t vec_size = ((size_t)N*M) * sizeof(int);
  //std::cout << "N = " << N << std::endl;
  //std::cout << "sizeof(int) = " << sizeof(int) << std::endl;
  //std::cout << "size = " << vec_size << std::endl;

  int seed = (int)time(NULL);

  //std::experimental::reseed(3);
  std::experimental::reseed(seed);
  std::cout << "seed = " << seed << std::endl;


  int x;
  x = std::experimental::randint(0,MAX_VAL);
  //x = N-1; // x is in the last block


  int v[N][M];
  //int *v;
  //v = (int*)malloc(vec_size);

  init_vec(v);
  show_vec(v);

  int *dev_v;
  HANDLE_ERROR(cudaMalloc((void**)&dev_v, vec_size));
  HANDLE_ERROR(cudaMemcpy(dev_v, v, vec_size, cudaMemcpyHostToDevice));

  int g_count=0;
  int b = compute_num_blocks(N,M);
  std::cout <<
    "\nN = " << N <<
    "\nM = " << M <<
    "\nN*M = " << N*M <<
    "\nthreads = " << THREADS_PER_BLOCK <<
    "\nb = " << b <<
    "\nb*threads = " << b*THREADS_PER_BLOCK << "\n" <<
    std::endl;

  gpu_count<<<b, THREADS_PER_BLOCK>>>(dev_v, x);
  HANDLE_ERROR(cudaDeviceSynchronize());

  HANDLE_ERROR(cudaMemcpyFromSymbol(&g_count, d_g_count, sizeof(int))); // better than cudaMemcpy // Look at ref in README

  int c_count = cpu_count(v,x);

  std::cout << "\nx = " << x << "\tMAX_VAL = " << MAX_VAL << "\n" << std::endl;
  std::cout << "c_count = " << c_count << std::endl;
  std::cout << "g_count = " << g_count << std::endl;

  if (c_count == g_count)
    std::cout << "\nCorrect" << std::endl;
  else
    std::cout << "\nWRONG!" << std::endl;

  cudaFree(&d_g_count);
  cudaFree(dev_v);
  //free(v);

  return 0;
}
