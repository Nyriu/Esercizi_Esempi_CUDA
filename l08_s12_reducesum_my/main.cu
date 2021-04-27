// Esempio collective  operations: reduce_sum

#include <cooperative_groups.h>
//using namespace cooperative_groups;
namespace cg = cooperative_groups;

#include <locale>
#include <stdlib.h>
#include <iostream>
#include <experimental/random>
#include <time.h>


#define RNG_MAX_VAL 3 // 5 // 50 // max rng val for array elems

static void HandleError( cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString( err ) << " in " << file << " line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))

void init_vec(int *v, int n) {
  for (int i=0; i<n; i++) {
    v[i] = std::experimental::randint(0,RNG_MAX_VAL);
    //v[i] = i;
  }
}

void show_vec(int *v, int n) {
  std::cout << "\n" << v[0];
  for (int i=1; i<n; i++) {
    std::cout << ", " << v[i];
  }
  std::cout << "\n" << std::endl;
}

int cpu_sum(int *v, int n) {
  int s=0;
  for (int i=0; i<n; i++) {
    s += v[i];
  }
  return s;
}

// Codice Prof
__device__ int reduce_sum(cg::thread_group g, int *temp, int val) {
  int lane = g.thread_rank();
  // ad ogni iterazione si dimezza il numero di thread attivi
  // ogni thread somma parziale temp[i] a temp[lane+i]
  for (int i=g.size()/2; i>0; i/=2) {
    temp[lane] = val;
    g.sync(); // attendo tutti thread del gruppo
    if (lane < i)  val += temp[lane+i];
    g.sync();
  }
  return val; // solo thread 0 restituisce la somma completa
}

__device__ int thread_sum(int *input, int n) {
  int sum=0;
  for (int i=blockIdx.x * blockDim.x + threadIdx.x;
      i<n/4;
      i+=blockDim.x * gridDim.x) { // accesso strided
    int4  in = ((int4*)input)[i]; // vector load e' piu' effciente
    sum += in.x + in.y + in.z + in.w;
  }
  return sum;
}

__global__ void sum_kernel_block(int *sum, int *input, int n) {
  int my_sum = thread_sum(input, n);
  extern __shared__ int temp[]; // extern perche' allocazione dinamica con
                                // terzo argomento della kernel call <<< ... >>>
  //auto g = cg::this_thread_block();
  cg::thread_block g = cg::this_thread_block();
  int block_sum = reduce_sum(g, temp, my_sum);
  if(g.thread_rank() == 0)
    atomicAdd(sum, block_sum);
}
// END // Codice Prof


// ATTENZIONE!! Funziona solo con n=2^k con k>1

//int n = 1<<24; // array len = 16M // n=2^24 // bit shift operation
//int blockSize = 256;
////int nBlocks = (n+blockSize-1) / blockSize; // work as ceiling
//int nBlocks = (n+(blockSize*4)-1) / (blockSize*4); // il numero di blocchi si puo' ridurre perche' sopra non si tiene conto degli int4
//int sharedBytes = blockSize * sizeof(int);

// toy example
int n = 16;
int blockSize = 2;
int nBlocks = (n+(blockSize*4)-1) / (blockSize*4); // il numero di blocchi si puo' ridurre perche' sopra non si tiene conto degli int4
int sharedBytes = blockSize * sizeof(int);


int main( void ) {
  //int seed = (int)time(NULL);
  int seed = 1619508961;
  std::experimental::reseed(seed);
  std::cout << "seed = " << seed << std::endl;

  std::cout << "\nn           = " << n << std::endl;
  std::cout << "blockSize   = " << blockSize << std::endl;
  std::cout << "nBlocks     = " << nBlocks << std::endl;
  std::cout << "sharedBytes = " << sharedBytes << "\n" << std::endl;

  size_t data_size = (size_t)n*sizeof(int);

  int *sum, *data;
  sum  = (int*)malloc(sizeof(int));
  data = (int*)malloc(data_size);

  int *d_sum, *d_data;
  HANDLE_ERROR(cudaMalloc((void**)&d_sum, sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&d_data, data_size));

  init_vec(data,n);
  if (n < 32) // mostra il vettore solo se e' piccolo
    show_vec(data,n);

  HANDLE_ERROR(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));

  sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(d_sum, d_data, n);

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaMemcpy(sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost));


  int c_sum = cpu_sum(data,n);

  std::cout << "c_sum = " << c_sum << std::endl;
  std::cout << "g_sum = " << *sum << std::endl;

  if (c_sum == *sum)
    std::cout << "\nCorrect" << std::endl;
  else
    std::cout << "\nWRONG!" << std::endl;

  cudaFree(d_data);
  cudaFree(d_sum);

  return 0;
}
