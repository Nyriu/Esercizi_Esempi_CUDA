// Esercizio
// Dato array di N interi vec[] e un valore intero x conta quanti sono gli elementi di vec[] uguali a x
//
// x e N forniti da linea di comando // TODO
// host alloca vec[] e lo inizializza
// allocazione su device a nostra discrezione (trasferimento esplicito o mem mapped)
// effettua il calcolo con kernel da B blocks e 256 thread per block (calcolare B in realzione ad N) // TODO calc B
// ogni thread accede ad un elem e verifica se uguale x, se lo e' aggiorna count con op atomica
// infine host recupera risultato e stampa (host verifica correttezza)

#include <iostream>
#include <experimental/random>

#define N 256*2
#define MAX_VAL int(N/5)
#define B 2

static void HandleError( cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString( err ) << " in " << file << " line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))

void init_vec(int *v) {
  for (int i=1; i<N; i++) {
    v[i] = std::experimental::randint(0,MAX_VAL);
  }
}

void show_vec(int *v) {
  std::cout << "\n" << v[0];
  for (int i=1; i<N; i++) {
    std::cout << ", " << v[i];
  }
  std::cout << "\n" << std::endl;
}

int cpu_count(int *v, const int x) {
  int c=0;
  for (int i=1; i<N; i++) {
    if (v[i] == x)
      c++;
  }
  return c;
}


__device__ int d_g_count = 0;
__global__ void gpu_count(int *d_v, const int x) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;

  if (d_v[tid] == x)
    atomicAdd(&d_g_count, 1);
}

int main( void ) {
  int x;
  x = std::experimental::randint(0,MAX_VAL);

  int v[N];
  init_vec(v);
  //show_vec(v);

  int *dev_v;
  HANDLE_ERROR(cudaMalloc((void**)&dev_v, N*sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(dev_v, v, N*sizeof(int), cudaMemcpyHostToDevice));

  int g_count=0;
  gpu_count<<<B, 256>>>(dev_v, x);
  //HANDLE_ERROR(cudaDeviceSynchronize());
  cudaMemcpyFromSymbol(&g_count, d_g_count, sizeof(int)); // better than cudaMemcpy // Look at ref in README

  int c_count = cpu_count(v,x);

  std::cout << "x = " << x << "\n" << std::endl;
  std::cout << "c_count = " << c_count << std::endl;
  std::cout << "g_count = " << g_count << std::endl;

  if (c_count == g_count)
    std::cout << "\nCorrect" << std::endl;
  else
    std::cout << "\nWRONG!" << std::endl;

  cudaFree(&d_g_count);
  cudaFree(dev_v);

  return 0;
}
