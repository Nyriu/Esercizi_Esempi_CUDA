// file esempio querydevice
#include "stdio.h"

static void HandleError( cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))



int main( void ) {
  cudaDeviceProp prop;

  int count;
  HANDLE_ERROR( cudaGetDeviceCount( &count) );
  for (int i=0; i<count; i++) {
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ));
    printf(" -- General Information for device %d --\n", i);
    printf("Name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Device copy overlap: ");
    if(prop.deviceOverlap)
      printf("Enabled\n");
    else
      printf("Disabled\n");
    printf("Kernel executionn timeout: ");
    if (prop.kernelExecTimeoutEnabled)
      printf("Enabled\n");
    else
      printf("Disabled\n");

    printf(" -- Memory Information for device %d --\n", i);
    printf("Total global mem: %ld\n", prop.totalGlobalMem);
    printf("Total constant mem: %ld\n", prop.totalConstMem);
    printf("Max mem pitch: %ld\n", prop.textureAlignment);

    printf(" -- MP Information for device %d --\n", i);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Shared mem per mp: %ld\n", prop.sharedMemPerMultiprocessor);
    printf("Register per mp: %d\n", prop.regsPerMultiprocessor);
    printf("Threads in warp: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max thread dimensions: (%d, %d, %d)\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid dimensions: (%d, %d, %d)\n",
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("\n");
  }

  return 0;
}
