#include <iostream>
#include <cuda.h>

static void HandleError( cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cout << "Error Name: " << cudaGetErrorName( err ) << std::endl;
    std::cout << cudaGetErrorString( err ) << " in " << file << " line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))


class Base {
  public:
    __device__ virtual void fun1() const  {
    //__device__ void fun1() const  {
      printf("Base fun1\n");
    }

    __device__ void fun2() const {
      printf("Base fun2\n");
      fun1();
    }
};

class Derived : public Base {
  public:
    __device__ void fun1() const override {
    //__device__ void fun1() const {
      printf("Derived fun2\n");
    }
};



static __global__ void kernel(
    const Base *obj
    ) {
  printf("inside kernel\n");
  obj->fun1();
  obj->fun2();
}


int main() {
  cudaDeviceProp prop;
  int dev;
  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 0;
  HANDLE_ERROR(
      cudaChooseDevice(&dev,&prop)
      );

  Base obj;
  Base *dev_obj = nullptr;
  
  //Derived obj;
  //obj = Derived();
  //Derived *dev_obj = nullptr;

  // Static allocation on device memory
  HANDLE_ERROR(
      cudaMalloc((void**)&dev_obj, sizeof(*dev_obj))
      );
  // Copy from host to device
  HANDLE_ERROR(
      cudaMemcpy((void*)dev_obj, (void*)&obj, sizeof(*dev_obj), cudaMemcpyHostToDevice)
      );

  float grids = 1;
  float threads = 1;
  kernel<<<grids,threads>>>(dev_obj);

  HANDLE_ERROR(cudaDeviceSynchronize());

  return 0;
}

