
#include <stdlib.h>
#include <iostream>

#include <string>
#include <fstream>

//#include <experimental/random>
//#include <time.h>

#include "Image.h"

static void HandleError( cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString( err ) << " in " << file << " line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))



#define DIM 1000


class cuComplex {
  private:
    float r;
    float i;

  public:
    __device__ __host__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2(void) { return r*r + i*i; }
    __device__ cuComplex operator* (const cuComplex& a) {
      return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+ (const cuComplex& a) {
      return cuComplex(r+a.r, i+a.i);
    }
};


__device__ int julia(int x, int y) {
  const float scale = 1.5;
  float jx = scale * (float)(DIM/2.f - x)/(DIM/2.f);
  float jy = scale * (float)(DIM/2.f - y)/(DIM/2.f);

  cuComplex c(-.8f, .156f);
  cuComplex a(jx,jy);

  int i=0;
  for (i=0; i<200; i++) {
    a = a*a + c;
    if (a.magnitude2() > 1000)
      return 0;
  }
  return 1;
}

__global__ void kernel(float *ptr) {
  // map from threadIdx/BlockIdx to pixel position
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x+y*gridDim.x;

  // now calculate the value at that position
  int juliaValue = julia(x,y);
  ptr[offset*3 + 0] = juliaValue;
  ptr[offset*3 + 1] = juliaValue;
  ptr[offset*3 + 2] = juliaValue;
}



int main( void ) {
  Image img(DIM, DIM);
  float *dev_img;
  
  //std::cout << "img.size() = " << img.size() << std::endl;
  HANDLE_ERROR(cudaMalloc((void**)&dev_img, img.size()));
  
  dim3 grid(DIM,DIM);
  kernel<<<grid, 1>>>(dev_img);
  
  HANDLE_ERROR(cudaMemcpy(
        img.get_ptr(),
        dev_img,
        img.size(),
        cudaMemcpyDeviceToHost));
  
  img.writePPM("img.ppm");

  return 0;
}
