#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

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

__global__ void kernel(int *ptr) {
  // map from threadIdx/BlockIdx to pixel position
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x+y*gridDim.x;

  // now calculate the value at that position
  int juliaValue = julia(x,y);
  ptr[offset*3 + 0] = 255 * juliaValue;
  ptr[offset*3 + 1] = 255 * juliaValue;
  ptr[offset*3 + 2] = 255 * juliaValue;
}


int main( void ) {
  int width, height;
  width = height = DIM;

  int img_size = width*height*3;
  size_t img_size_t = (size_t)width*height*3*sizeof(float);
  int *img;
  img = (int*)malloc(img_size_t);
  for (int i=0; i<img_size; i+=3) { // init empty img
      img[i+0] = 125;
      img[i+1] = 125;
      img[i+2] = 125;
  }

  //std::cout << "sizeof(float)                  = " << sizeof(float) << std::endl;
  //std::cout << "img_size                       = " << img_size<< std::endl;
  //std::cout << "img_size*sizeof(float)         = " << img_size*sizeof(float) << std::endl;
  //std::cout << "(size_t)img_size*sizeof(float) = " << (size_t)img_size*sizeof(float) << std::endl;
  //std::cout << "img_size_t                     = " << img_size_t << std::endl;

  int *dev_img;
  HANDLE_ERROR(cudaMalloc(&dev_img, img_size_t));

  dim3 grid(DIM,DIM);
  kernel<<<grid, 1>>>(dev_img);

  HANDLE_ERROR(cudaMemcpy(
        img,
        dev_img,
        img_size_t,
        cudaMemcpyDeviceToHost));

  // write img
  std::ofstream ofs;
  ofs.open("img.ppm");
  ofs << "P3\n" << width << " " << height << "\n255\n";
  for (int i=0; i<img_size; i+=3) {
    ofs
      << img[i+0] << " "
      << img[i+1] << " "
      << img[i+2] << "\n";
  }
  ofs.close();


  HANDLE_ERROR(cudaFree(dev_img));

  return 0;
}
