#include <cmath>
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

#define IMG_WIDTH  2024
#define IMG_HEIGHT 2024

#define SPHERES 10

#define INF 2e10f
#define rnd(x) (x*rand() / (float)RAND_MAX)

class Sphere {
  public:
    float r,g,b;
    float radius;
    float x,y,z;
    __device__ float hit(float ox, float oy, float *n) {
      float dx = ox - x;
      float dy = oy - y;
      if (dx*dx + dy*dy < radius*radius) {
        float dz = sqrtf(radius*radius - dx*dx - dy*dy);
        *n = dz/sqrtf(radius*radius);
        return dz+z;
      }
      return -INF;
    }
};

__constant__ Sphere dev_s[SPHERES];

__global__ void kernel(int *ptr) { //, Sphere *dev_s) {
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;
  float ox = (x - (float)IMG_WIDTH/2);
  float oy = (y - (float)IMG_HEIGHT/2);
  float r=0, g=0, b=0;
  float maxz = -INF;
  for (int i=0; i < SPHERES; i++) {
    float n;
    float t = dev_s[i].hit(ox,oy,&n);
    if (t > maxz) {
      float fscale = n;
      r = dev_s[i].r * fscale;
      g = dev_s[i].g * fscale;
      b = dev_s[i].b * fscale;
    }
  }
  ptr[offset*3 + 0] = (int) 255 * r;
  ptr[offset*3 + 1] = (int) 255 * g;
  ptr[offset*3 + 2] = (int) 255 * b;
}




int main( void ) {
  // Init img on host
  int img_size = IMG_WIDTH*IMG_HEIGHT*3;
  size_t img_size_t = (size_t)IMG_WIDTH*IMG_HEIGHT*3*sizeof(float);
  int *img;
  img = (int*)malloc(img_size_t);
  for (int i=0; i<img_size; i+=3) { // init empty img
      img[i+0] = 0;
      img[i+1] = 0;
      img[i+2] = 0;
  }
  // Init spheres on host
  Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
  for (int i=0; i<SPHERES; i++) {
    temp_s[i].r =  (float) rnd(1.0f);
    temp_s[i].g =  (float) rnd(1.0f);
    temp_s[i].b =  (float) rnd(1.0f);
    temp_s[i].x = (float) rnd(1000.0f) - 500;
    temp_s[i].y = (float) rnd(1000.0f) - 500;
    temp_s[i].z = (float) rnd(1000.0f) - 500;
    temp_s[i].radius = (float) rnd(100.0f) + 20;
  }

  cudaEvent_t start, stop;
  HANDLE_ERROR( cudaEventCreate( &start ) );
  HANDLE_ERROR( cudaEventCreate( &stop ) );
  HANDLE_ERROR( cudaEventRecord( start,0 ) );

  int *dev_img;
  HANDLE_ERROR(cudaMalloc(&dev_img, img_size_t));
  HANDLE_ERROR(cudaMemcpyToSymbol(
        dev_s,
        temp_s,
        sizeof(Sphere) * (size_t)SPHERES));
  free(temp_s);


  dim3 grids(IMG_WIDTH/16,IMG_HEIGHT/16);
  dim3 threads(16,16);
  kernel<<<grids, threads>>>(dev_img);

  HANDLE_ERROR(cudaMemcpy(
        img,
        dev_img,
        img_size_t,
        cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(dev_img));
  HANDLE_ERROR( cudaEventRecord( stop,0 ) );
  HANDLE_ERROR( cudaEventSynchronize( stop ) );

  float elapsedTime;
  HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
        start, stop ));
  std::cout << "Time to generate: " << elapsedTime << "ms" << std::endl;




  // write img
  std::ofstream ofs;
  ofs.open("img.ppm");
  ofs << "P3\n" << IMG_WIDTH << " " << IMG_HEIGHT << "\n255\n";
  for (int i=0; i<img_size; i+=3) {
    ofs
      << img[i+0] << " "
      << img[i+1] << " "
      << img[i+2] << "\n";
  }
  ofs.close();


  return 0;
}
