#include <iostream>
#include <vector>
#include <cuda.h>
//#include <cooperative_groups.h>
//using namespace cooperative_groups;
//// Alternatively use an alias to avoid polluting the namespace with collective algorithms
////namespace cg = cooperative_groups;

////for __syncthreads()
//#ifndef __CUDACC__ 
//#define __CUDACC__
//#endif
//#include <device_functions.h>
#include <cuda_runtime_api.h> 
#include <cooperative_groups.h>



static void HandleError( cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cout << "Error Name: " << cudaGetErrorName( err ) << std::endl;
    std::cout << cudaGetErrorString( err ) << " in " << file << " line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))


enum PolygonType { none, rect, triang };
struct PolygonInfo {
  public:
    int width, height;
    PolygonType ptype;
    __host__ __device__ PolygonInfo(int w, int h, PolygonType t) :
      width(w), height(h), ptype(t) {}
};
class Polygon {
  protected:
    int width, height;
  public:
    __host__ __device__ Polygon(int w, int h) : width(w), height(h) {}
    __host__ __device__ Polygon(const PolygonInfo& pi) : width(pi.width), height(pi.height) {}
    __host__ __device__  void set_values(int a, int b) {
      width=a;
      height=b;
    }
    __host__ __device__  virtual int area() {
      printf("\nLOL here!\n");
      printf(" width = %d\n height = %d\n", width, height);
      return 0;
    }
    __host__ __device__  virtual PolygonInfo get_info() {
      return PolygonInfo(width,height,PolygonType::none);
    }
};

class Rectangle: public Polygon {
  public:
    __host__ __device__ Rectangle(int w, int h) : Polygon(w,h) {}
    __host__ __device__ Rectangle(const PolygonInfo& pi) : Polygon(pi) {}
    __host__ __device__  int area() override {
      return width * height;
    }
    __host__ __device__  PolygonInfo get_info() override {
      printf("rect get_info\n");
      return PolygonInfo(width,height,PolygonType::rect);
    }
};

class Triangle: public Polygon {
  public:
    __host__ __device__ Triangle(int w, int h) : Polygon(w,h) {}
    __host__ __device__ Triangle(const PolygonInfo& pi) : Polygon(pi) {}
    __host__ __device__   int area() override {
      return (width * height / 2);
    }
    __host__ __device__  PolygonInfo get_info() override {
      printf("triang get_info\n");
      return PolygonInfo(width,height,PolygonType::triang);
    }
};



static __global__ void wrong_example_kernel(Polygon *pols, int n_pols) {
  Polygon *p = pols;
  for (int i=0; i<n_pols; i++) {
    printf("(device) p->area() = %d", p->area());
    p++;
  }
}


static __global__ void inst_obj_dev_kernel(PolygonInfo *pols_infos, int n_pols) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  //printf("%d %d\n", x,y);

  size_t pols_size = sizeof(Polygon)*n_pols;
  Polygon *pols = (Polygon*) malloc(pols_size);

  for (int i=0; i<n_pols; i++) {
    Polygon *tmp_p = nullptr;
    //PolygonInfo tmp_pi = *(pols_infos+i);
    PolygonInfo *pi_p = pols_infos+i;

    //printf("w=%d, h=%d, ptype=%d\n", pi_p->width, pi_p->height, pi_p->ptype);

    if (pi_p->ptype == PolygonType::rect) {
      tmp_p = new Rectangle(*(pols_infos+i));
    } else if (pi_p->ptype == PolygonType::triang) {
      tmp_p = new Triangle(*(pols_infos+i));
    } else if (pi_p->ptype == PolygonType::none) {
      tmp_p = new Polygon(*(pols_infos+i));
    } else {
      printf("we have a problem...\n");
    }

    memcpy(pols+i, tmp_p, sizeof(*tmp_p));
  }

  printf("%d %d\n", x,y);
  for (int i=0; i<n_pols; i++) {
    printf("\t area = %d\n", pols[i].area());
  }

}


__device__ Polygon *d_pols = nullptr;
__device__ int d_n_pols = 0;
static __global__ void copy_kernel(PolygonInfo *pols_infos, int n_pols) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  //printf("%d %d\n", x,y);

  if (x == 0 && y == 0) {
    printf("%d %d\n I'm instancing...\n", x,y);
    size_t pols_size = sizeof(Polygon)*n_pols;
    d_pols = (Polygon*) malloc(pols_size);
    d_n_pols = n_pols;

    for (int i=0; i<n_pols; i++) {
      Polygon *tmp_p = nullptr;
      //PolygonInfo tmp_pi = *(pols_infos+i);
      PolygonInfo *pi_p = pols_infos+i;

      //printf("w=%d, h=%d, ptype=%d\n", pi_p->width, pi_p->height, pi_p->ptype);

      if (pi_p->ptype == PolygonType::rect) {
        tmp_p = new Rectangle(*(pols_infos+i));
      } else if (pi_p->ptype == PolygonType::triang) {
        tmp_p = new Triangle(*(pols_infos+i));
      } else if (pi_p->ptype == PolygonType::none) {
        tmp_p = new Polygon(*(pols_infos+i));
      } else {
        printf("we have a problem...\n");
      }
      memcpy(d_pols+i, tmp_p, sizeof(*tmp_p));
    }
  }
}

static __global__ void kernel() {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (d_pols == nullptr) {
    printf("%d %d\n pols=%p, d_n_pols=%d\n", x, y, d_pols, d_n_pols);
    return;
  }
  for (int i=0; i<d_n_pols; i++) {
    int area = d_pols[i].area();
    if (area != 6 && area != 35)
      printf("%d %d\n pols=%p, area = %d\n", x, y, d_pols, d_pols[i].area());
  }
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

  // Host init
  //Polygon tri = Triangle(3,4);
  Triangle tri(3,4);
  std::cout << "tri.area() = " << tri.area() << std::endl;
  Rectangle rec(5,7);
  std::cout << "rec.area() = " << rec.area() << std::endl;

  std::vector<Polygon*> pols;
  pols.push_back(&tri);
  pols.push_back(&rec);

  for (Polygon *p : pols) {
    std::cout << "p->area() = " << p->area() << std::endl;
  }

  // Now I want to move the vector to GPU and
  // for each elem call area() from device


  // { /// WRONG WAY
  //   size_t total_size = 0;
  //   for (Polygon *p : pols) {
  //     total_size += sizeof(*p);
  //   }

  //   Polygon *dev_pols = nullptr;
  //   HANDLE_ERROR(
  //       cudaMalloc((void**)&dev_pols, total_size)
  //       );

  //   int offset = 0;
  //   for (Polygon *p : pols) {
  //     HANDLE_ERROR(
  //         cudaMemcpy((void*)(dev_pols+offset), (void*)p, sizeof(*p), cudaMemcpyHostToDevice)
  //         );
  //     offset++;
  //   }
  //   float grids = 1;
  //   float threads = 1;
  //   wrong_example_kernel<<<grids,threads>>>(dev_pols, pols.size());
  //   // this generate a wrong mem access because vtable on host
  // } /// END // WRONG WAY

  //{ /// INSTACING OBJS ON DEVICE
  //  size_t pols_infos_size = sizeof(PolygonInfo)*pols.size();
  //  PolygonInfo *pols_infos = (PolygonInfo *) malloc(pols_infos_size);
  //  int i = 0;
  //  for (Polygon *p : pols) {
  //    PolygonInfo pi = p->get_info();
  //    memcpy(&pols_infos[i], &pi, sizeof(pi));
  //    i++;
  //  }

  //  PolygonInfo *dev_pols_infos = nullptr;
  //  HANDLE_ERROR(
  //      cudaMalloc((void**)&dev_pols_infos, pols_infos_size)
  //      );
  //  HANDLE_ERROR(
  //      cudaMemcpy((void*)dev_pols_infos, (void*)pols_infos, pols_infos_size, cudaMemcpyHostToDevice)
  //      );

  //  free(pols_infos);

  //  dim3 grids(1);
  //  dim3 threads(1);
  //  inst_obj_dev_kernel<<<grids,threads>>>(dev_pols_infos, pols.size());
  //} /// END // INSTACING OBJS ON DEVICE


  { /// AS BEFORE BUT ONE KERNEL INST AND THE OTHER USE
    size_t pols_infos_size = sizeof(PolygonInfo)*pols.size();
    PolygonInfo *pols_infos = (PolygonInfo *) malloc(pols_infos_size);
    int i = 0;
    for (Polygon *p : pols) {
      PolygonInfo pi = p->get_info();
      memcpy(&pols_infos[i], &pi, sizeof(pi));
      i++;
    }

    PolygonInfo *dev_pols_infos = nullptr;
    HANDLE_ERROR(
        cudaMalloc((void**)&dev_pols_infos, pols_infos_size)
        );
    HANDLE_ERROR(
        cudaMemcpy((void*)dev_pols_infos, (void*)pols_infos, pols_infos_size, cudaMemcpyHostToDevice)
        );

    free(pols_infos);

    copy_kernel<<<1,1>>>(dev_pols_infos, pols.size());
    HANDLE_ERROR(cudaDeviceSynchronize());
    dim3 grids(3,3);
    dim3 threads(10,10);
    kernel<<<grids,threads>>>();
  } /// END

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaDeviceReset());

  return 0;
}

