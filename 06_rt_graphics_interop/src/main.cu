#include <glm/common.hpp>
#define GL_GLEXT_PROTOTYPES

#include <iostream>
//#include <stdlib.h>
//
//#ifdef __clang__
//#include <__clang_cuda_math.h>
//#include <__clang_cuda_builtin_vars.h>
//#endif

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
//#include <glm/vec3.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>

#include <cuda.h>
#include <cuda_gl_interop.h>

static void HandleError( cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cout << "Error Name: " << cudaGetErrorName( err ) << std::endl;
    std::cout << cudaGetErrorString( err ) << " in " << file << " line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}


// Aliases
using vec3   = glm::vec3;
using point3 = glm::vec3;
using color  = glm::vec3;

// Globals
#define IMG_H 512/4
#define IMG_W 512/4

//#define IMG_H 1080
//#define IMG_W IMG_H*16/9;


class Sphere {
  private:
    float radius_;
    point3 center_;

  public:
    Sphere(const float& radius) : radius_(radius) {
      center_ = point3(0);
    }

    float getDist(const point3& point) const {
      return glm::length(point) - radius_;
    }
};

class Ray {
  public:
    const point3 orig_;
    const vec3 dir_;

    __device__ Ray() : orig_(point3(0)), dir_(glm::normalize(vec3(0,0,-1))) { }
    __device__ Ray(const point3& origin, const vec3& direction) :
      orig_(origin), dir_(glm::normalize(direction)) {}

    const point3 at(float t) const {
      return orig_ + t*dir_;
    }
};

class Camera {
  private:
    point3 center_;
    vec3 dir_;

    float aspect_;
    float fov_;

  public:
    Camera() {
      center_ = point3(0,0,5);
      dir_ = glm::normalize(vec3(0,0,-1));
      aspect_ = 1;
      fov_ = 45;
    }

    __device__ Ray generate_ray(float u, float v) const { // input NDC Coords
      // Put coords in [-1,1]
      float su = 2 * u - 1; // Screen Coord
      float sv = 1 - 2 * v; // Screen Coord (flip y axis)

      // Aspect Ratio
      su *= aspect_; // x in [-asp ratio, asp ratio]
      // y in [-1,1] (as before)

      // Field Of View
      su *= std::tan(fov_/2);
      sv *= std::tan(fov_/2);

      //float scale = 1;
      //// Scale
      //su *= scale;
      //sv *= scale;

      // From ScreenCoords to WorldCoords
      point3 p    = point3(su,sv,center_.z - 1) ;
      point3 orig = center_;
      return Ray(orig, glm::normalize((orig - p)));
    }
};


class Tracer {
  public:
    __device__ color trace(const Ray *r, const Sphere *sph) const {
      //TODO sphere tracing
      return color((float).25);
    };
  private:
    // TODO
    //  Color sphereTrace(const Ray& r); // better with pointer?
    //  Color shade(const Point3& p, const Vec3& viewDir, const ImplicitShape *shape);
    //  bool sphereTraceShadow(const Ray& r, const ImplicitShape *shapeToShadow);
};

static __global__ void kernel(uchar4 *ptr,
    const Camera *cam,
    const Sphere *sph,
    const Tracer *trc
    ) {

  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  //// now calculate the value at that position
  //float fx = x/(float)IMG_W - 0.5f;
  //float fy = y/(float)IMG_H - 0.5f;
  //unsigned char green =
  //  128 + 127 * sin(abs(fx*100) - abs(fy*100));

  // in img coord (0,0) is top-left
  // Put coords in [0,1]
  float u = (x + .5) / ((float) IMG_W -1); // NDC Coord
  float v = (y + .5) / ((float) IMG_H -1); // NDC Coord
  Ray r = cam->generate_ray(u,v);

  color c = trc->trace(&r, sph);

  //img[j*IMG_W+i + 0] = (int) 255 * c.r;
  //img[j*IMG_W+i + 1] = (int) 255 * c.g;
  //img[j*IMG_W+i + 2] = (int) 255 * c.b;
  //img[j*IMG_W+i + 3] = (int) 255 * 1;
  //printf("(x,y) = (%d, %d)\n", x,y);
  //printf("(u,v) = (%f, %f)\n", u,v);
  //printf("[%f, %f, %f]\n", c.r, c.g, c.b);
  //printf("[%d, %d, %d]\n", (int)(255*c.r), (int)(255*c.g), (int)(255*c.b));

  // accessing uchar4 vs unsigned char*
  ptr[offset].x = (int) (255 * c.r); // (int) (u * 255); //0;
  ptr[offset].y = (int) (255 * c.g); // (int) (v * 255); //(int)255/2;
  ptr[offset].z = (int) (255 * c.b); // 0;
  ptr[offset].w = 255;
}
class Renderer {
  private:
    Tracer *tracer_;
  public:
    void render(
        const Camera *cam,
        const Sphere *sph,
              uchar4 *devPtr) {
      // --- Generate One Frame ---

      // TODO qui va il kernel al posto dei for
      dim3 grids(IMG_W/16, IMG_H/16);
      dim3 threads(16,16);
      kernel<<<grids,threads>>>(devPtr, cam, sph, tracer_);

      //// in img coord (0,0) is top-left
      //for (int j=0; j<IMG_H; ++j) {
      //  for (int i=0; i<IMG_W; ++i) {
      //    // Put coords in [0,1]
      //    float u = double(i + .5) / (IMG_W -1); // NDC Coord
      //    float v = double(j + .5) / (IMG_H -1); // NDC Coord
      //    Ray r = cam->generate_ray(u,v);

      //    color c = tracer_->trace(r);

      //    img[j*IMG_W+i + 0] = (int) 255 * c.r;
      //    img[j*IMG_W+i + 1] = (int) 255 * c.g;
      //    img[j*IMG_W+i + 2] = (int) 255 * c.b;
      //    img[j*IMG_W+i + 3] = (int) 255 * 1;
      //  }
      //}
    }
  private:
    // TODO far funzionare qua
    //static __global__ void kernel(uchar4 *ptr) {
    //  // map from threadIdx/BlockIdx to pixel position
    //  int x = threadIdx.x + blockIdx.x * blockDim.x;
    //  int y = threadIdx.y + blockIdx.y * blockDim.y;
    //  int offset = x + y * blockDim.x * gridDim.x;

    //  // now calculate the value at that position
    //  float fx = x/(float)IMG_W - 0.5f;
    //  float fy = y/(float)IMG_H - 0.5f;
    //  unsigned char green =
    //    128 + 127 * sin(abs(fx*100) - abs(fy*100));

    //  // accessing uchar4 vs unsigned char*
    //  ptr[offset].x = 0;
    //  ptr[offset].y = green;
    //  ptr[offset].z = 0;
    //  ptr[offset].w = 255;
    //}
};



GLuint bufferObj;
cudaGraphicsResource *resource;



int main() {
  cudaDeviceProp prop;
  int dev;

  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 0;
  HANDLE_ERROR(
      cudaChooseDevice(&dev,&prop)
      );

	if (!glfwInit()) exit(EXIT_FAILURE);
	if (atexit(glfwTerminate)) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	GLFWwindow* window;
	window = glfwCreateWindow(IMG_W, IMG_H, "GLFW Window", NULL, NULL);
	if (!window) exit(EXIT_FAILURE);

  glfwSetKeyCallback(window, key_callback);

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);


  std::cout << glGetString(GL_VENDOR) << std::endl;
  std::cout << glGetString(GL_RENDERER) << std::endl;

  // TODO ARB or not ARB
	//glGenBuffers(1, &pbo);
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	//glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLubyte)*WIDTH*HEIGHT, NULL, GL_DYNAMIC_DRAW);
  glGenBuffers(1, &bufferObj);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, IMG_H*IMG_W*4, NULL, GL_DYNAMIC_DRAW_ARB);

  HANDLE_ERROR(
      cudaGraphicsGLRegisterBuffer(
        &resource,
        bufferObj,
        cudaGraphicsMapFlagsNone
        )
      );

  uchar4* devPtr;
  size_t size;
  HANDLE_ERROR(
      cudaGraphicsMapResources(1, &resource, NULL)
      );
  HANDLE_ERROR(
      cudaGraphicsResourceGetMappedPointer(
        (void**)&devPtr,
        &size,
        resource
        )
      );

  Camera cam;
  Sphere sph(1);
  Renderer renderer;
  renderer.render(&cam, &sph, devPtr);

  

  HANDLE_ERROR(
      cudaGraphicsUnmapResources(1, &resource, NULL)
      );


	while (!glfwWindowShouldClose(window)) {
		//kernelUpdate(WIDTH, HEIGHT);
		glDrawPixels(IMG_W, IMG_H, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glfwSwapBuffers(window);

    // Poll for and process events
    glfwPollEvents();
	}



  HANDLE_ERROR(
      cudaGraphicsUnregisterResource(resource)
      );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  glDeleteBuffers(1, &bufferObj);

  glfwTerminate();

  return 0;
}

