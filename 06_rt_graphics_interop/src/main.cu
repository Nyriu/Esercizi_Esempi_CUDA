#define GL_GLEXT_PROTOTYPES

#include <iostream>
//#include <stdlib.h>
//
//#ifdef __clang__
//#include <__clang_cuda_math.h>
//#include <__clang_cuda_builtin_vars.h>
//#endif

#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#define rnd(x) (x*rand() / (float)RAND_MAX)

#include "common.h"
#include "Sphere.h"
#include "Light.h"
#include "Scene.h"
#include "Ray.h"
#include "Camera.h"
#include "Tracer.h"


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

// Globals
#define IMG_H 512
#define IMG_W 512

// TODO
//#define IMG_H 1080
//#define IMG_W IMG_H*16/9;

static __global__ void kernel(uchar4 *ptr,
    const Camera *cam,
    const Scene *sce,
    const Tracer *trc
    ) {

  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // in img coord (0,0) is bottom-left
  // Put coords in [0,1]
  float u = (x + .5) / ((float) IMG_W -1); // NDC Coord
  float v = (y + .5) / ((float) IMG_H -1); // NDC Coord
  Ray r = cam->generate_ray(u,v);

  color c = trc->trace(&r, sce);
  //color c(0.2);


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
    __host__ void render(
        Camera *cam,
        Scene *sce,
        uchar4 *devPtr) {
      // --- Generate One Frame ---
      // TODO dims
      //dim3 grids(IMG_W/16, IMG_H/16);
      //dim3 threads(16,16);
      dim3 grids(IMG_W, IMG_H);
      dim3 threads(1);

      Camera *devCamPtr = nullptr;
      Tracer *devTrcPtr = nullptr; // TODO

      // Static allocation on device memory
      HANDLE_ERROR(
          cudaMalloc((void**)&devCamPtr, sizeof(Camera))
          );
      // Copy from host to device
      HANDLE_ERROR(
          cudaMemcpy((void*)devCamPtr, (void*)cam, sizeof(Camera), cudaMemcpyHostToDevice)
          );
      Scene *devScePtr = sce->to_device();

      kernel<<<grids,threads>>>(devPtr, devCamPtr, devScePtr, devTrcPtr);

      HANDLE_ERROR(cudaFree((void*)devCamPtr));
      HANDLE_ERROR(cudaFree((void*)devScePtr));
      HANDLE_ERROR(cudaFree((void*)devTrcPtr));
    }
  private:
    // TODO far funzionare qua
    //static __global__ void kernel(uchar4 *ptr) {
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
  Scene sce;

  // Init Random scene
  int spheres_num = 100;
  srand( (unsigned)time(NULL) );
  for (int i=0; i<spheres_num; i++) {
    point3 pos(
        (float) rnd(4.0f) - 2,
        (float) rnd(4.0f) - 2,
        (float) rnd(4.0f) - 2
        );
    float radius = (float) rnd(0.3f) + 0.1;
    color c(
        (float) rnd(1.0f),
        (float) rnd(1.0f),
        (float) rnd(1.0f)
        );
    sce.addShape(new Sphere(
          pos,
          radius,
          c
          )
        );
  }

  //sce.addShape(new Sphere(1, color(0.5, 0.8, 0.7)));
  //sce.addShape(new Sphere(point3(1.5,0,0), .5));
  //sce.addLight(new PointLight(point3(5,4,3), color(1), 80));

  Renderer renderer;
  renderer.render(&cam, &sce, devPtr);

  HANDLE_ERROR(cudaDeviceSynchronize()); // helps with debugging!!
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

