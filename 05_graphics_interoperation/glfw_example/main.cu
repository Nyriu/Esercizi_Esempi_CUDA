#define GL_GLEXT_PROTOTYPES

#include <iostream>
#include <stdlib.h>

#include <GLFW/glfw3.h>
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


// Globals
#define DIM 512

GLuint bufferObj;
cudaGraphicsResource *resource;


__global__ void kernel(uchar4 *ptr) {
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // now calculate the value at that position
  float fx = x/(float)DIM - 0.5f;
  float fy = y/(float)DIM - 0.5f;
  unsigned char green =
    128 + 127 * sin(abs(fx*100) - abs(fy*100));

  // accessing uchar4 vs unsigned char*
  ptr[offset].x = 0;
  ptr[offset].y = green;
  ptr[offset].z = 0;
  ptr[offset].w = 255;
}


int main(int argc, char **argv) {
  cudaDeviceProp prop;
  int dev;

  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 0;
  HANDLE_ERROR(
      cudaChooseDevice(&dev,&prop)
      );
  //HANDLE_ERROR(
  //    cudaGLSetGLDevice(dev) // deprecated
  //    );


	if (!glfwInit()) exit(EXIT_FAILURE);
	if (atexit(glfwTerminate)) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	GLFWwindow* window;
	window = glfwCreateWindow(DIM, DIM, "Interop Example (GLFW)", NULL, NULL);
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
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM*DIM*4, NULL, GL_DYNAMIC_DRAW_ARB);

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

  dim3 grids(DIM/16, DIM/16);
  dim3 threads(16,16);
  kernel<<<grids,threads>>>(devPtr);

  HANDLE_ERROR(
      cudaGraphicsUnmapResources(1, &resource, NULL)
      );

	while (!glfwWindowShouldClose(window)) {
		//kernelUpdate(WIDTH, HEIGHT);
		glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
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

