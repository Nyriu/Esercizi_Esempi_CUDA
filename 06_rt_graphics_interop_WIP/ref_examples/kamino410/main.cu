#include <iostream>
#include <stdlib.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define WIDTH 512
#define HEIGHT 512

extern "C" void kernelBindPbo(GLuint pixelBufferObj);
extern "C" void kernelUpdate(int width, int height);
extern "C" void kernelExit(GLuint pixelBufferObj);

GLuint pbo;

int main() {
	if (!glfwInit()) exit(EXIT_FAILURE);
	if (atexit(glfwTerminate)) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	GLFWwindow* window;
	window = glfwCreateWindow(WIDTH, HEIGHT, "gl-cuda-test", NULL, NULL);
	if (!window) exit(EXIT_FAILURE);

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	if (glewInit() != GLEW_OK) exit(EXIT_FAILURE);

  std::cout << glGetString(GL_VENDOR) << std::endl;
  std::cout << glGetString(GL_RENDERER) << std::endl;

	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLubyte)*WIDTH*HEIGHT, NULL, GL_DYNAMIC_DRAW);

	kernelBindPbo(pbo);

	while (!glfwWindowShouldClose(window)) {
		kernelUpdate(WIDTH, HEIGHT);
		glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glfwSwapBuffers(window);
	}

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	kernelExit(pbo);
	glDeleteBuffers(1, &pbo);

	//getchar();

	return 0;
}

#include <iostream>

#include <GL/glew.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

cudaGraphicsResource *cudapbo;

extern "C" void kernelBindPbo(GLuint pixelBufferObj) {
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&cudapbo, pixelBufferObj, cudaGraphicsRegisterFlagsWriteDiscard));
}

extern "C" void kernelExit(GLuint pixelBufferObj) {
	gpuErrchk(cudaGLUnregisterBufferObject(pixelBufferObj));
	gpuErrchk(cudaGraphicsUnregisterResource(cudapbo));
}

__global__ void kernel(uchar4 *map, unsigned char frame) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id = x + y * blockDim.x * gridDim.x;

	map[id].x = x / 2;
	map[id].y = y / 2;
	map[id].z = frame;
	map[id].w = 255;
}

extern "C" void kernelUpdate(int width, int height) {
	static unsigned char frame = 0;
	frame++;
	uchar4 *dev_map;

	gpuErrchk(cudaGraphicsMapResources(1, &cudapbo, NULL));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dev_map, NULL, cudapbo));

	dim3 threads(8, 8);
	dim3 grids(width / 8, height / 8);
	kernel << <grids, threads >> > (dev_map, frame);

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaGraphicsUnmapResources(1, &cudapbo, NULL));
}
