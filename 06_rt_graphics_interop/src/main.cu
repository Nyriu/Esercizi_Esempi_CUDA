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
#include <glm/common.hpp>
#include <glm/vec3.hpp>
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

constexpr float infinity = std::numeric_limits<float>::max();


// Aliases
using vec3   = glm::vec3;
using point3 = glm::vec3;
using color  = glm::vec3;

// Globals
#define IMG_H 512
#define IMG_W 512

// TODO
//#define IMG_H 1080
//#define IMG_W IMG_H*16/9;


class Sphere {
  private:
    float radius_;
    point3 center_;

    static constexpr float gradient_delta_ = 10e-5; // delta used to compute gradient (normal)

  public:
    Sphere(const float& radius) : radius_(radius) {
      center_ = point3(0);
    }

    __device__ float getDist(const point3& point) const {
      return glm::length(point) - radius_;
    }

    __device__ vec3 getNormalAt(const point3& p) const {
      return glm::normalize(vec3(
            getDist(
              p+vec3(gradient_delta_,0,0)) - getDist(p + vec3(-gradient_delta_,0,0)),
            getDist(
              p+vec3(0,gradient_delta_,0)) - getDist(p + vec3(0,-gradient_delta_,0)),
            getDist(
              p+vec3(0,0,gradient_delta_)) - getDist(p + vec3(0,0,-gradient_delta_))
            ));
    }
};

class Ray {
  public:
    const point3 orig_;
    const vec3 dir_;

    __device__ Ray() : orig_(point3(0)), dir_(glm::normalize(vec3(0,0,-1))) { }
    __device__ Ray(const point3& origin, const vec3& direction) :
      orig_(origin), dir_(glm::normalize(direction)) {}

   __device__ const point3 at(float t) const {
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
      // Put coords in [-1,1] // (-1,-1) is bottom-left
      float su = u * 2 - 1; // Screen Coord
      float sv = v * 2 - 1; // Screen Coord

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
      vec3 dir = glm::normalize((p - orig));
      return Ray(orig, dir);
    }
};


class Tracer {
  public:
    __device__ color trace(const Ray *r, const Sphere *sph) const {
      //return color( (r->dir_.x + 1)+.5, (r->dir_.y + 1)+.5, (r->dir_.z + 1)+.5);
      float t = sphereTrace(r,sph);

      if (t >= max_distance_) // Background
        //return color(0);
        return color((r->dir_.x + 1)+.5, (r->dir_.y + 1)+.5, (r->dir_.z + 1)+.5);
        //return color((r->dir_.x + 1)+.5,0,0);
        //return color(0,(r->dir_.y + 1)+.5,0);
      point3 p = r->at(t);
      vec3   d = r->dir_;
      color c = shade(p, d, sph);
      return c;
    };
  private:
    static constexpr float max_distance_= 100;
    static constexpr float hit_threshold_ = 10e-6; // min distance to signal a ray-surface hit

    __device__ float sphereTrace(const Ray *r, const Sphere *sph) const {
      // // DEBUG STUFF
      // float u = ((threadIdx.x + blockIdx.x * blockDim.x) + .5) / ((float) IMG_W -1); // NDC Coord
      // float v = ((threadIdx.y + blockIdx.y * blockDim.y) + .5) / ((float) IMG_H -1); // NDC Coord
      // bool enable_print = (u == 0.5) && (v == 0.5); // img center
      // enable_print = false;
      // if (enable_print)
      //   printf("ray dir = (%f,%f,%f)\n", r->dir_.x, r->dir_.y, r->dir_.z);
      // // END // DEBUG STUFF

      float t=0;
      float minDistance = infinity;
      float d = infinity;
      while (t < max_distance_) {
        minDistance = infinity;
        d = sph->getDist(r->at(t));
        /// if (enable_print) printf("d = %f\n", d); // DEBUG STUFF
        if (d < minDistance) {
          minDistance = d;
          // if (enable_print) printf("mDist upd = %f\n", minDistance); // DEBUG STUFF
        }
        // did we intersect the shape?
        if (minDistance <= hit_threshold_ * t) {
          // if (enable_print) printf("hit at t = %f\n", t); // DEBUG STUFF
          return t;
        }
        t += minDistance;
      }
      return t;
    }

    __device__ color shade(const point3 p, const vec3 v, const Sphere *sph) const {
      vec3 n = sph->getNormalAt(p);

      // // DEBUG STUFF
      // float idx_u = ((threadIdx.x + blockIdx.x * blockDim.x) + .5) / ((float) IMG_W -1); // NDC Coord
      // float idx_v = ((threadIdx.y + blockIdx.y * blockDim.y) + .5) / ((float) IMG_H -1); // NDC Coord
      // bool enable_print = (idx_u == 0.5) && (idx_v == 0.5); // img center
      // bool enable_print = !true;
      // if (enable_print) printf("n = (%f,%f,%f)\n",
      //     (n.x+1)*.5,
      //     (n.y+1)*.5,
      //     (n.z+1)*.5);
      // // END // DEBUG STUFF

      color outRadiance(0);

      vec3 l;
      float nDotl;
      color brdf;

      bool shadow;
      float dist2 = 0;

      // TODO
      //Color cdiff = shape->getAlbedo(p);
      //float shininess_factor = shape->getShininess(p);
      //Color cspec = shape->getSpecular(p);
      color cdiff(.5,.3,.8);
      float shininess_factor = 2;
      color cspec(0.04);

      // TODO
      //for (const auto& light : scene_->getLights()) {
      //l = (light->getPosition() - p);
      l = (point3(5,4,3) - p);
      dist2 = glm::length(l); // squared dist
      l = glm::normalize(l);
      nDotl = glm::dot(n,l);

      if (nDotl > 0) {
        vec3 r = 2 * nDotl * n - l;
        float vDotr = glm::dot(v, r);
        brdf =
          cdiff / color(M_PI) +
          cspec * powf(vDotr, shininess_factor);

        // With shadows below
        //shadow = sphereTraceShadow(Ray(p,lightDir), shape);
        shadow = false; // TODO
        color lightColor(1);
        color lightIntensity(80);

        outRadiance += color(1-shadow) * brdf * lightColor * lightIntensity * nDotl
          / (float) (4 * dist2) // with square falloff
          ;
      }
      //}
      // TODO
      //if (scene_->hasAmbientLight()) {
      //Light* ambientLight = scene_->getAmbientLight();
      color ambientColor(1);
      color ambientIntensity(.17);
      outRadiance += ambientColor * ambientIntensity * cdiff;
      //}
      return glm::clamp(outRadiance, color(0,0,0), color(1,1,1));
    }

    // TODO
    //  bool sphereTraceShadow(const Ray& r, const ImplicitShape *shapeToShadow);
};

// TODO Tracer returns HitRecord and Shader uses it
//class HitRecord
//class Shader

static __global__ void kernel(uchar4 *ptr,
    const Camera *cam,
    const Sphere *sph,
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

  color c = trc->trace(&r, sph);

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
        const Camera *cam,
        const Sphere *sph,
              uchar4 *devPtr) {
      // --- Generate One Frame ---
      // TODO dims
      //dim3 grids(IMG_W/16, IMG_H/16);
      //dim3 threads(16,16);
      dim3 grids(IMG_W, IMG_H);
      dim3 threads(1);

      Camera *devCamPtr = nullptr;
      Sphere *devSphPtr = nullptr;
      Tracer *devTrcPtr = nullptr;

      // Static allocation on device memory
      HANDLE_ERROR(
          cudaMalloc((void**)&devCamPtr, sizeof(Camera))
          );
      HANDLE_ERROR(
          cudaMalloc((void**)&devSphPtr, sizeof(Sphere))
          );
      //HANDLE_ERROR(
      //    cudaMalloc((void**)&devTrcPtr, sizeof(Tracer))
      //    );

      // Copy from host to device
      HANDLE_ERROR(
          cudaMemcpy((void*)devCamPtr, (void*)cam, sizeof(Camera), cudaMemcpyHostToDevice)
          );
      HANDLE_ERROR(
        cudaMemcpy((void*)devSphPtr, (void*)sph, sizeof(Sphere), cudaMemcpyHostToDevice)
        );
      //HANDLE_ERROR(
      //  cudaMemcpy((void*)devTrcPtr, (void*)tracer_, sizeof(Tracer), cudaMemcpyHostToDevice)
      //  );

      kernel<<<grids,threads>>>(devPtr, devCamPtr, devSphPtr, devTrcPtr);

      cudaFree((void*)devCamPtr);
      cudaFree((void*)devSphPtr);
      cudaFree((void*)devTrcPtr);
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
  Sphere sph(2);
  Renderer renderer;
  renderer.render(&cam, &sph, devPtr);


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

