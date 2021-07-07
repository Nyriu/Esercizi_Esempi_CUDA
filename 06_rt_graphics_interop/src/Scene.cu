#include "Scene.h"
#include <iostream>

//void addShape(ImplicitShape* shape); // TODO
void Scene::addShape(Sphere* sph) {
  spheres_num_++;
  shapes_.push_back(sph);
}

void Scene::addLight(Light* light) {
  lights_num_++;
  lights_.push_back(light);
}

void Scene::addAmbientLight(AmbientLight* light) {
  ambientLight_ = light;
}

__host__ void Scene::shapes_to_device() {
  size_t total_size = 0;
  for (Sphere *sph : shapes_) {
    total_size += sizeof(*sph);
  }
  // Static allocation on device memory
  HANDLE_ERROR(
      cudaMalloc((void**)&devSpheres_, total_size)
      );

  int offset = 0;
  for (Sphere *sph : shapes_) {
    // Copy from host to device
    HANDLE_ERROR(
        cudaMemcpy((void*)(devSpheres_+offset), (void*)sph, sizeof(*sph), cudaMemcpyHostToDevice)
        );
    offset++;
  }

  if (offset != spheres_num_) {
    std::cout << "ERROR"
      "offset = " << offset <<
      "spheres_num_ = " << spheres_num_ <<
     std::endl;
    exit(1);
  }
}

__host__ void Scene::lights_to_device() {
  if (lights_num_ > 0) {
    size_t total_size = 0;
    for (Light *lgt : lights_) {
      total_size += sizeof(*lgt);
    }
    // Static allocation on device memory
    HANDLE_ERROR(
        cudaMalloc((void**)&devLights_, total_size)
        );

    int offset = 0;
    for (Light *lgt : lights_) {
      // Copy from host to device
      HANDLE_ERROR(
          cudaMemcpy((void*)(devLights_+offset), (void*)lgt, sizeof(*lgt), cudaMemcpyHostToDevice)
          );
      offset++;
    }

    if (offset != lights_num_) {
      std::cout << "ERROR"
        "offset = " << offset <<
        "lights_num_ = " << lights_num_ <<
        std::endl;
      exit(1);
    }
  }
  if (hasAmbientLight()) {
    HANDLE_ERROR(
        cudaMalloc((void**)&devAmbLight_, sizeof(AmbientLight))
        );
    HANDLE_ERROR(
        cudaMemcpy((void*)devAmbLight_, (void*)ambientLight_, sizeof(AmbientLight), cudaMemcpyHostToDevice)
        );
  }
}

__host__ Scene* Scene::to_device() {
  if (spheres_num_ > 0) {
    shapes_to_device();
  }
  if (lights_num_ > 0) {
    lights_to_device();
  }

  // Static allocation on device memory
  HANDLE_ERROR(
      cudaMalloc((void**)&devPtr_, sizeof(Scene))
      );
  // Copy from host to device
  HANDLE_ERROR(
      cudaMemcpy((void*)devPtr_, (void*)this, sizeof(Scene), cudaMemcpyHostToDevice)
      );
  return devPtr_;
}
