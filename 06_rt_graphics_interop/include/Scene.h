#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "common.h"
#include "Sphere.h"
#include "Camera.h"
#include "Light.h"
//#include "Ray.h"

class Scene {
  public:
    using Shapes = std::vector<Sphere*>; 
    using Lights = std::vector<Light*>;

  private:
    Shapes shapes_;
    Lights lights_;
    AmbientLight* ambientLight_ = nullptr;

    Scene *devPtr_ = nullptr;
    Sphere *devSpheres_ = nullptr;
    int spheres_num_ = 0; // number of spheres
  public:
    Scene() = default;
    Scene(Shapes shapes, Lights lights) : shapes_(shapes), lights_(lights) {}

    //void addShape(ImplicitShape* shape); // TODO shape hierarchy
    void addShape(Sphere* sph);
    void addLight(Light* light);
    void addAmbientLight(AmbientLight* light);

    bool hasAmbientLight() const { return ambientLight_ != nullptr; }

    __device__ Sphere* getShapes() const { return devSpheres_; }
    __device__ int getShapesNum() const { return spheres_num_; }
    //__device__ Lights getLights() const { return lights_; } // TODO move lights data to device
    //__device__ Light* getAmbientLight() const { return ambientLight_; }

  private:
    __host__ void shapes_to_device();
    __host__ void lights_to_device(); // TODO NEXT
  public:
    /** moves Scene's data to device and returns the device pointer to scene **/
    __host__ Scene* to_device();
};

#endif
