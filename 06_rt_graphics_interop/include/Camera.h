#ifndef CAMERA_H
#define CAMERA_H

#include "common.h"
#include "Ray.h"

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

    __device__ Ray generate_ray(float u, float v) const; // input NDC Coords
};

#endif
