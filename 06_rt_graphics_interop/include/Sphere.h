#ifndef SPHERE_H
#define SPHERE_H

#include "common.h"

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

#endif
