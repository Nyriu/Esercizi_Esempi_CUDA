#ifndef RAY_H
#define RAY_H

#include "common.h"

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

#endif
