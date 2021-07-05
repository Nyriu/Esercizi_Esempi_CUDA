#ifndef SPHERE_H
#define SPHERE_H

#include "common.h"

class Sphere {
  private:
    point3 center_ = point3(0);
    float radius_ = 0.5;
    color albedo_ = color(0.35);

    static constexpr float gradient_delta_ = 10e-5; // delta used to compute gradient (normal)

  public:
    Sphere(const float& radius) : radius_(radius) {}
    Sphere(const float& radius, const color& albedo) : radius_(radius), albedo_(albedo) {}
    Sphere(const point3& center, const float& radius, const color& albedo) : center_(center), radius_(radius), albedo_(albedo) {}
    Sphere(const point3& center, const float& radius) : center_(center), radius_(radius) {}

    __device__ float getDist(const point3& point) const {
      point3 p = point - center_; // very basic from world to local
      return glm::length(p) - radius_;
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
    __device__ color getAlbedo() const { return albedo_; }
};

#endif
