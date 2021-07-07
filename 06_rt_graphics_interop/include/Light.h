#ifndef LIGHT_H
#define LIGHT_H

#include "common.h"

class Light {
  protected:
    point3 position_ = color(0);
    color color_ = color(1);
    color intensity_ = color(80);
  public:
    //    virtual ~Light() {}

    // virtual cannot be used in CUDA if instance is generated on host
    // if virtual needed must rework to generate objects only on device
    // or re-instanciate them (too heavy/uneficcient)
    //__device__ virtual point3 getPosition() const { return position_; }
    //__device__ virtual color getColor() const { return color_; }
    //__device__ virtual color getIntensity() const { return intensity_; }
    __device__ point3 getPosition() const { return position_; }
    __device__ color getColor() const { return color_; }
    __device__ color getIntensity() const { return intensity_; }
};

class PointLight : public Light {
  public:
    PointLight(const point3& position, const color& c) {
      position_ = position;
      color_ = c;
    }
    PointLight(const point3& position, const color& c, const float& intensity) {
      position_ = position;
      color_ = c;
      intensity_ = color(intensity);
    }
};

class AmbientLight : public Light {
  public:
    AmbientLight(const color& c) {
      color_ = c;
      intensity_ = color(.17);
    }
    AmbientLight() {
      color_ = color(1);
      intensity_ = color(.17);
    }
    AmbientLight(const color& c, const float& intensity) {
      color_ = c;
      intensity_ = color(intensity, intensity,  intensity);
    }
    __device__ point3 getPosition() const { printf("AmbLight has no position!!"); }
};


#endif
