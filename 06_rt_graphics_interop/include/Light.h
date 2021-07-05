#ifndef LIGHT_H
#define LIGHT_H

#include "common.h"

class Light {
  protected:
    point3 position_;
    color color_;
    float intensity_;
  public:
    virtual ~Light() {}

    virtual point3 getPosition() const { return position_; }

    virtual color getColor() const { return color_; }

    virtual float getIntensity() const { return intensity_; }
};

class PointLight : public Light {
  public:
    PointLight(const point3& position, const color& color, const float& intensity) {
      position_ = position;
      color_ = color;
      intensity_ = intensity;
    }
};

class AmbientLight : public Light {
  public:
    AmbientLight(const color& color, const float& intensity) {
      position_ = point3(0);
      color_ = color;
      intensity_ = intensity;
    }
};


#endif
