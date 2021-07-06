#ifndef LIGHT_H
#define LIGHT_H

#include "common.h"

/**
 * ATM Class Hierarchy does NOT work
 * Must understand how to move correctly to device PointLight when it's 'under' Light
 **/

//class Light {
//  protected:
//    point3 position_ = color(0);
//    color color_ = color(1);
//    color intensity_ = color(1);
//  public:
//    virtual ~Light() {}
//
//    __device__ virtual point3 getPosition() const { return position_; }
//
//    __device__ virtual color getColor() const { return color_; }
//
//    __device__ virtual color getIntensity() const { return intensity_; }
//};

//class PointLight { //: public Light {
class Light { //: public Light {
  protected:
    point3 position_ = color(0);
    color color_ = color(1);
    color intensity_ = color(80);
  public:
    Light(const point3& position, const color& c) {
    //PointLight(const point3& position, const color& c) {
      position_ = position;
      color_ = c;
    }
    Light(const point3& position, const color& c, const float intensity) {
    //PointLight(const point3& position, const color& c, const float& intensity) {
      position_ = position;
      color_ = c;
      intensity_ = color(intensity);
    }
    __device__ point3 getPosition() const { return position_; }

    __device__ color getColor() const { return color_; }

    __device__ color getIntensity() const { return intensity_; }
};

//class AmbientLight : public Light {
////  public:
////    AmbientLight(const color& color, const float& intensity) {
////      position_ = point3(0);
////      color_ = color;
////      //intensity_ = color(intensity, intensity,  intensity);
////    }
//};


#endif
