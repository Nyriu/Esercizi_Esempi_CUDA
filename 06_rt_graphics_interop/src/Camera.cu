#include "Camera.h"

__device__ Ray Camera::generate_ray(float u, float v) const { // input NDC Coords
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
