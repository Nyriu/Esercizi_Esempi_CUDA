#ifndef TRACER_H
#define TRACER_H

#include "common.h"
#include "Ray.h"
#include "Sphere.h"

class Tracer {
  public:
    __device__ color trace(const Ray *r, const Sphere *sph) const {
      //return color( (r->dir_.x + 1)+.5, (r->dir_.y + 1)+.5, (r->dir_.z + 1)+.5);
      float t = sphereTrace(r,sph);

      if (t >= max_distance_) // Background
        //return color(0);
        return color((r->dir_.x + 1)+.5, (r->dir_.y + 1)+.5, (r->dir_.z + 1)+.5);
        //return color((r->dir_.x + 1)+.5,0,0);
        //return color(0,(r->dir_.y + 1)+.5,0);
      point3 p = r->at(t);
      vec3   d = r->dir_;
      color c = shade(p, d, sph);
      return c;
    };
  private:
    static constexpr float max_distance_= 100;
    static constexpr float hit_threshold_ = 10e-6; // min distance to signal a ray-surface hit

    __device__ float sphereTrace(const Ray *r, const Sphere *sph) const {
      // // DEBUG STUFF
      // float u = ((threadIdx.x + blockIdx.x * blockDim.x) + .5) / ((float) IMG_W -1); // NDC Coord
      // float v = ((threadIdx.y + blockIdx.y * blockDim.y) + .5) / ((float) IMG_H -1); // NDC Coord
      // bool enable_print = (u == 0.5) && (v == 0.5); // img center
      // enable_print = false;
      // if (enable_print)
      //   printf("ray dir = (%f,%f,%f)\n", r->dir_.x, r->dir_.y, r->dir_.z);
      // // END // DEBUG STUFF

      float t=0;
      float minDistance = infinity;
      float d = infinity;
      while (t < max_distance_) {
        minDistance = infinity;
        d = sph->getDist(r->at(t));
        /// if (enable_print) printf("d = %f\n", d); // DEBUG STUFF
        if (d < minDistance) {
          minDistance = d;
          // if (enable_print) printf("mDist upd = %f\n", minDistance); // DEBUG STUFF
        }
        // did we intersect the shape?
        if (minDistance <= hit_threshold_ * t) {
          // if (enable_print) printf("hit at t = %f\n", t); // DEBUG STUFF
          return t;
        }
        t += minDistance;
      }
      return t;
    }

    __device__ color shade(const point3 p, const vec3 v, const Sphere *sph) const {
      vec3 n = sph->getNormalAt(p);

      // // DEBUG STUFF
      // float idx_u = ((threadIdx.x + blockIdx.x * blockDim.x) + .5) / ((float) IMG_W -1); // NDC Coord
      // float idx_v = ((threadIdx.y + blockIdx.y * blockDim.y) + .5) / ((float) IMG_H -1); // NDC Coord
      // bool enable_print = (idx_u == 0.5) && (idx_v == 0.5); // img center
      // bool enable_print = !true;
      // if (enable_print) printf("n = (%f,%f,%f)\n",
      //     (n.x+1)*.5,
      //     (n.y+1)*.5,
      //     (n.z+1)*.5);
      // // END // DEBUG STUFF

      color outRadiance(0);

      vec3 l;
      float nDotl;
      color brdf;

      bool shadow;
      float dist2 = 0;

      // TODO
      //Color cdiff = shape->getAlbedo(p);
      //float shininess_factor = shape->getShininess(p);
      //Color cspec = shape->getSpecular(p);
      color cdiff(.5,.3,.8);
      float shininess_factor = 2;
      color cspec(0.04);

      // TODO
      //for (const auto& light : scene_->getLights()) {
      //l = (light->getPosition() - p);
      l = (point3(5,4,3) - p);
      dist2 = glm::length(l); // squared dist
      l = glm::normalize(l);
      nDotl = glm::dot(n,l);

      if (nDotl > 0) {
        vec3 r = 2 * nDotl * n - l;
        float vDotr = glm::dot(v, r);
        brdf =
          cdiff / color(M_PI) +
          cspec * powf(vDotr, shininess_factor);

        // With shadows below
        //shadow = sphereTraceShadow(Ray(p,lightDir), shape);
        shadow = false; // TODO
        color lightColor(1);
        color lightIntensity(80);

        outRadiance += color(1-shadow) * brdf * lightColor * lightIntensity * nDotl
          / (float) (4 * dist2) // with square falloff
          ;
      }
      //}
      // TODO
      //if (scene_->hasAmbientLight()) {
      //Light* ambientLight = scene_->getAmbientLight();
      color ambientColor(1);
      color ambientIntensity(.17);
      outRadiance += ambientColor * ambientIntensity * cdiff;
      //}
      return glm::clamp(outRadiance, color(0,0,0), color(1,1,1));
    }

    // TODO
    //  bool sphereTraceShadow(const Ray& r, const ImplicitShape *shapeToShadow);
};

#endif
