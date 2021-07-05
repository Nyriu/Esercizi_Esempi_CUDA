#ifndef TRACER_H
#define TRACER_H

#include "common.h"
#include "Ray.h"
#include "Scene.h"

// TODO It this the right place?
class HitRecord {
  //private: // TODO + getters and setters
  public:
    const Ray *r_ = nullptr;
    //const ImplicitShape *shape_ = nullptr; // TODO
    const Sphere *sphere_ = nullptr; // TODO use impl shapes
    float t_ = -1;         // hit time
    point3 p_ = point3(0); // hit point
    vec3 n_ = vec3(0);     // normal at p_
    //color alb_ = color(0); // albedo at p_
    //const float d_ = -1; // distance from surface
  public:
    __device__ HitRecord() : // Empty HitRecord // that's a miss
      r_(nullptr),
      //shape_(nullptr),
      sphere_(nullptr),
      t_(-1), p_(point3()), n_(vec3())//, alb_(Color(.35))
  {}
    __device__ HitRecord(const float t_max) : // that's a miss
      r_(nullptr),
      //shape_(nullptr),
      sphere_(nullptr),
      t_(t_max), p_(point3()), n_(vec3())//, alb_(Color(.35))
  {}
    __device__ HitRecord(
        const Ray *r,
        //const ImplicitShape *shape,
        const Sphere *sphere,
        const float t,
        const point3 p,
        const vec3 n//, //const color alb//, const float d
        ) : r_(r),
    //shape_(shape),
    sphere_(sphere),
    t_(t), p_(p), n_(n)//, alb_(alb)
  {}

    __device__ bool isMiss() const {
      return
        r_     == nullptr ||
        //shape_ == nullptr ||
        sphere_ == nullptr ||
        t_ < 0             //|| other info?
        ;
    }

    // TODO
    //HitRecord& operator=(Tracer::HitRecord ht) {
    //  r_     = ht.r_;
    //  shape_ = ht.shape_;
    //  t_     = ht.t_;
    //  p_     = ht.p_;
    //  n_     = ht.n_;
    //  alb_   = ht.alb_;
    //  return *this;
    //}

    //friend std::ostream& operator<<(std::ostream& out, const HitRecord& ht) {
    //  // TODO
    //  return out << "{"
    //    "\n\tr = "     << ht.r_       <<
    //    //"\n\tshape = " << ht.shape_   <<
    //    //"\n\sphere = " << ht.sphere_   <<
    //    "\n\tt = "     << ht.t_       <<
    //    //"\n\tp = "     << ht.p_       <<
    //    //"\n\tn = "     << ht.n_       <<
    //    //"\n\talb = "   << ht.alb_     <<
    //    "\n}";
    //}

};



class Tracer {
  public:
    __device__ color trace(const Ray *r, const Scene *sce) const;

  private:
    static constexpr float max_distance_= 100;
    static constexpr float hit_threshold_ = 10e-6; // min distance to signal a ray-surface hit

    __device__ HitRecord sphereTrace(const Ray *r, const Scene *sce) const;
    __device__ color shade(const HitRecord *ht, const Scene *sce) const;

    // TODO
    //  bool sphereTraceShadow(const Ray& r, const ImplicitShape *shapeToShadow);
};

#endif
