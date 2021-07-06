#include "Tracer.h"

__device__ color Tracer::trace(const Ray *r, const Scene *sce) const {
  //return color( (r->dir_.x + 1)+.5, (r->dir_.y + 1)+.5, (r->dir_.z + 1)+.5);
  HitRecord ht = sphereTrace(r,sce);

  if (ht.isMiss() || ht.t_ >= max_distance_) // Background
    //return color(0);
    return color((r->dir_.x + 1)+.5, (r->dir_.y + 1)+.5, (r->dir_.z + 1)+.5);
  color c = shade(&ht, sce);
  return c;
};


__device__ HitRecord Tracer::sphereTrace(const Ray *r, const Scene *sce) const {
  if (sce->getShapesNum() <= 0) return HitRecord();

  float t=0;
  float minDistance = infinity;
  float d = infinity;
  Sphere *hit_shape = nullptr;
  while (t < max_distance_) {
    minDistance = infinity; // TODO REMOVEME
    Sphere *sph = sce->getShapes();
    for (int i=0; i < sce->getShapesNum(); i++) {
      d = sph->getDist(r->at(t));
      if (d < minDistance) {
        minDistance = d;
        hit_shape = sph;
      }
      sph++;
    }
    // did we intersect the shape?
    if (minDistance < 0 ||  minDistance <= hit_threshold_ * t) {
      point3 p = r->at(t);
      return HitRecord(r, hit_shape, t, p, hit_shape->getNormalAt(p));
    }
    t += minDistance;
  }
  return HitRecord(t);
}

__device__ color Tracer::shade(const HitRecord *ht, const Scene *sce) const {
  point3 p = ht->p_;
  vec3 v = ht->r_->dir_;
  vec3 n = ht->n_;
  const Sphere *sph = ht->sphere_;

  color outRadiance(0);

  vec3 l;
  float nDotl;
  color brdf;

  bool shadow;
  float dist2 = 0;

  // TODO "materials"
  color cdiff = sph->getAlbedo();
  //float shininess_factor = shape->getShininess(p);
  //Color cspec = shape->getSpecular(p);
  //color cdiff(.5,.3,.8);
  float shininess_factor = 2;
  color cspec(0.04);

  // TODO scene lights
  if (sce->getLightsNum() > 0) {
    Light *lgt = sce->getLights();

    for (int i=0; i < sce->getLightsNum(); i++) {
      l = (lgt->getPosition() - p); // TODO // HERE ERROR if PointLight : public Light
      //l = (point3(5,4,3) - p);
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
        shadow = false; // TODO trace shadow
        //color lightColor(1);
        //color lightIntensity(80);
        color lightColor = lgt->getColor();
        color lightIntensity = lgt->getIntensity();

        outRadiance += color(1-shadow) * brdf * lightColor * lightIntensity * nDotl
          / (float) (4 * dist2) // with square falloff
          ;
      }
      lgt++;
    }
  }
  // TODO scene amb light
  //if (scene_->hasAmbientLight()) {
  //Light* ambientLight = scene_->getAmbientLight();
  color ambientColor(1);
  color ambientIntensity(.17);
  outRadiance += ambientColor * ambientIntensity * cdiff;
  //}
  return glm::clamp(outRadiance, color(0,0,0), color(1,1,1));
}


