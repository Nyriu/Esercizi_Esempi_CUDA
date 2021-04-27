#ifndef COLOR_H
#define COLOR_H

#include <fstream>
#include <string>

#define imin(a,b) (a<b?a:b)
#define imax(a,b) (a>b?a:b)

inline float clamp_(const float val, const float minVal, const float maxVal) {
  return imin(imax(val, minVal), maxVal);
}
//inline float clamp_lower(const float val, const float minVal) {
//  return imax(val, minVal);
//}

class Color {
  private:
    float r_,g_,b_;
    // alpha ?

  public:
    Color() = default;

    Color(float c) {
      r_ = c;
      g_ = c;
      b_ = c;
    }
    Color(float r, float g, float b) :
      r_(r),
      g_(g),
      b_(b) {}

    float r() const { return r_; }
    float g() const { return g_; }
    float b() const { return b_; }

    int r255() const { return clamp_(static_cast<int>(r_ * 255.999f), 0, 255); }
    int g255() const { return clamp_(static_cast<int>(g_ * 255.999f), 0, 255); }
    int b255() const { return clamp_(static_cast<int>(b_ * 255.999f), 0, 255); }

    Color& operator +=(const Color &c){
      r_ += c.r_;
      g_ += c.g_;
      b_ += c.b_;
      return *this;
    }

    Color& clamp(const float minVal, float maxVal) {
      r_ = clamp_(r_, minVal, maxVal);
      g_ = clamp_(g_, minVal, maxVal);
      b_ = clamp_(b_, minVal, maxVal);
      return *this;
    }

    std::string to_string() const {
      return "Color: " +
        std::to_string(r_) + " " +
        std::to_string(g_) + " " +
        std::to_string(b_);
    }


    // Friends
    friend inline Color operator+(const Color& c, const float& f);
    friend inline Color operator+(const float& f, const Color& c);
    friend inline Color operator+(const Color& c1, const Color& c2);

    friend inline Color operator-(const Color& c1, const Color& c2);

    friend inline Color operator*(const Color& c, const float& f);
    friend inline Color operator*(const float& f, const Color& c);
    friend inline Color operator*(const Color& c1, const Color& c2);

    friend inline Color operator/(const Color& c, const float& f);
    friend inline Color operator/(const float& f, const Color& c);
};


inline Color operator+(const Color& c, const float& f) {
  return Color(
      c.r_ + f,
      c.g_ + f,
      c.b_ + f);
}
inline Color operator+(const float& f, const Color& c) {
  return c+f;
}
inline Color operator+(const Color& c1, const Color& c2) {
  return Color(
      c1.r_ + c2.r_,
      c1.g_ + c2.g_,
      c1.b_ + c2.b_);
}


inline Color operator-(const Color& c1, const Color& c2) {
  return Color(
      c1.r_ - c2.r_,
      c1.g_ - c2.g_,
      c1.b_ - c2.b_);
}


inline Color operator*(const Color& c, const float& f) {
  return Color(
      c.r_ * f,
      c.g_ * f,
      c.b_ * f);
}
inline Color operator*(const float& f, const Color& c) {
  return c*f;
}
inline Color operator*(const Color& c1, const Color& c2) {
  return Color(
      c1.r_ * c2.r_,
      c1.g_ * c2.g_,
      c1.b_ * c2.b_);
}

inline Color operator/(const Color& c, const float& f) {
  return Color(
      c.r_ / f,
      c.g_ / f,
      c.b_ / f);
}
inline Color operator/(const float& f, const Color& c) {
  return c/f;
}


inline void write_color(std::ofstream &out, const Color& pixel_color) {
  out << pixel_color.r255() << " "
      << pixel_color.g255() << " "
      << pixel_color.b255() << "\n";
}

inline std::ostream& operator<<(std::ostream &out, const Color &c) {
  return out
    << c.r255() << " "
    << c.g255() << " "
    << c.b255() << " ";
}

inline Color pow(const Color& base_c, const Color& exp_c) {
  return Color(
      base_c.r() * exp_c.r(),
      base_c.g() * exp_c.g(),
      base_c.b() * exp_c.b()
      );
}

#endif
