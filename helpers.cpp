#include <cmath>
#include <vector>

typedef std::vector<double> Vec3D;

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    // #pragma omp simd reduction(+:result) 
    for (size_t n = 0; n < a.size(); ++n){
        result += a[n] * b[n];
    }
    return result;
}

Vec3D operator+(const Vec3D& a, const Vec3D& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3D operator-(const Vec3D& a, const Vec3D& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3D operator/(const Vec3D& a, const int L) {
    return {a[0] / static_cast<double>(L), a[1] / static_cast<double>(L), a[2] / static_cast<double>(L)};
}

