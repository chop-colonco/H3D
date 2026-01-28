#pragma once
#include <cmath>
#include <vector>

typedef std::vector<double> Vec3D;

double dot(const std::vector<double>& a, const std::vector<double>& b);
Vec3D operator+(const Vec3D& a, const Vec3D& b);
Vec3D operator-(const Vec3D& a, const Vec3D& b);
Vec3D operator/(const Vec3D& a, const int L);