#include "ostream_operators.h"

//namespace dart {

std::ostream & operator<<(std::ostream & os, const float3 & v) {
    os << v.x << ", " << v.y << ", " << v.z;
    return os;
}

std::ostream & operator<<(std::ostream & os, const float4 & v) {
    os << v.x << ", " << v.y << ", " << v.z << ", " << v.w;
    return os;
}

std::ostream & operator<<(std::ostream & os, const dart::SE3 & v) {
    os << v.r0 << std::endl << v.r1 << std::endl << v.r2;
    return os;
}

std::ostream & operator<<(std::ostream & os, const dart::se3 & v) {
    os << v.p[0] << ", " << v.p[1] << ", " << v.p[2] << "; " << v.p[3] << ", " << v.p[4] << ", " << v.p[5];
    return os;
}

//} // namespace dart
