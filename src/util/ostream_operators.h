#ifndef OSTREAM_OPERATORS_H
#define OSTREAM_OPERATORS_H

#include <iostream>

#include "geometry/SE3.h"

//namespace dart {

std::ostream & operator<<(std::ostream & os, const float3 & v);

std::ostream & operator<<(std::ostream & os, const float4 & v);

std::ostream & operator<<(std::ostream & os, const dart::SE3 & v);

std::ostream & operator<<(std::ostream & os, const dart::se3 & v);

//} // namespace dart

#endif // OSTREAM_OPERATORS_H
