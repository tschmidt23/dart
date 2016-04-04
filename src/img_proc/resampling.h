#ifndef RESAMPLING_H
#define RESAMPLING_H

#include <vector_types.h>

namespace dart {

void downsampleAreaAverage(const float * imgIn, const uint2 dimIn, float * imgOut, const int factor);

void downsampleAreaAverage(const uchar3 * imgIn, const uint2 dimIn, uchar3 * imgOut, const int factor);

void downsampleAreaAverage(const uchar4 * imgIn, const uint2 dimIn, uchar4 * imgOut, const int factor);

void downsampleNearest(const float * imgIn, const uint2 dimIn, float * imgOut, const int factor);

void downsampleNearest(const uchar3 * imgIn, const uint2 dimIn, uchar3 * imgOut, const int factor);

void downsampleMin(const float * imgIn, const uint2 dimIn, float * imgOut, const int factor, bool ignoreZero = true);

}

#endif // RESAMPLING_H

