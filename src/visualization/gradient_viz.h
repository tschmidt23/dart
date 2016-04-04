#ifndef GRADIENT_VIZ_H
#define GRADIENT_VIZ_H

#include <vector_types.h>

namespace dart {

void visualizeImageGradient(const float2 * imgGradient, uchar3 * gradientViz, const int width, const int height, const float minMag = 0.0, const float maxMag = 1.0);

}

#endif // GRADIENT_VIZ_H
