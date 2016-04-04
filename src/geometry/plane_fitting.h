#ifndef PLANE_FITTING_H
#define PLANE_FITTING_H

#include <vector_types.h>

namespace dart {

void fitPlane(float3 & planeNormal,
              float & planeIntercept,
              const float4 * dObsVertMap,
              const float4 * dObsNormMap,
              const int width,
              const int height,
              const float distanceThreshold,
              const float normalThreshold,
              const int maxIters,
              const float regularization,
              int * dbgAssociated = 0);

void fitPlaneIter(const float3 planeNormal,
                  const float planeIntercept,
                  const float4 * dObsVertMap,
                  const float4 * dObsNormMap,
                  const int width,
                  const int height,
                  const float distanceThreshold,
                  const float normalThreshold,
                  const float regularization,
                  float * dResult,
                  int * dbgAssocaited);

void subtractPlane_(float4 * dObsVertMap,
                   float4 * dObsNormMap,
                   const int width,
                   const int height,
                   const float3 planeNormal,
                   const float planeIntercept,
                   const float distanceThreshold,
                   const float normalThreshold);

}

#endif // PLANE_FITTING_H
