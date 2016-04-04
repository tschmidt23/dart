#ifndef RAYCAST_H
#define RAYCAST_H

#include "geometry/grid_3d.h"
#include "geometry/SE3.h"

namespace dart {

void raycastPrediction(float2 fl,
                       float2 pp,
                       const int width,
                       const int height,
                       const int modelNum,
                       const SE3 T_mc,
                       const SE3 * T_fms,
                       const SE3 * T_mfs,
                       const int * sdfFrames,
                       const Grid3D<float> * sdfs,
                       const int nSdfs,
                       float4 * prediction,
                       const float levelSet,
                       cudaStream_t stream);

void raycastPredictionDebugRay(float2 fl,
                               float2 pp,
                               const int x,
                               const int y,
                               const int width,
                               const int modelNum,
                               const SE3 T_mc,
                               const SE3 * T_fms,
                               const SE3 * T_mfs,
                               const int * sdfFrames,
                               const Grid3D<float> * sdfs,
                               const int nSdfs,
                               float4 * prediction,
                               const float levelSet,
                               float3 * boxIntersects,
                               float2 * raySteps,
                               const int maxRaySteps);

}


#endif // RAYCAST_H
