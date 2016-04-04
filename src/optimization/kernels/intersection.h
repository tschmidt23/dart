#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "geometry/grid_3d.h"
#include "geometry/SE3.h"
#include "model/mirrored_model.h"
#include "util/dart_types.h"

namespace dart {

int countSelfIntersections(const float4 * testSites,
                           const int nSites,
                           const SE3 * T_mfs,
                           const SE3 * T_fms,
                           const int * sdfFrames,
                           const Grid3D<float> * sdfs,
                           const int nSdfs,
                           const int * potentialIntersection);

void normEqnsSelfIntersection(const float4 * testSites,
                              const int nSites,
                              const int dims,
                              const MirroredModel & model,
                              const int * potentialIntersection,
                              float * result,
                              float * debugError = 0);

void normEqnsSelfIntersectionReduced(const float4 * testSites,
                                     const int nSites,
                                     const int fullDims,
                                     const int redDims,
                                     const MirroredModel & model,
                                     const float * dtheta_dalpha,
                                     const int * potentialIntersection,
                                     float * result);

void normEqnsSelfIntersectionParamMap(const float4 * testSites,
                                      const int nSites,
                                      const int fullDims,
                                      const int redDims,
                                      const MirroredModel & model,
                                      const int * dMapping,
                                      const int * potentialIntersection,
                                      float * result);

void normEqnsIntersection(const float4 * testSites,
                          const int nSites,
                          const int dims,
                          const SE3 T_ds,
                          const SE3 T_sd,
                          const MirroredModel & srcModel,
                          const MirroredModel & dstModel,
                          float * result,
                          float * debugError = 0);

void normEqnsIntersectionReduced(const float4 * testSites,
                                 const int nSites,
                                 const int fullDims,
                                 const int redDims,
                                 const SE3 T_ds,
                                 const SE3 T_sd,
                                 const MirroredModel & srcModel,
                                 const MirroredModel & dstModel,
                                 const float * dtheta_dalpha_src,
                                 float * result);

void normEqnsIntersectionParamMap(const float4 * testSites,
                                  const int nSites,
                                  const int fullDims,
                                  const int redDims,
                                  const SE3 T_ds,
                                  const SE3 T_sd,
                                  const MirroredModel & srcModel,
                                  const MirroredModel & dstModel,
                                  const int * dMapping_src,
                                  float * result);

void initDebugIntersectionError(float * debugError,
                                const int nSites);

}

#endif // INTERSECTION_H
