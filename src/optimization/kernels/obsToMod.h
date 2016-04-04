#ifndef OBSTOMOD_H
#define OBSTOMOD_H

#include "geometry/grid_3d.h"
#include "geometry/SE3.h"
#include "optimization/optimization.h"
#include "model/mirrored_model.h"
#include "util/dart_types.h"

namespace dart {

void normEqnsObsToMod(const int dims,
                      const float4 * dObsVertMap,
                      const int width,
                      const int height,
                      const MirroredModel & model,
                      const OptimizationOptions & opts,
                      DataAssociatedPoint * dPts,
                      int nElements,
                      float * dResult,
                      float4 * debugJs = 0);

void normEqnsObsToModReduced(const int dims,
                             const int reductionDims,
                             const float * d_dtheta_dalpha,
                             const float4 * dObsVertMap,
                             const int width,
                             const int height,
                             const MirroredModel & model,
                             const OptimizationOptions & opts,
                             DataAssociatedPoint * dPts,
                             int nElements,
                             float * dResult);

void normEqnsObsToModParamMap(const int dims,
                              const int reductionDims,
                              const int * dMapping,
                              const float4 * dObsVertMap,
                              const int width,
                              const int height,
                              const MirroredModel & model,
                              const OptimizationOptions & opts,
                              DataAssociatedPoint * dPts,
                              int nElements,
                              float * dResult);

void normEqnsObsToModAndPinkyPos(const int dims,
                                 const float4 * dObsVertMap,
                                 const float4 * dObsNormMap,
                                 const int width,
                                 const int height,
                                 const SE3 T_mc,
                                 const SE3 * dT_fms,
                                 const SE3 * dT_mfs,
                                 const SE3 T_pm,
                                 const int * dSdfFrames,
                                 const Grid3D<float> * dSdfs,
                                 const int nSdfs,
                                 const float distanceThreshold,
                                 const float normalThreshold,
                                 const int * dDependencies,
                                 const JointType * dJointTypes,
                                 const float3 * dJointAxes,
                                 const int * dFrameParents,
                                 const float planeOffset,
                                 const float3 planeNormal,
                                 DataAssociatedPoint * dPts,
                                 int * dLastElement,
                                 int * hLastElement,
                                 float * dResult,
                                 int * debugDataAssociation,
                                 float * debugError,
                                 float4 * debugNorm);

void errorAndDataAssociation(const float4 * dObsVertMap,
                             const float4 * dObsNormMap,
                             const int width,
                             const int height,
                             const MirroredModel & model,
                             const OptimizationOptions & opts,
                             DataAssociatedPoint * dPts,
                             int * dLastElement,
                             int * hLastElement,
                             int * debugDataAssociation = 0,
                             float * debugError = 0,
                             float4 * debugNorm = 0);

void errorAndDataAssociationMultiModel(const float4 * dObsVertMap,
                                       const float4 * dObsNormMap,
                                       const int width,
                                       const int height,
                                       const int nModels,
                                       const SE3 * T_mcs,
                                       const SE3 * const * T_fms,
                                       const int * const * sdfFrames,
                                       const Grid3D<float> * const * sdfs,
                                       const int * nSdfs,
                                       const float * distanceThresholds,
                                       const float * normalThresholds,
                                       const float * planeOffsets,
                                       const float3 * planeNormals,
                                       int * lastElements,
                                       DataAssociatedPoint * * pts,
                                       int * dDebugDataAssociation,
                                       float * dDebugError,
                                       float4 * dDebugNorm,
                                       cudaStream_t stream = 0);


__device__ void getErrorJacobianOfModelPoint(float * J, const float4 & point_m, const int frame, const float3 & errorGrad3D_m,
                                             const int dims, const int * dependencies, const JointType * jointTypes,
                                             const float3 * jointAxes, const SE3 * T_fms, const SE3 * T_mfs);

}

#endif // OBSTOMOD_H
