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

/*
 * src_T_mfs, src_T_fms, dst_T_mfs, dst_T_fms are all arrays of transforms on the
 * gpu as computed by computeKinematicsBatchGPU, they are both nConfigs arrays
 * of model-frame and frame-model transforms of their respective models
 *
 * result and resultsdf are nconfigs * nsites long arrays on the gpu to be used
 * for results of distance checking. (result + config_idx * nsites)[site_idx] is
 * the way to access a particular result for joint config config_idx and testSite
 * site_idx. (Note that the memory should be copied over to the cpu first if that
 * is where you intend to read the memory.)
 * */
void normEqnsIntersectionRaw(const float4 * testSites,
                          const int nSites,
                          const SE3 T_ds,
                          const SE3 T_sd,
                          const SE3 *src_T_mfs,
                          const SE3 *src_T_fms,
                          const int nFramesSrc, // number of frames per joint config
                          const SE3 *dst_T_mfs,
                          const SE3 *dst_T_fms,
                          const int nFramesDst, // number of frames per joint config
                          const int nConfigs,// number of total joint configs
                          const MirroredModel & srcModel,
                          const MirroredModel & dstModel,
                          // distance to closest object
                          // results array is a block of results (per joint config) with layout paralleling the joint configs 
                          // (nsites results for jtconfig 1 at result, nsites results for jtconfig 2 at result + nsites)
                          float * result, // should be an nsites * nconfigs long array that has already been allocated on the gpu
                          // index of the sdf corresponding to above
                          int * resultsdf, // should be an nsites * nconfigs long array that has already been allocated on the gpu
                          float * debugError = 0);

// same as above except your destination to intersect with is a single object instead of a
// multitude of joint configurations
void normEqnsIntersectionRawSingleTgt(const float4 * testSites,
                          const int nSites,
                          const SE3 T_ds,
                          const SE3 T_sd,
                          const SE3 *src_T_mfs,
                          const SE3 *src_T_fms,
                          const int nFramesSrc, // number of frames per joint config
                          const SE3 *dst_T_mf,
                          const SE3 *dst_T_fm,
                          const int nFramesDst, // number of frames per joint config
                          const int nConfigs,// number of total joint configs
                          const MirroredModel & srcModel,
                          const MirroredModel & dstModel,
                          // distance to closest object
                          // results array is a block of results (per joint config) with layout paralleling the joint configs 
                          // (nsites results for jtconfig 1 at result, nsites results for jtconfig 2 at result + nsites)
                          float * result, // should be an nsites * nconfigs long array that has already been allocated on the gpu
                          // index of the sdf corresponding to above
                          int * resultsdf, // should be an nsites * nconfigs long array that has already been allocated on the gpu
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
