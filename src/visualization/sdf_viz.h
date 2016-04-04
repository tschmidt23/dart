#ifndef SDF_VIZ
#define SDF_VIZ

#include "geometry/grid_3d.h"
#include "geometry/SE3.h"
#include "model/mirrored_model.h"

namespace dart {

enum ColorRamp {
    ColorRampGrayscale,
    ColorRampHeatMap,
    ColorRampRedGreen
};

void visualizeModelSdfPlane(uchar3 * img,
                       const int width,
                       const int height,
                       const float2 origin,
                       const float2 size,
                       const SE3 & T_mc,
                       const SE3 * T_fms,
                       const int * sdfFrames,
                       const Grid3D<float> * sdfs,
                       const int nSdfs,
                       const float planeDepth,
                       const float minVal,
                       const float maxVal,
                       const ColorRamp ramp = ColorRampHeatMap);

void visualizeModelSdfPlaneProjective(uchar3 * img,
                                      const int width,
                                      const int height,
                                      const SE3 & T_mc,
                                      const SE3 * T_fms,
                                      const int * sdfFrames,
                                      const Grid3D<float> * sdfs,
                                      const int nSdfs,
                                      const float planeDepth,
                                      const float focalLength,
                                      const float minVal,
                                      const float maxVal,
                                      const ColorRamp ramp = ColorRampHeatMap);

void getSdfSlice(float * sdfSlice,
                 const int width,
                 const int height,
                 const float2 origin,
                 const float2 size,
                 const SE3 & T_sp,
                 const Grid3D<float> * deviceSdf);

void getModelSdfSlice(float * sdfSlice,
                      const int width,
                      const int height,
                      const float2 origin,
                      const float2 size,
                      const SE3 & T_pm,
                      const MirroredModel & model);

void getMultiModelSdfSlice(float * sdfSlice,
                           const int width,
                           const int height,
                           const float2 origin,
                           const float2 size,
                           const std::vector<SE3> & T_pm,
                           const std::vector<MirroredModel*> & models);

void getModelSdfPlaneProjective(float * sdf,
                                const int width,
                                const int height,
                                const SE3 & T_mc,
                                const SE3 * T_fms,
                                const int * sdfFrames,
                                const Grid3D<float> * sdfs,
                                const int nSdfs,
                                const float planeDepth,
                                const float focalLength);

void getModelSdfGradientPlaneProjective(float3 * grad,
                                        const int width,
                                        const int height,
                                        const SE3 & T_mc,
                                        const SE3 * T_fms,
                                        const int * sdfFrames,
                                        const Grid3D<float> * sdfs,
                                        const int nSdfs,
                                        const float planeDepth,
                                        const float focalLength);

void getObservationSdfPlane(float * sdfVals,
                            const int width,
                            const int height,
                            const Grid3D<float> * sdf,
                            const float planeDepth);

void getObservationSdfPlaneProjective(float * sdfVals,
                                      const int width,
                                      const int height,
                                      const Grid3D<float> * sdf,
                                      const float planeDepth,
                                      const float focalLength);

void visualizeDataAssociationPlane(uchar3 * img,
                                   const int width,
                                   const int height,
                                   const float2 origin,
                                   const float2 size,
                                   const SE3 & T_mc,
                                   const SE3 * T_fms,
                                   const int * sdfFrames,
                                   const Grid3D<float> * sdfs,
                                   const int nSdfs,
                                   const uchar3 * sdfColors,
                                   const float planeDepth);

void visualizeDataAssociationPlaneProjective(uchar3 * img,
                                             const int width,
                                             const int height,
                                             const SE3 & T_mc,
                                             const SE3 * T_fms,
                                             const int * sdfFrames,
                                             const Grid3D<float> * sdfs,
                                             const int nSdfs,
                                             const uchar3 * sdfColors,
                                             const float planeDepth,
                                             const float focalLength);

void visualizeDataAssociationPlaneProjective(uchar4 * img,
                                             const int width,
                                             const int height,
                                             const SE3 & T_mc,
                                             const SE3 * T_fms,
                                             const int * sdfFrames,
                                             const Grid3D<float> * sdfs,
                                             const int nSdfs,
                                             const uchar3 * sdfColors,
                                             const float planeDepth,
                                             const float focalLength);

}

#endif // SDF_VIZ
