#ifndef COLOR_RAMPS_H
#define COLOR_RAMPS_H

namespace dart {

void colorRampHeatMap(uchar3 * colored,
                      const float * vals,
                      const int width,
                      const int height,
                      const float minVal = 0.0f,
                      const float maxVal = 1.0f);

void colorRampHeatMap(uchar4 * colored,
                      const float * vals,
                      const int width,
                      const int height,
                      const float minVal = 0.0f,
                      const float maxVal = 1.0f);

void colorRampHeatMapUnsat(uchar3 * colored,
                           const float * vals,
                           const int width,
                           const int height,
                           const float minVal,
                           const float maxVal);

void colorRampHeatMapUnsat(uchar4 * colored,
                           const float * vals,
                           const int width,
                           const int height,
                           const float minVal,
                           const float maxVal);

void colorRampTopographic(uchar4 * colored,
                          const float * vals,
                          const int width,
                          const int height,
                          const float lineThickness,
                          const float lineSpacing,
                          const bool showZeroLevel = true);

void colorRamp2DGradient(uchar4 * color,
                         const float2 * grad,
                         const int width,
                         const int height,
                         const bool normalize = true);

void colorRamp3DGradient(uchar4 * color,
                         const float3 * grad,
                         const int width,
                         const int height,
                         const bool normalize = true);

}

#endif // COLOR_RAMPS_H
