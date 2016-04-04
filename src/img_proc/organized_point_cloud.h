#ifndef ORGANIZED_POINT_CLOUD_H
#define ORGANIZED_POINT_CLOUD_H

#include <limits>
#include <vector_types.h>
#include <vector_functions.h>

namespace dart {

template <typename DepthType>
void depthToVertices(const DepthType * depthIn, float4 * vertOut, const int width, const int height, const float2 pp, const float2 fl, const float2 range = make_float2(0,std::numeric_limits<float>::infinity()));

template <typename DepthType>
void depthToVertices(const DepthType * depthIn, float4 * vertOut, const int width, const int height, const float2 pp, const float2 fl, const float2 range, const float scale);

template <typename DepthType>
void depthToVertices(const DepthType * depthIn, float4 * vertOut, const int width, const int height, const float * calibrationParams, const float2 range);

template <typename DepthType>
void depthToVertices(const DepthType * depthIn, float4 * vertOut, const int width, const int height, const float * calibrationParams, const float2 range, const float scale);

void verticesToNormals(const float4 * vertIn, float4 * normOut, const int width, const int height);

void eliminatePlane(float4 * verts, const float4 * norms, const int width, const int height, const float3 planeNormal, const float planeD, const float epsDist = 0.01, const float epsNorm = 0.1);

void cropBox(float4 * verts, const int width, const int height, const float3 & boxMin, const float3 & boxMax);

void maskPointCloud(float4 * verts, const int width, const int height, const int * mask);

}

#endif // ORGANIZED_POINT_CLOUD_H
