#include "organized_point_cloud.h"

#include <iostream>
#include <stdio.h>
#include <helper_math.h>

namespace dart {

// -=-=-=-=-=-=-=-=-=- kernels -=-=-=-=-=-=-=-=-=-
template <typename DepthType>
__global__ void gpu_depthToVertices(const DepthType * depthIn,
                                    float4 * vertOut,
                                    const int width,
                                    const int height,
                                    const float2 pp,
                                    const float2 fl,
                                    const float2 range) {

    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const int index = u + v*width;

    if (u >= width || v >= height)
        return;

    float depth = depthIn[index];

    if (depth >= range.x && depth <= range.y) {

        vertOut[index] = make_float4( (u - pp.x)*(depth/fl.x),
                                      (v - pp.y)*(depth/fl.y),
                                      depth,
                                      1.0f);

    }
    else {
        vertOut[index].w = 0;
    }

}

template <typename DepthType>
__global__ void gpu_depthToVertices(const DepthType * depthIn,
                                    float4 * vertOut,
                                    const int width,
                                    const int height,
                                    const float2 pp,
                                    const float2 fl,
                                    const float2 range,
                                    const float scale) {

    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const int index = u + v*width;

    if (u >= width || v >= height)
        return;

    float depth = scale*depthIn[index];

    if (depth >= range.x && depth <= range.y) {

        vertOut[index] = make_float4( (u - pp.x)*(depth/fl.x),
                                      (v - pp.y)*(depth/fl.y),
                                      depth,
                                      1.0f);

    }
    else {
        vertOut[index].w = 0;
    }

}

template <typename DepthType>
__global__ void gpu_depthToVertices(const DepthType * depthIn,
                                    float4 * vertOut,
                                    const int width,
                                    const int height,
                                    const float4 * Kinv,
                                    const float2 range) {

    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const int index = u + v*width;

    if (u >= width || v >= height)
        return;

    float depth = depthIn[index];

    if (depth >= range.x && depth <= range.y) {

        const float4 p = make_float4( u, v, depth, 1);
        float4 vert = make_float4( dot(Kinv[0],p),
                                   dot(Kinv[1],p),
                                   dot(Kinv[2],p),
                                   dot(Kinv[3],p));
        vert /= vert.w;
        vert.w = 1;
        vert.z = -vert.z;
        vertOut[index] = vert;

    }
    else {
        vertOut[index].w = 0;
    }

}

template <typename DepthType>
__global__ void gpu_depthToVertices(const DepthType * depthIn,
                                    float4 * vertOut,
                                    const int width,
                                    const int height,
                                    const float4 * Kinv,
                                    const float2 range,
                                    const float scale) {

    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const int index = u + v*width;

    if (u >= width || v >= height)
        return;

    float depth = scale*depthIn[index];

    if (depth >= range.x && depth <= range.y) {

        const float4 p = make_float4( u, v, depth, 1);
        float4 vert = make_float4( dot(Kinv[0],p),
                                   dot(Kinv[1],p),
                                   dot(Kinv[2],p),
                                   dot(Kinv[3],p));
        vert /= vert.w;
        vert.w = 1;
        vert.z = -vert.z;
        vertOut[index] = vert;

    }
    else {
        vertOut[index].w = 0;
    }

}

template <typename DepthType, int iters>
__global__ void gpu_depthToVertices(const DepthType * depthIn,
                                    float4 * vertOut,
                                    const int width,
                                    const int height,
                                    const float * cameraParams,
                                    const float2 range) {

    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const int index = u + v*width;

    if (u >= width || v >= height)
        return;

    float depth = depthIn[index];

    if (depth >= range.x && depth <= range.y) {

        // http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

        const float &fx = cameraParams[0];
        const float &fy = cameraParams[1];
        const float &cx = cameraParams[2];
        const float &cy = cameraParams[3];
        const float &k1 = cameraParams[4];
        const float &k2 = cameraParams[5];
        const float &p1 = cameraParams[6];
        const float &p2 = cameraParams[7];
        const float &k3 = cameraParams[8];

        float xp, yp, xpp, ypp;
        xpp = xp = (u - cx) / fx;
        ypp = yp = (v - cy) / fy;

#pragma unroll
        for (int i=0; i<iters; ++i) {

            float r2 = xp*xp + yp*yp;
            float r4 = r2*r2;
            float r6 = r4*r2;
            float denom = 1 + k1*r2 + k2*r4 + k3*r6;
            float dxp = 2*p1*xp*yp + p2*(r2 + 2*xp*xp);
            float dyp = p1*(r2 + 2*yp*yp) + 2*p2*xp*yp;
            xp = (xpp - dxp)/denom;
            yp = (ypp - dyp)/denom;

        }

        vertOut[index] = make_float4(xp*depth,yp*depth,depth,1.0f);

    }
    else {
        vertOut[index].w = 0;
    }

}

template <typename DepthType, int iters>
__global__ void gpu_depthToVertices(const DepthType * depthIn,
                                    float4 * vertOut,
                                    const int width,
                                    const int height,
                                    const float * cameraParams,
                                    const float2 range,
                                    const float scale) {

    const int u = blockIdx.x*blockDim.x + threadIdx.x;
    const int v = blockIdx.y*blockDim.y + threadIdx.y;
    const int index = u + v*width;

    if (u >= width || v >= height)
        return;

    float depth = scale*depthIn[index];

    if (depth >= range.x && depth <= range.y) {

        // http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

        const float& fx = cameraParams[0];
        const float& fy = cameraParams[1];
        const float& cx = cameraParams[2];
        const float& cy = cameraParams[3];
        const float& k1 = cameraParams[4];
        const float& k2 = cameraParams[5];
        const float& p1 = cameraParams[6];
        const float& p2 = cameraParams[7];
        const float& k3 = cameraParams[8];

        float xp, yp, xpp, ypp;
        xpp = xp = (u - cx) / fx;
        ypp = yp = (v - cy) / fy;

#pragma unroll
        for (int i=0; i<iters; ++i) {

            float r2 = xp*xp + yp*yp;
            float r4 = r2*r2;
            float r6 = r4*r2;
            float denom = 1 + k1*r2 + k2*r4 + k3*r6;
            float dxp = 2*p1*xp*yp + p2*(r2 + 2*xp*xp);
            float dyp = p1*(r2 + 2*yp*yp) + 2*p2*xp*yp;
            xp = (xpp - dxp)/denom;
            yp = (ypp - dyp)/denom;

        }

        vertOut[index] = make_float4(xp*depth,yp*depth,depth,1.0f);

    }
    else {
        vertOut[index].w = 0;
    }

}

__global__ void gpu_verticesToNormals(const float4 * vertIn,
                                      float4 * normOut,
                                      const int width,
                                      const int height) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int index = x + y*width;

    const float4 & v = vertIn[index];

//    // don't process invalid vertices
    if ( v.w == 0) {
        normOut[index] = make_float4(0);
        return;
    }

    const float4 & vLeft = vertIn[ x == 0 ? index : index-1];
    const float4 & vRight = vertIn[ x == width-1 ? index : index+1];
    const float4 & vUp = vertIn[ y == 0 ? index : index-width];
    const float4 & vDown = vertIn[ y == height-1 ? index : index+width];

    const float3 vX = make_float3( (vRight.w == 0 ? v : vRight) - (vLeft.w == 0 ? v : vLeft) );
    const float3 vY = make_float3( (vDown.w == 0 ? v : vDown) - (vUp.w == 0 ? v : vUp) );
    const float3 n = cross(vY,vX);

    const float len2 = dot(n,n);

    if (len2 > 0) {
        const float invLen = 1.0f / (float)sqrtf(len2);
        normOut[index] = make_float4(n.x*invLen,n.y*invLen,n.z*invLen,1);
    }
    else {
        normOut[index] = make_float4(0);
    }

}

__global__ void gpu_eliminatePlane(float4 * verts, const float4 * norms, const int width, const int height, const float3 planeNormal, const float planeD, const float epsDist, const float epsNorm) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int index = x + y*width;

    // check vertex validity
    float4& v = verts[index];
    if ( v.w == 0) {
        return;
    }

    // check normal threshold
    const float4& n = norms[index];
    if (dot(make_float3(n),planeNormal) < epsNorm) {
        return;
    }

    // check distance threshold
    if (abs(dot(make_float3(v),planeNormal) - planeD) < epsDist ) {
        v.w = -1;
    }

}

__global__ void gpu_cropBox(float4 * verts, const int width, const int height, const float3 boxMin, const float3 boxMax) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int index = x + y*width;

    // check vertex validity
    float4& v = verts[index];
    if ( v.w == 0) {
        return;
    }

    if (v.x < boxMin.x || v.x > boxMax.x ||
            v.y < boxMin.y || v.y > boxMax.y ||
            v.z < boxMin.z || v.z > boxMax.z) {
        v.w = -1;
    }

}

__global__ void gpu_maskPointCloud(float4* verts, const int width, const int height, const int* mask) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int index = x + y*width;

    int m = mask[index];
    if (m == 0) {
        verts[index].w = -1;
    }

}

// -=-=-=-=-=-=-=-=-=- interface -=-=-=-=-=-=-=-=-=-
template <typename DepthType>
void depthToVertices(const DepthType * depthIn, float4 * vertOut, const int width, const int height, const float2 pp, const float2 fl, const float2 range) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil( height / (float)block.y ));

    gpu_depthToVertices<<<grid,block>>>(depthIn, vertOut, width, height, pp, fl, range);

}

template <typename DepthType>
void depthToVertices(const DepthType * depthIn, float4 * vertOut, const int width, const int height, const float2 pp, const float2 fl, const float2 range, const float scale) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil( height / (float)block.y ));

    gpu_depthToVertices<<<grid,block>>>(depthIn, vertOut, width, height, pp, fl, range, scale);

}

template <typename DepthType>
void depthToVertices(const DepthType * depthIn, float4 * vertOut, const int width, const int height, const float * calibrationParams, const float2 range) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil( height / (float)block.y ));

    gpu_depthToVertices<DepthType,5><<<grid,block>>>(depthIn, vertOut, width, height, calibrationParams, range);

}

template <typename DepthType>
void depthToVertices(const DepthType * depthIn, float4 * vertOut, const int width, const int height, const float * calibrationParams, const float2 range, const float scale) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil( height / (float)block.y ));

    gpu_depthToVertices<DepthType,5><<<grid,block>>>(depthIn, vertOut, width, height, calibrationParams, range, scale);

}

void verticesToNormals(const float4 * vertIn, float4 * normOut, const int width, const int height) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil( height / (float)block.y ));

    gpu_verticesToNormals<<<grid,block>>>(vertIn,normOut,width,height);
}

void eliminatePlane(float4 * verts, const float4 * norms, const int width, const int height, const float3 planeNormal, const float planeD, const float epsDist, const float epsNorm) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil( height / (float)block.y ));

    gpu_eliminatePlane<<<grid,block>>>(verts,norms,width,height,planeNormal,planeD,epsDist,epsNorm);

}

void cropBox(float4 * verts, const int width, const int height, const float3 & boxMin, const float3 & boxMax) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil( height / (float)block.y ));

    gpu_cropBox<<<grid,block>>>(verts,width,height,boxMin,boxMax);

}

void maskPointCloud(float4 * verts, const int width, const int height, const int * mask) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil( height / (float)block.y ));

    gpu_maskPointCloud<<<grid,block>>>(verts,width,height,mask);

}

#define COMPILE_DEPTH_TYPE(type) \
    template void depthToVertices<type>(const type * depthIn, float4 * vertOut, const int width, const int height, const float2 pp, const float2 fl, const float2 range); \
    template void depthToVertices<type>(const type * depthIn, float4 * vertOut, const int width, const int height, const float2 pp, const float2 fl, const float2 range, const float scale); \
    template void depthToVertices<type>(const type * depthIn, float4 * vertOut, const int width, const int height, const float * calibrationparams, const float2 range); \
    template void depthToVertices<type>(const type * depthIn, float4 * vertOut, const int width, const int height, const float * calibrationparams, const float2 range, const float scale);


COMPILE_DEPTH_TYPE(float)
COMPILE_DEPTH_TYPE(ushort)

}
