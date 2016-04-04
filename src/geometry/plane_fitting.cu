#include "plane_fitting.h"

#include <iostream>
#include <stdio.h>
#include <vector_functions.h>
#include <helper_math.h>

namespace dart {

// -=-=-=-=- kernel -=-=-=-=-
template <bool dbgAssoc>
__global__ void gpu_fitPlane(const float3 planeNormal,
                             const float planeIntercept,
                             const float4 * obsVertMap,
                             const float4 * obsNormMap,
                             const int width,
                             const int height,
                             const float distanceThreshold,
                             const float normalThreshold,
                             float * result,
                             int * dbgAssociated) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    if (dbgAssoc) { dbgAssociated[index] = 0; }

    if (obsVertMap[index].w == 0 || obsNormMap[index].w == 0) {
        return;
    }

    float3 v = make_float3(obsVertMap[index]);

    float dist = dot(planeNormal,v) - planeIntercept;
    if (fabs(dist) > distanceThreshold) {
        return;
    }

    float3 n = make_float3(obsNormMap[index]);
    if (dot(planeNormal,n) < normalThreshold) {
        return;
    }

    if (dbgAssoc) { dbgAssociated[index] = 1; }

    float J[4];
    J[0] = v.x;
    J[1] = v.y;
    J[2] = v.z;
    J[3] = -1;

    float * eJ = result;
    float * JTJ = &result[4];
    float * e = &result[4 + 16];

    for (int i=0; i<4; ++i) {
        float ejVal = dist*J[i];
        atomicAdd(&eJ[i],ejVal);
        for (int j=0; j<4; ++j) {
            float jtjVal = J[i]*J[j];
            atomicAdd(&JTJ[i*4 + j],jtjVal);
        }
    }

    atomicAdd(e,dist*dist);
}

__global__ void gpu_subtractPlane(float4 * obsVertMap,
                                  float4 * obsNormMap,
                                  const int width,
                                  const int height,
                                  const float3 planeNormal,
                                  const float planeIntercept,
                                  const float distanceThreshold,
                                  const float normalThreshold) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;

    float3 v = make_float3(obsVertMap[index]);

    float dist = dot(planeNormal,v) - planeIntercept;
    //if (fabs(dist) > distanceThreshold) {
    if (dist > distanceThreshold) {
        return;
    }

    if (obsNormMap[index].w == 1.0) {
        float3 n = make_float3(obsNormMap[index]);
        if (dot(planeNormal,n) < normalThreshold) {
            return;
        }
    }

    obsVertMap[index].w = -1.0f;
    obsNormMap[index].w = -1.0f;

}

// -=-=-=-=- interface -=-=-=-=-
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
                  int * dbgAssociated) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    if (dbgAssociated == 0) {
        gpu_fitPlane<false><<<grid,block>>>(planeNormal,
                                            planeIntercept,
                                            dObsVertMap,
                                            dObsNormMap,
                                            width,
                                            height,
                                            distanceThreshold,
                                            normalThreshold,
                                            dResult,
                                            dbgAssociated);
    } else {
        gpu_fitPlane<true><<<grid,block>>>(planeNormal,
                                           planeIntercept,
                                           dObsVertMap,
                                           dObsNormMap,
                                           width,
                                           height,
                                           distanceThreshold,
                                           normalThreshold,
                                           dResult,
                                           dbgAssociated);
    }


}

void subtractPlane_(float4 * dObsVertMap,
                    float4 * dObsNormMap,
                    const int width,
                    const int height,
                    const float3 planeNormal,
                    const float planeIntercept,
                    const float distanceThreshold,
                    const float normalThreshold) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_subtractPlane<<<grid,block>>>(dObsVertMap,
                                      dObsNormMap,
                                      width,
                                      height,
                                      planeNormal,
                                      planeIntercept,
                                      distanceThreshold,
                                      normalThreshold);

}

}
