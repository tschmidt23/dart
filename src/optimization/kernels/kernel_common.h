#ifndef KERNEL_COMMON_H
#define KERNEL_COMMON_H

#include "geometry/SE3.h"
#include "util/dart_types.h"

#include <vector_functions.h>
#include <helper_math.h>

namespace dart {

__device__ inline void getErrorJacobianOfModelPoint(float * J, const float4 & point_m, const int frame, const float3 & errorGrad3D_m,
                                             const int dims, const int * dependencies, const JointType * jointTypes,
                                             const float3 * jointAxes, const SE3 * T_fms, const SE3 * T_mfs) {

    J[0] = dot(errorGrad3D_m,make_float3(-1, 0, 0));
    J[1] = dot(errorGrad3D_m,make_float3( 0,-1, 0));
    J[2] = dot(errorGrad3D_m,make_float3( 0, 0,-1));

    // rotation
    J[3] = dot(errorGrad3D_m,make_float3(         0, point_m.z,-point_m.y));
    J[4] = dot(errorGrad3D_m,make_float3(-point_m.z,         0, point_m.x));
    J[5] = dot(errorGrad3D_m,make_float3( point_m.y,-point_m.x,         0));

#pragma unroll
    for (int i=0; i<(dims-6); i++) {

        if (dependencies[frame*(dims-6) + i] == 0) {
            J[i+6] = 0;
            continue;
        }

        const int jointFrame = i+1;
        if (jointTypes[i] == RotationalJoint) {
            const float4 axisRelativePoint = T_fms[jointFrame]*point_m;
            const float3 dx_a = cross(jointAxes[i],make_float3(axisRelativePoint));
            const float3 dx = SE3Rotate(T_mfs[jointFrame],dx_a);
            J[i+6] = dot(errorGrad3D_m,dx);
        } else if (jointTypes[i] == PrismaticJoint) {
            const float3 axis_m = SE3Rotate(T_mfs[jointFrame],jointAxes[i]);
            J[i+6] = dot(errorGrad3D_m,axis_m);
        }

    }
}

__device__ inline void getErrorJacobianOfModelPointArticulationOnly(float * J, const float4 & point_m, const int frame, const float3 & errorGrad3D_m,
                                                                    const int dims, const int * dependencies, const JointType * jointTypes,
                                                                    const float3 * jointAxes, const SE3 * T_fms, const SE3 * T_mfs) {

#pragma unroll
    for (int i=0; i<(dims-6); i++) {

        if (dependencies[frame*(dims-6) + i] == 0) {
            J[i] = 0;
            continue;
        }

        const int jointFrame = i+1;
        if (jointTypes[i] == RotationalJoint) {
            float4 axisRelativePoint = T_fms[jointFrame]*point_m;
            float3 dx_a = cross(jointAxes[i],make_float3(axisRelativePoint));
            float3 dx = SE3Rotate(T_mfs[jointFrame],dx_a);
            J[i] = dot(errorGrad3D_m,dx);
        } else if (jointTypes[i] == PrismaticJoint) {
            const float3 axis_m = SE3Rotate(T_mfs[jointFrame],jointAxes[i]);
            J[i] = dot(errorGrad3D_m,axis_m);
        }

    }
}

__device__ inline void doPoseGradientReduction(float * J, const float * de_dtheta, const float * dtheta_dalpha, const int fullDims, const int redDims) {

    // copy over 6DoF gradients [TODO: this is wasteful]
    for (int r=0; r<6; ++r) {
        J[r] = de_dtheta[r];
    }

    // articulation reduction
    for (int r=6; r<redDims; ++r) {
        J[r] = 0;
        for (int f=6; f<fullDims; ++f) {
            float mult = dtheta_dalpha[r-6 + (f-6)*(redDims-6)];
            if (mult == 0.f) { continue; }
            J[r] += de_dtheta[f]*mult;
        }
    }

}

__device__ inline void doPoseGradientReductionArticulationOnly(float * J, const float * de_dtheta, const float * dtheta_dalpha, const int fullDims, const int redDims) {

    // articulation reduction
    for (int r=0; r<redDims-6; ++r) {
        J[r] = 0;
        for (int f=0; f<fullDims-6; ++f) {
            float mult = dtheta_dalpha[r + (f)*(redDims-6)];
            if (mult == 0.f) { continue; }
            J[r] += de_dtheta[f]*mult;
        }
    }

}

__device__ inline void doParamMapping(float * J, const float * de_dtheta, const int * dMapping, const int fullDims, const int redDims) {

    // copy over 6DoF gradients [TODO: this is wasteful]
    for (int r=0; r<6; ++r) {
        J[r] = de_dtheta[r];
    }

    // initialize to zero
    for (int r=6; r<redDims; ++r) {
        J[r] = 0;
    }

    // do mapping
    for (int f=6; f<fullDims; ++f) {
        const int r = dMapping[f-6] + 6;
        J[r] += de_dtheta[f];
    }

}

__device__ inline void doParamMappingArticulationOnly(float * J, const float * de_dtheta, const int * dMapping, const int fullDims, const int redDims) {

    // initialize to zero
    for (int r=0; r<redDims-6; ++r) {
        J[r] = 0;
    }

    // do mapping
    for (int f=0; f<fullDims-6; ++f) {
        const int r = dMapping[f];
        J[r] += de_dtheta[f];
    }

}

__device__ inline void computeSquaredLossResult(const int dims, const float residual, const float * J,
                                                float * e, float * JTr, float * JTJ) {
    for (int i=0; i<dims; i++) {
        if( J[i]==0.0f)  continue;
        float v = J[i]*residual;
        atomicAdd(&JTr[i],v);
        for (int j=0; j<=i; j++) {
            float v2 = J[i]*J[j];
            atomicAdd(&JTJ[((i*(i+1))>>1) + j],v2);
        }
    }
    atomicAdd(e,0.5*residual*residual);
}

}

#endif // KERNEL_COMMON_H
