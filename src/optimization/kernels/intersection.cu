#include "intersection.h"

#include "optimization/optimization.h"
#include "util/mirrored_memory.h"
#include "kernel_common.h"

namespace dart {

// -=-=-=-=-=-=-=-=-=- kernels -=-=-=-=-=-=-=-=-=-

__global__ void gpu_selfIntersectionCount(const float4 * testSites,
                                          const int nSites,
                                          const SE3 * T_mfs,
                                          const SE3 * T_fms,
                                          const int * sdfFrames,
                                          const Grid3D<float> * sdfs,
                                          const int nSdfs,
                                          const int * potentialIntersection,
                                          int * nCollisions) {

    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    // overflow
    if (index >= nSites) {
        return;
    }

    float4 v_src_f = testSites[index];
    const int srcGrid = round(v_src_f.w);
    const int srcFrame = sdfFrames[srcGrid];

    v_src_f.w = 1;
    const float4 v_m = T_mfs[srcFrame]*v_src_f;

    for (int dstGrid=0; dstGrid<nSdfs; ++dstGrid) {

        if (potentialIntersection[dstGrid + srcGrid*nSdfs]) {

            const int dstFrame = sdfFrames[dstGrid];
            const float4 v_dst_f = T_fms[dstFrame]*v_m;

            const Grid3D<float> &dstSdf = sdfs[dstGrid];
            const float3 v_dst_g = dstSdf.getGridCoords(make_float3(v_dst_f));

            //printf("%f %f %f   ",v_dst_g.x,v_dst_g.y,v_dst_g.z);

            if (dstSdf.isInBoundsGradientInterp(v_dst_g)) {

                const float d = dstSdf.getValueInterpolated(v_dst_g)*dstSdf.resolution;
                if (d < 0) {

                    // collision detected
                    atomicAdd(nCollisions,1);

                    return;
                }
            }

        }

    }


}

template <bool dbgErr>
__global__ void gpu_normEqnsSelfIntersection(const float4 * testSites,
                                             const int nSites,
                                             const int dims,
                                             const SE3 * T_mfs,
                                             const SE3 * T_fms,
                                             const int * sdfFrames,
                                             const Grid3D<float> * sdfs,
                                             const int nSdfs,
                                             const int * dependencies,
                                             const JointType * jointTypes,
                                             const float3 * jointAxes,
                                             const int * potentialIntersection,
                                             float * result,
                                             float * debugError) {

    extern __shared__ float s[];

    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    // overflow
    if (index >= nSites) {
        return;
    }

//    if (dbgErr) { debugError[index] = NAN; }

    float4 v_src_f = testSites[index];
    const int srcGrid = round(v_src_f.w);
    const int srcFrame = sdfFrames[srcGrid];

    v_src_f.w = 1;
    const float4 v_m = T_mfs[srcFrame]*v_src_f;

    for (int dstGrid=0; dstGrid<nSdfs; ++dstGrid) {

        if (potentialIntersection[dstGrid + srcGrid*nSdfs]) {

            const int dstFrame = sdfFrames[dstGrid];
            const float4 v_dst_f = T_fms[dstFrame]*v_m;

            const Grid3D<float> & dstSdf = sdfs[dstGrid];
            const float3 v_dst_g = dstSdf.getGridCoords(make_float3(v_dst_f));

            //printf("%f %f %f   ",v_dst_g.x,v_dst_g.y,v_dst_g.z);

            if (dstSdf.isInBoundsGradientInterp(v_dst_g)) {

                const float residual = dstSdf.getValueInterpolated(v_dst_g)*dstSdf.resolution;
                if (residual < 0) {

                    // collision detected
                    float * J = &s[threadIdx.x*(dims-6)]; // dims-6 because self-intersection doesn't depend on global transform

                    const float3 dstSdfGrad_dst_f = dstSdf.getGradientInterpolated(v_dst_g);
                    const float3 dstSdfGrad_m = make_float3(SE3Rotate(T_mfs[dstFrame],make_float4(dstSdfGrad_dst_f,0)));

                    getErrorJacobianOfModelPointArticulationOnly(J,v_m,srcFrame,dstSdfGrad_m,dims,dependencies,jointTypes,jointAxes,T_fms,T_mfs);

                    float * JTr = result;
                    float * JTJ = &result[dims-6];
                    float * e = &result[dims-6 + JTJSize(dims-6)];

                    computeSquaredLossResult(dims-6,residual,J,e,JTr, JTJ);

                    if (dbgErr) {
                        debugError[index] = residual*residual;
                    }

                    return;

                }
            }

        }

    }


}

__global__ void gpu_normEqnsSelfIntersectionReduced(const float4 * testSites,
                                                    const int nSites,
                                                    const int fullDims,
                                                    const int redDims,
                                                    const SE3 * T_mfs,
                                                    const SE3 * T_fms,
                                                    const int * sdfFrames,
                                                    const Grid3D<float> * sdfs,
                                                    const int nSdfs,
                                                    const int * dependencies,
                                                    const JointType * jointTypes,
                                                    const float3 * jointAxes,
                                                    const float * dtheta_dalpha,
                                                    const int * potentialIntersection,
                                                    float * result) {

    extern __shared__ float s[];

    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    // overflow
    if (index >= nSites) {
        return;
    }

    float4 v_src_f = testSites[index];
    const int srcGrid = round(v_src_f.w);
    const int srcFrame = sdfFrames[srcGrid];

    v_src_f.w = 1;
    const float4 v_m = T_mfs[srcFrame]*v_src_f;

    for (int dstGrid=0; dstGrid<nSdfs; ++dstGrid) {

        if (potentialIntersection[dstGrid + srcGrid*nSdfs]) {

            const int dstFrame = sdfFrames[dstGrid];
            const float4 v_dst_f = T_fms[dstFrame]*v_m;

            const Grid3D<float> &dstSdf = sdfs[dstGrid];
            const float3 v_dst_g = dstSdf.getGridCoords(make_float3(v_dst_f));

            if (dstSdf.isInBoundsGradientInterp(v_dst_g)) {

                const float residual = dstSdf.getValueInterpolated(v_dst_g)*dstSdf.resolution;
                if (residual < 0) {

                    //printf("*");

                    // collision detected
                    float * de_dtheta = &s[threadIdx.x*(fullDims-6+redDims-6)]; // redDims-6 because self-intersection doesn't depend on global transform
                    float * J = &s[threadIdx.x*(fullDims-6+redDims-6) + fullDims-6];

                    const float3 dstSdfGrad_dst_f = dstSdf.getGradientInterpolated(v_dst_g);
                    const float3 dstSdfGrad_m = make_float3(SE3Rotate(T_mfs[dstFrame],make_float4(dstSdfGrad_dst_f,0)));

                    getErrorJacobianOfModelPointArticulationOnly(de_dtheta,v_m,srcFrame,dstSdfGrad_m,fullDims,dependencies,jointTypes,jointAxes,T_fms,T_mfs);

                    // reduction
                    doPoseGradientReductionArticulationOnly(J,de_dtheta,dtheta_dalpha,fullDims,redDims);

                    float * JTr = result;
                    float * JTJ = &result[redDims-6];
                    float * e = &result[redDims-6 + JTJSize(redDims-6)];

                    computeSquaredLossResult(redDims-6,residual,J,e,JTr,JTJ);

                    return;

                }
            }

        }

    }

}

__global__ void gpu_normEqnsSelfIntersectionParamMap(const float4 * testSites,
                                                    const int nSites,
                                                    const int fullDims,
                                                    const int redDims,
                                                    const SE3 * T_mfs,
                                                    const SE3 * T_fms,
                                                    const int * sdfFrames,
                                                    const Grid3D<float> * sdfs,
                                                    const int nSdfs,
                                                    const int * dependencies,
                                                    const JointType * jointTypes,
                                                    const float3 * jointAxes,
                                                    const int * dMapping,
                                                    const int * potentialIntersection,
                                                    float * result) {

    extern __shared__ float s[];

    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    // overflow
    if (index >= nSites) {
        return;
    }

    float4 v_src_f = testSites[index];
    const int srcGrid = round(v_src_f.w);
    const int srcFrame = sdfFrames[srcGrid];

    v_src_f.w = 1;
    const float4 v_m = T_mfs[srcFrame]*v_src_f;

    for (int dstGrid=0; dstGrid<nSdfs; ++dstGrid) {

        if (potentialIntersection[dstGrid + srcGrid*nSdfs]) {

            const int dstFrame = sdfFrames[dstGrid];
            const float4 v_dst_f = T_fms[dstFrame]*v_m;

            const Grid3D<float> &dstSdf = sdfs[dstGrid];
            const float3 v_dst_g = dstSdf.getGridCoords(make_float3(v_dst_f));

            if (dstSdf.isInBoundsGradientInterp(v_dst_g)) {

                const float residual = dstSdf.getValueInterpolated(v_dst_g)*dstSdf.resolution;
                if (residual < 0) {

                    //printf("*");

                    // collision detected
                    float * de_dtheta = &s[threadIdx.x*(fullDims-6+redDims-6)]; // redDims-6 because self-intersection doesn't depend on global transform
                    float * J = &s[threadIdx.x*(fullDims-6+redDims-6) + fullDims-6];

                    const float3 dstSdfGrad_dst_f = dstSdf.getGradientInterpolated(v_dst_g);
                    const float3 dstSdfGrad_m = make_float3(SE3Rotate(T_mfs[dstFrame],make_float4(dstSdfGrad_dst_f,0)));

                    getErrorJacobianOfModelPointArticulationOnly(de_dtheta,v_m,srcFrame,dstSdfGrad_m,fullDims,dependencies,jointTypes,jointAxes,T_fms,T_mfs);

                    doParamMappingArticulationOnly(J,de_dtheta,dMapping,fullDims,redDims);

                    float * JTr = result;
                    float * JTJ = &result[redDims-6];
                    float * e = &result[redDims-6 + JTJSize(redDims-6)];

                    computeSquaredLossResult(redDims-6,residual,J,e,JTr,JTJ);

                    return;

                }
            }

        }

    }

}


__global__ void gpu_initDebugIntersectionError(float * debugError,
                                               const int nSites) {
    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    // overflow
    if (index >= nSites) {
        return;
    }

    debugError[index] = NAN;
}

__global__ void gpu_intersectionCount(const float4 * testSites,
                                      const int nSites,
                                      const SE3 T_ds,
                                      const SE3 * T_mfs_src,
                                      const int * sdfFrames_src,
                                      const SE3 * T_fms_dst,
                                      const int * sdfFrames_dst,
                                      const Grid3D<float> * sdfs_dst,
                                      const int nSdfs_dst,
                                      int * nCollisions) {

    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    // overflow
    if (index >= nSites) {
        return;
    }

    float4 v_src_f = testSites[index];
    const int srcGrid = round(v_src_f.w);
    const int srcFrame = sdfFrames_src[srcGrid];

    v_src_f.w = 1;
    const float4 v_src_m = T_mfs_src[srcFrame]*v_src_f;
    const float4 v_dst_m = T_ds*v_src_m;

    for (int dstGrid=0; dstGrid<nSdfs_dst; ++dstGrid) {

        const int dstFrame = sdfFrames_dst[dstGrid];
        const float4 v_dst_f = T_fms_dst[dstFrame]*v_dst_m;

        const Grid3D<float> & dstSdf = sdfs_dst[dstGrid];
        const float3 v_dst_g = dstSdf.getGridCoords(make_float3(v_dst_f));

        //printf("%f %f %f   ",v_dst_g.x,v_dst_g.y,v_dst_g.z);

        if (dstSdf.isInBoundsGradientInterp(v_dst_g)) {

            const float d = dstSdf.getValueInterpolated(v_dst_g)*dstSdf.resolution;
            if (d < 0) {

                // collision detected
                atomicAdd(nCollisions,1);

                return;
            }
        }

    }

}

template <bool dbgErr>
__global__ void gpu_normEquationsIntersection(const float4 * testSites,
                                              const int nSites,
                                              const int dims,
                                              const SE3 T_ds,
                                              const SE3 T_sd,
                                              const SE3 * T_mfs_src,
                                              const SE3 * T_fms_src,
                                              const int * sdfFrames_src,
                                              const SE3 * T_mfs_dst,
                                              const SE3 * T_fms_dst,
                                              const int * sdfFrames_dst,
                                              const Grid3D<float> * sdfs_dst,
                                              const int nSdfs_dst,
                                              const int * dependencies_src,
                                              const JointType * jointTypes_src,
                                              const float3 * jointAxes_src,
                                              float * result,
                                              float * debugError) {

    extern __shared__ float s[];

    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int tid = threadIdx.x + threadIdx.y*blockDim.x;

    // overflow
    if (index >= nSites) {
        return;
    }

    float4 v_src_f = testSites[index];
    const int srcGrid = round(v_src_f.w);
    const int srcFrame = sdfFrames_src[srcGrid];

    v_src_f.w = 1;
    const float4 v_src_m = T_mfs_src[srcFrame]*v_src_f;
    const float4 v_dst_m = T_ds*v_src_m;

    for (int dstGrid=0; dstGrid<nSdfs_dst; ++dstGrid) {

        const int dstFrame = sdfFrames_dst[dstGrid];
        const float4 v_dst_f = T_fms_dst[dstFrame]*v_dst_m;

        const Grid3D<float> & dstSdf = sdfs_dst[dstGrid];
        const float3 v_dst_g = dstSdf.getGridCoords(make_float3(v_dst_f));

        if (dstSdf.isInBoundsGradientInterp(v_dst_g)) {

            const float residual = dstSdf.getValueInterpolated(v_dst_g)*dstSdf.resolution;
            if (residual < 0) {

                // collision detected
                float * J = &s[tid*dims];

                const float3 dstSdfGrad_dst_f = dstSdf.getGradientInterpolated(v_dst_g);
                const float3 dstSdfGrad_dst_m = make_float3(SE3Rotate(T_mfs_dst[dstFrame],make_float4(dstSdfGrad_dst_f,0)));
                const float3 dstSdfGrad_src_m = SE3Rotate(T_sd,dstSdfGrad_dst_m);

                getErrorJacobianOfModelPoint(J,v_src_m,srcFrame,dstSdfGrad_src_m,dims,dependencies_src,jointTypes_src,jointAxes_src,T_fms_src,T_mfs_src);

                float * JTr = result;
                float * JTJ = &result[dims];
                float * e = &result[dims + JTJSize(dims)];

                computeSquaredLossResult(dims,residual,J,e,JTr,JTJ);

                if (dbgErr) {
                    debugError[index] += (residual*residual);
                }

                return;
            }
        }

    }

}

__global__ void gpu_normEqnsIntersectionReduced(const float4 * testSites,
                                                const int nSites,
                                                const int fullDims,
                                                const int redDims,
                                                const SE3 T_ds,
                                                const SE3 T_sd,
                                                const SE3 * T_mfs_src,
                                                const SE3 * T_fms_src,
                                                const int * sdfFrames_src,
                                                const SE3 * T_mfs_dst,
                                                const SE3 * T_fms_dst,
                                                const int * sdfFrames_dst,
                                                const Grid3D<float> * sdfs_dst,
                                                const int nSdfs_dst,
                                                const int * dependencies_src,
                                                const JointType * jointTypes_src,
                                                const float3 * jointAxes_src,
                                                const float * dtheta_dalpha_src,
                                                float * result) {

    extern __shared__ float s[];

    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int tid = threadIdx.x + threadIdx.y*blockDim.x;

    // overflow
    if (index >= nSites) {
        return;
    }

    float4 v_src_f = testSites[index];
    const int srcGrid = round(v_src_f.w);
    const int srcFrame = sdfFrames_src[srcGrid];

    v_src_f.w = 1;
    const float4 v_src_m = T_mfs_src[srcFrame]*v_src_f;
    const float4 v_dst_m = T_ds*v_src_m;

    for (int dstGrid=0; dstGrid<nSdfs_dst; ++dstGrid) {

        const int dstFrame = sdfFrames_dst[dstGrid];
        const float4 v_dst_f = T_fms_dst[dstFrame]*v_dst_m;

        const Grid3D<float> & dstSdf = sdfs_dst[dstGrid];
        const float3 v_dst_g = dstSdf.getGridCoords(make_float3(v_dst_f));

        if (dstSdf.isInBoundsGradientInterp(v_dst_g)) {

            const float residual = dstSdf.getValueInterpolated(v_dst_g)*dstSdf.resolution;
            if (residual < 0) {

                // collision detected
                float * de_dtheta = &s[tid*(fullDims+redDims)];
                float * J = &s[tid*(fullDims+redDims)+fullDims];

                const float3 dstSdfGrad_dst_f = dstSdf.getGradientInterpolated(v_dst_g);
                const float3 dstSdfGrad_dst_m = make_float3(SE3Rotate(T_mfs_dst[dstFrame],make_float4(dstSdfGrad_dst_f,0)));
                const float3 dstSdfGrad_src_m = SE3Rotate(T_sd,dstSdfGrad_dst_m);

                getErrorJacobianOfModelPoint(de_dtheta,v_src_m,srcFrame,dstSdfGrad_src_m,fullDims,dependencies_src,jointTypes_src,jointAxes_src,T_fms_src,T_mfs_src);

                doPoseGradientReduction(J,de_dtheta,dtheta_dalpha_src,fullDims,redDims);

                float * JTr = result;
                float * JTJ = &result[redDims];
                float * e = &result[redDims + JTJSize(redDims)];

                computeSquaredLossResult(redDims,residual,J,e,JTr,JTJ);

                return;
            }
        }

    }

}

__global__ void gpu_normEqnsIntersectionParamMap(const float4 * testSites,
                                                const int nSites,
                                                const int fullDims,
                                                const int redDims,
                                                const SE3 T_ds,
                                                const SE3 T_sd,
                                                const SE3 * T_mfs_src,
                                                const SE3 * T_fms_src,
                                                const int * sdfFrames_src,
                                                const SE3 * T_mfs_dst,
                                                const SE3 * T_fms_dst,
                                                const int * sdfFrames_dst,
                                                const Grid3D<float> * sdfs_dst,
                                                const int nSdfs_dst,
                                                const int * dependencies_src,
                                                const JointType * jointTypes_src,
                                                const float3 * jointAxes_src,
                                                const int * dMapping_src,
                                                float * result) {

    extern __shared__ float s[];

    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int tid = threadIdx.x + threadIdx.y*blockDim.x;

    // overflow
    if (index >= nSites) {
        return;
    }

    float4 v_src_f = testSites[index];
    const int srcGrid = round(v_src_f.w);
    const int srcFrame = sdfFrames_src[srcGrid];

    v_src_f.w = 1;
    const float4 v_src_m = T_mfs_src[srcFrame]*v_src_f;
    const float4 v_dst_m = T_ds*v_src_m;

    for (int dstGrid=0; dstGrid<nSdfs_dst; ++dstGrid) {

        const int dstFrame = sdfFrames_dst[dstGrid];
        const float4 v_dst_f = T_fms_dst[dstFrame]*v_dst_m;

        const Grid3D<float> & dstSdf = sdfs_dst[dstGrid];
        const float3 v_dst_g = dstSdf.getGridCoords(make_float3(v_dst_f));

        if (dstSdf.isInBoundsGradientInterp(v_dst_g)) {

            const float residual = dstSdf.getValueInterpolated(v_dst_g)*dstSdf.resolution;
            if (residual < 0) {

                // collision detected
                float * de_dtheta = &s[tid*(fullDims+redDims)];
                float * J = &s[tid*(fullDims+redDims)+fullDims];

                const float3 dstSdfGrad_dst_f = dstSdf.getGradientInterpolated(v_dst_g);
                const float3 dstSdfGrad_dst_m = make_float3(SE3Rotate(T_mfs_dst[dstFrame],make_float4(dstSdfGrad_dst_f,0)));
                const float3 dstSdfGrad_src_m = SE3Rotate(T_sd,dstSdfGrad_dst_m);

                getErrorJacobianOfModelPoint(de_dtheta,v_src_m,srcFrame,dstSdfGrad_src_m,fullDims,dependencies_src,jointTypes_src,jointAxes_src,T_fms_src,T_mfs_src);

                doParamMapping(J,de_dtheta,dMapping_src,fullDims,redDims);

                float * JTr = result;
                float * JTJ = &result[redDims];
                float * e = &result[redDims + JTJSize(redDims)];

                computeSquaredLossResult(redDims,residual,J,e,JTr,JTJ);

                return;
            }
        }

    }

}


__global__ void gpu_intersectionCheckRigidObjInHand(const float4 * testSites,
                                                    const int nSites,
                                                    const SE3 T_ho,
                                                    const SE3 T_oh,
                                                    const SE3 * T_mfs_h,
                                                    const SE3 * T_fms_h,
                                                    const int * sdfFrames_h,
                                                    const Grid3D<float> * sdfs_h,
                                                    const int nSdfs_h,
                                                    float * result) {

    extern __shared__ float s[];

    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    // overflow
    if (index >= nSites) {
        return;
    }

    float4 v_o = testSites[index];

    v_o.w = 1;
    const float4 v_h = T_ho*v_o;

    for (int hGrid=0; hGrid<nSdfs_h; ++hGrid) {

        const int hFrame = sdfFrames_h[hGrid];
        const float4 v_f = T_fms_h[hFrame]*v_h;

        const Grid3D<float> &hSdf = sdfs_h[hGrid];
        const float3 v_g = hSdf.getGridCoords(make_float3(v_f));

        if (hSdf.isInBoundsGradientInterp(v_g)) {

            const float d = hSdf.getValueInterpolated(v_g)*hSdf.resolution;
            if (d < 0) {

                // collision detected
                float * J = &s[tid*12];

                const float3 hSdfGrad_f = hSdf.getGradientInterpolated(v_g);
                const float3 hSdfGrad_h = SE3Rotate(T_mfs_h[hFrame],hSdfGrad_f);
                const float3 hSdfGrad_o = SE3Rotate(T_oh,hSdfGrad_h);

                // hand derivative
                J[0]  = dot(hSdfGrad_h,make_float3(-1, 0, 0));
                J[1]  = dot(hSdfGrad_h,make_float3( 0,-1, 0));
                J[2]  = dot(hSdfGrad_h,make_float3( 0, 0,-1));

                J[3]  = dot(hSdfGrad_h,make_float3(     0, v_h.z,-v_h.y));
                J[4]  = dot(hSdfGrad_h,make_float3(-v_h.z,     0, v_h.x));
                J[5]  = dot(hSdfGrad_h,make_float3( v_h.y,-v_h.x,     0));

                // object derivative
                J[6]  = dot(hSdfGrad_o,make_float3(-1, 0, 0));
                J[7]  = dot(hSdfGrad_o,make_float3( 0,-1, 0));
                J[8]  = dot(hSdfGrad_o,make_float3( 0, 0,-1));

                J[9]  = dot(hSdfGrad_o,make_float3(     0, v_o.z,-v_o.y));
                J[10] = dot(hSdfGrad_o,make_float3(-v_o.z,     0, v_o.x));
                J[11] = dot(hSdfGrad_o,make_float3( v_o.y,-v_o.x,     0));

                float * eJ = result;
                float * JTJ = &result[12];
                float * e = &result[12 + JTJSize(12)];

                for (int i=0; i<12; ++i) {
                    if (J[i] == 0.0f) { continue; }
                    float eJval = -d*-J[i];
                    atomicAdd(&eJ[i],eJval);
                    for (int j=0; j<=i; ++j) {
                        float JTJval = J[i]*J[j];
                        atomicAdd(&JTJ[((i*(i+1))>>1) + j],JTJval);
                    }
                }
                atomicAdd(e,d*d);

                return;
            }
        }

    }

}

__global__ void gpu_getDistanceToSdf(const float4 * testSites,
                                    const int nSites,
                                    const SE3 T_ds,
                                    const Grid3D<float> * sdf_dst,
                                    float * distances) {

    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    // overflow
    if (index >= nSites) {
        return;
    }

    float4 v_src = testSites[index];
    v_src.w = 1.0;
    float4 v_dst = T_ds*v_src;

    const float3 v_dst_g = sdf_dst->getGridCoords(make_float3(v_dst));

    if (!sdf_dst->isInBoundsGradientInterp(v_dst_g)) {
        distances[index] = 1e20;
          //  printf("%f ",sdf_dst->resolution);
    } else {
        distances[index] = sdf_dst->getValueInterpolated(v_dst_g)*sdf_dst->resolution;
    }

}

// -=-=-=-=-=-=-=-=-=- interface -=-=-=-=-=-=-=-=-=-
int countSelfIntersections(const float4 * testSites,
                           const int nSites,
                           const SE3 * T_mfs,
                           const SE3 * T_fms,
                           const int * sdfFrames,
                           const Grid3D<float> * sdfs,
                           const int nSdfs,
                           const int * potentialIntersection) {

    dim3 block(128,1,1);
    dim3 grid(ceil(nSites/(float)block.x),1,1);

    MirroredVector<int> nCollisions(1);
    cudaMemset(nCollisions.devicePtr(),0,sizeof(int));

    gpu_selfIntersectionCount<<<grid,block>>>(testSites,nSites,T_mfs,T_fms,
                                              sdfFrames,sdfs,nSdfs,potentialIntersection,
                                              nCollisions.devicePtr());

    nCollisions.syncDeviceToHost();
    return nCollisions.hostPtr()[0];

}

int countIntersections(const float4 * testSites,
                       const int nSites,
                       const SE3 & T_ds,
                       const SE3 * T_mfs_src,
                       const int * sdfFrames_src,
                       const SE3 * T_fms_dst,
                       const int * sdfFrames_dst,
                       const Grid3D<float> * sdfs_dst,
                       const int nSdfs_dst) {

    dim3 block(128,1,1);
    dim3 grid(ceil(nSites/(float)block.x),1,1);

    MirroredVector<int> nCollisions(1);
    cudaMemset(nCollisions.devicePtr(),0,sizeof(int));

    gpu_intersectionCount<<<grid,block>>>(testSites,nSites,T_ds,T_mfs_src,sdfFrames_src,
                                          T_fms_dst,sdfFrames_dst,sdfs_dst,nSdfs_dst,
                                          nCollisions.devicePtr());

    nCollisions.syncDeviceToHost();
    return nCollisions.hostPtr()[0];

}

void normEqnsSelfIntersection(const float4 * testSites,
                              const int nSites,
                              const int dims,
                              const MirroredModel & model,
                              const int * potentialIntersection,
                              float * result,
                              float * debugError) {

    dim3 block(64,1,1);
    dim3 grid(ceil(nSites/(float)block.x),1,1);

    cudaMemset(result,0,((dims-6)+JTJSize(dims-6)+1)*sizeof(float));
    if (debugError == 0) {
        gpu_normEqnsSelfIntersection<false><<<grid,block,64*(dims-6)*sizeof(float)>>>(testSites, nSites, dims, model.getDeviceTransformsFrameToModel(), model.getDeviceTransformsModelToFrame(),
                                                                            model.getDeviceSdfFrames(), model.getDeviceSdfs(), model.getNumSdfs(), model.getDeviceDependencies(),
                                                                            model.getDeviceJointTypes(), model.getDeviceJointAxes(),
                                                                            potentialIntersection, result, debugError);
    } else {
        gpu_normEqnsSelfIntersection<true><<<grid,block,64*(dims-6)*sizeof(float)>>>(testSites, nSites, dims, model.getDeviceTransformsFrameToModel(), model.getDeviceTransformsModelToFrame(),
                                                                            model.getDeviceSdfFrames(), model.getDeviceSdfs(), model.getNumSdfs(), model.getDeviceDependencies(),
                                                                            model.getDeviceJointTypes(), model.getDeviceJointAxes(),
                                                                            potentialIntersection, result, debugError);
    }


}

void normEqnsIntersection(const float4 * testSites,
                          const int nSites,
                          const int dims,
                          const SE3 T_ds,
                          const SE3 T_sd,
                          const MirroredModel & srcModel,
                          const MirroredModel & dstModel,
                          float * result,
                          float * debugError) {

    dim3 block(64,1,1);
    dim3 grid(ceil(nSites/(float)block.x),1,1);

    cudaMemset(result,0,((dims)+JTJSize(dims)+1)*sizeof(float));

    if (debugError == 0) {
        gpu_normEquationsIntersection<false><<<grid,block,64*(dims)*sizeof(float)>>>(testSites, nSites, dims,
                                                                                     T_ds, T_sd,
                                                                                     srcModel.getDeviceTransformsFrameToModel(),
                                                                                     srcModel.getDeviceTransformsModelToFrame(),
                                                                                     srcModel.getDeviceSdfFrames(),
                                                                                     dstModel.getDeviceTransformsFrameToModel(),
                                                                                     dstModel.getDeviceTransformsModelToFrame(),
                                                                                     dstModel.getDeviceSdfFrames(),
                                                                                     dstModel.getDeviceSdfs(),
                                                                                     dstModel.getNumSdfs(),
                                                                                     srcModel.getDeviceDependencies(),
                                                                                     srcModel.getDeviceJointTypes(),
                                                                                     srcModel.getDeviceJointAxes(),
                                                                                     result, debugError);
    } else {
        gpu_normEquationsIntersection<true><<<grid,block,64*(dims)*sizeof(float)>>>(testSites, nSites, dims,
                                                                                    T_ds, T_sd,
                                                                                    srcModel.getDeviceTransformsFrameToModel(),
                                                                                    srcModel.getDeviceTransformsModelToFrame(),
                                                                                    srcModel.getDeviceSdfFrames(),
                                                                                    dstModel.getDeviceTransformsFrameToModel(),
                                                                                    dstModel.getDeviceTransformsModelToFrame(),
                                                                                    dstModel.getDeviceSdfFrames(),
                                                                                    dstModel.getDeviceSdfs(),
                                                                                    dstModel.getNumSdfs(),
                                                                                    srcModel.getDeviceDependencies(),
                                                                                    srcModel.getDeviceJointTypes(),
                                                                                    srcModel.getDeviceJointAxes(),
                                                                                    result, debugError);
    }


}


void normEqnsSelfIntersectionReduced(const float4 * testSites,
                                  const int nSites,
                                  const int fullDims,
                                  const int redDims,
                                  const MirroredModel & model,
                                  const float * dtheta_dalpha,
                                  const int * potentialIntersection,
                                  float * result) {

    dim3 block(64,1,1);
    dim3 grid(ceil(nSites/(float)block.x),1,1);

    cudaMemset(result,0,((redDims-6)+JTJSize(redDims-6)+1)*sizeof(float));
    gpu_normEqnsSelfIntersectionReduced<<<grid,block,64*(fullDims-6+redDims-6)*sizeof(float)>>>(testSites, nSites, fullDims, redDims,
                                                                                             model.getDeviceTransformsFrameToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceSdfFrames(), model.getDeviceSdfs(), model.getNumSdfs(),
                                                                                             model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(),
                                                                                             dtheta_dalpha, potentialIntersection, result);


}

void normEqnsSelfIntersectionParamMap(const float4 * testSites,
                                      const int nSites,
                                      const int fullDims,
                                      const int redDims,
                                      const MirroredModel & model,
                                      const int * dMapping,
                                      const int * potentialIntersection,
                                      float * result) {

    dim3 block(64,1,1);
    dim3 grid(ceil(nSites/(float)block.x),1,1);

    cudaMemset(result,0,((redDims-6)+JTJSize(redDims-6)+1)*sizeof(float));
    gpu_normEqnsSelfIntersectionParamMap<<<grid,block,64*(fullDims-6+redDims-6)*sizeof(float)>>>(testSites, nSites, fullDims, redDims,
                                                                                                 model.getDeviceTransformsFrameToModel(),
                                                                                                 model.getDeviceTransformsModelToFrame(),
                                                                                                 model.getDeviceSdfFrames(),
                                                                                                 model.getDeviceSdfs(),
                                                                                                 model.getNumSdfs(),
                                                                                                 model.getDeviceDependencies(),
                                                                                                 model.getDeviceJointTypes(),
                                                                                                 model.getDeviceJointAxes(),
                                                                                                 dMapping,
                                                                                                 potentialIntersection, result);


}

void normEqnsIntersectionReduced(const float4 * testSites,
                                 const int nSites,
                                 const int fullDims,
                                 const int redDims,
                                 const SE3 T_ds,
                                 const SE3 T_sd,
                                 const MirroredModel & srcModel,
                                 const MirroredModel & dstModel,
                                 const float * dtheta_dalpha_src,
                                 float * result) {

    dim3 block(64,1,1);
    dim3 grid(ceil(nSites/(float)block.x),1,1);

    cudaMemset(result,0,((redDims)+JTJSize(redDims)+1)*sizeof(float));
    gpu_normEqnsIntersectionReduced<<<grid,block,64*(fullDims+redDims)*sizeof(float)>>>(testSites, nSites, fullDims, redDims,
                                                                                        T_ds, T_sd,
                                                                                        srcModel.getDeviceTransformsFrameToModel(),
                                                                                        srcModel.getDeviceTransformsModelToFrame(),
                                                                                        srcModel.getDeviceSdfFrames(),
                                                                                        dstModel.getDeviceTransformsFrameToModel(),
                                                                                        dstModel.getDeviceTransformsModelToFrame(),
                                                                                        dstModel.getDeviceSdfFrames(),
                                                                                        dstModel.getDeviceSdfs(),
                                                                                        dstModel.getNumSdfs(),
                                                                                        srcModel.getDeviceDependencies(),
                                                                                        srcModel.getDeviceJointTypes(),
                                                                                        srcModel.getDeviceJointAxes(),
                                                                                        dtheta_dalpha_src, result);


}

void normEqnsIntersectionParamMap(const float4 * testSites,
                                  const int nSites,
                                  const int fullDims,
                                  const int redDims,
                                  const SE3 T_ds,
                                  const SE3 T_sd,
                                  const MirroredModel & srcModel,
                                  const MirroredModel & dstModel,
                                  const int * dMapping_src,
                                  float * result) {

    dim3 block(64,1,1);
    dim3 grid(ceil(nSites/(float)block.x),1,1);

    cudaMemset(result,0,((redDims)+JTJSize(redDims)+1)*sizeof(float));
    gpu_normEqnsIntersectionParamMap<<<grid,block,64*(fullDims+redDims)*sizeof(float)>>>(testSites, nSites, fullDims, redDims,
                                                                                         T_ds, T_sd, srcModel.getDeviceTransformsFrameToModel(), srcModel.getDeviceTransformsModelToFrame(),
                                                                                         srcModel.getDeviceSdfFrames(), dstModel.getDeviceTransformsFrameToModel(),
                                                                                         dstModel.getDeviceTransformsModelToFrame(), dstModel.getDeviceSdfFrames(), dstModel.getDeviceSdfs(),
                                                                                         dstModel.getNumSdfs(), srcModel.getDeviceDependencies(), srcModel.getDeviceJointTypes(), srcModel.getDeviceJointAxes(),
                                                                                         dMapping_src, result);

}


void intersectionCheckRigidObjInHand(const float4 * testSites,
                                     const int nSites,
                                     const SE3 T_ho,
                                     const SE3 T_oh,
                                     const SE3 * T_mfs_h,
                                     const SE3 * T_fms_h,
                                     const int * sdfFrames_h,
                                     const Grid3D<float> * sdfs_h,
                                     const int nSdfs_h,
                                     float * result) {

    dim3 block(64,1,1);
    dim3 grid(ceil(nSites/(float)block.x),1,1);

    cudaMemset(result,0,(12+JTJSize(12)+1)*sizeof(float));
    gpu_intersectionCheckRigidObjInHand<<<grid,block,64*12*sizeof(float)>>>(testSites, nSites,
                                                                            T_ho, T_oh,
                                                                            T_mfs_h, T_fms_h,
                                                                            sdfFrames_h, sdfs_h,
                                                                            nSdfs_h, result);

}


void getDistanceToSdf(const float4 * testSites,
                      const int nSites,
                      const SE3 T_ds,
                      const Grid3D<float> * sdf_dst,
                      float * distances,
                      const cudaStream_t stream) {

    dim3 block(64,1,1);
    dim3 grid(ceil(nSites/(float)block.x),1,1);

    gpu_getDistanceToSdf<<<grid,block,0,stream>>>(testSites,nSites,T_ds,sdf_dst,distances);

}

void initDebugIntersectionError(float * debugError,
                                const int nSites) {

    dim3 block(64,1,1);
    dim3 grid(ceil(nSites/(float)block.x),1,1);

    gpu_initDebugIntersectionError<<<grid,block>>>(debugError, nSites);

}

}
