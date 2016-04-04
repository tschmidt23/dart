#include "modToObs.h"

#include "kernel_common.h"
#include "geometry/grid_3d.h"
#include "geometry/SE3.h"
#include "optimization/optimization.h"
#include "util/mirrored_memory.h"

namespace dart {

static const float truncVal = 1000.0;

// -=-=-=-=-=-=-=-=-=- kernels -=-=-=-=-=-=-=-=-=-
template <bool dbgDA, bool dbgErr, bool dbgNorm>
__global__ void gpu_normEqnsModToObs(const int dims,
                                     const float4 * labeledPredictedVertMap,
                                     const int width,
                                     const int height,
                                     const int modelNum,
                                     const SE3 T_mc,
                                     const SE3 * T_fms,
                                     const SE3 * T_mfs,
                                     const Grid3D<float> * obsSdf,
                                     const int * labelFrames,
                                     const int * dependencies,
                                     const JointType * jointTypes,
                                     const float3 * jointAxes,
                                     float * result,
                                     int * numPredictions,
                                     int * debugDataAssociation,
                                     float * debugError,
                                     float4 * debugNorm) {

    extern __shared__ float s[];

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    // overflow
    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    const int tid = threadIdx.x + blockDim.x*threadIdx.y;

    if (dbgDA) { if (modelNum == 0) { debugDataAssociation[index] = -1; } }
    if (dbgErr) { if (modelNum == 0) { debugError[index] = NAN; } }
    if (dbgNorm) { debugNorm[index] = make_float4(0); }

    const float4 & predV_c = labeledPredictedVertMap[index];

    // no prediction
    if (predV_c.z == 0) { return; }

    const float3 predV_m = SE3Transform(T_mc,make_float3(predV_c));

    const float3 predVGrid = obsSdf->getGridCoords(predV_m);
    if (!obsSdf->isInBoundsGradientInterp(predVGrid)) {
        return;
    }

    const float residual = obsSdf->getValueInterpolated(predVGrid)*obsSdf->resolution;

    if (dbgErr) { debugError[index] = residual; }

    const int label = round(predV_c.w);
    const int model = label >> 16;
    const int sdf = label & 65535;
    if (model != modelNum) {
        return;
    }

    if (dbgDA) { debugDataAssociation[index] = label; }

    const int predFrame = labelFrames[sdf];

    float * J = &s[tid*dims];

    const float3 sdfGrad_m = obsSdf->getGradientInterpolated(predVGrid);

    if (dbgNorm) { debugNorm[index] = make_float4(sdfGrad_m,1); }

//    const float3 sdfGrad_m = SE3Rotate(T_mc,sdfGrad_m);

    getErrorJacobianOfModelPoint(J,make_float4(predV_m,1),predFrame,sdfGrad_m,dims,dependencies,jointTypes,jointAxes,T_fms,T_mfs);

    atomicAdd(numPredictions,1);

    float * JTr = result;
    float * JTJ = &result[dims];
    float * e = &result[dims + JTJSize(dims)];

//    //#pragma unroll
//    for (int i=0; i<dims; i++) {
//        if( J[i] == 0.0f)  continue;
//        float v = residual*J[i];
//        atomicAdd(&JTr[i],v);
//        //#pragma unroll
//        for (int j=0; j<=i; j++) {
//            float v2 = J[i]*J[j];
//            atomicAdd(&JTJ[((i*(i+1))>>1) + j],v2);
//        }
//    }

//    atomicAdd(e,0.5*residual*residual);

    computeSquaredLossResult(dims,residual,J,e,JTr,JTJ);
}

template <bool dbgDA, bool dbgErr, bool dbgNorm>
__global__ void gpu_normEqnsModToObsTruncated(const int dims,
                                              const float4 * labeledPredVertMap,
                                              const int width,
                                              const int height,
                                              const int modelNum,
                                              const SE3 T_mc,
                                              const SE3 * T_fms,
                                              const SE3 * T_mfs,
                                              const Grid3D<float> * obsSdf,
                                              const int * labelFrames,
                                              const int * dependencies,
                                              const JointType * jointTypes,
                                              const float3 * jointAxes,
                                              const float truncationDist,
                                              float * result,
                                              int * numPredictions,
                                              int * debugDataAssociation,
                                              float * debugError,
                                              float4 * debugNorm) {

    extern __shared__ float s[];

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    // overflow
    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    const int tid = threadIdx.x + blockDim.x*threadIdx.y;

    if (dbgDA) { if (modelNum == 0) { debugDataAssociation[index] = -1; } }
    if (dbgErr) { if (modelNum == 0) { debugError[index] = NAN; } }
    if (dbgNorm) { debugNorm[index] = make_float4(0); }

    const float4 & predV_c = labeledPredVertMap[index];

    // no prediction
    if (predV_c.z == 0) { return; }

    const float3 predV_m = SE3Transform(T_mc,make_float3(predV_c));

    const float3 predVGrid = obsSdf->getGridCoords(predV_m);
    if (!obsSdf->isInBoundsGradientInterp(predVGrid)) {
        return;
    }

    const float err = obsSdf->getValueInterpolated(predVGrid)*obsSdf->resolution;

    // make sure we're in the truncation region and violating free space
    if (err >= truncationDist || err < 0) {
        return;
    }

    if (dbgErr) { debugError[index] = err; }

//    const float4 predV_m = T_mc*make_float4(predV_c.x,predV_c.y,predV_c.z,1);
    const int label = round(predV_c.w);
    const int model = label >> 16;
    const int sdf = label & 65535;
    if (model != modelNum) {
        return;
    }

    if (dbgDA) { debugDataAssociation[index] = label; }

    const int predFrame = labelFrames[sdf];

    float * J = &s[tid*dims];

    const float3 sdfGrad_m = obsSdf->getGradientInterpolated(predVGrid);

    if (dbgNorm) { debugNorm[index] = make_float4(sdfGrad_m,1); }

//    const float3 sdfGrad_m = SE3Rotate(T_mc,sdfGrad_m);

    getErrorJacobianOfModelPoint(J,make_float4(predV_m,1),predFrame,sdfGrad_m,dims,dependencies,jointTypes,jointAxes,T_fms,T_mfs);

    atomicAdd(numPredictions,1);

    float * eJ = result;
    float * JTJ = &result[dims];
    float * e = &result[dims + JTJSize(dims)];

    //#pragma unroll
    for (int i=0; i<dims; i++) {
        if( J[i] == 0.0f)  continue;
        float v = err*J[i];
        atomicAdd(&eJ[i],v);
        //#pragma unroll
        for (int j=0; j<=i; j++) {
            float v2 = J[i]*J[j];
            atomicAdd(&JTJ[((i*(i+1))>>1) + j],v2);
        }
    }

    atomicAdd(e,0.5*err*err);

}

template <bool dbgDA, bool dbgErr, bool dbgNorm>
__global__ void gpu_normEqnsModToObsReduced(const int fullDims,
                                            const int redDims,
                                            const float4 * labeledPredictedVertMap,
                                            const int width,
                                            const int height,
                                            const int modelNum,
                                            const SE3 T_mc,
                                            const SE3 * T_fms,
                                            const SE3 * T_mfs,
                                            const Grid3D<float> * obsSdf,
                                            const int * labelFrames,
                                            const int * dependencies,
                                            const JointType * jointTypes,
                                            const float3 * jointAxes,
                                            const float * dtheta_dalpha,
                                            float * result,
                                            int * numPredictions,
                                            int * debugDataAssociation,
                                            float * debugError,
                                            float4 * debugNorm) {

    extern __shared__ float s[];

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    // overflow
    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    const int tid = threadIdx.x + blockDim.x*threadIdx.y;

    if (dbgDA) { if (modelNum == 0) { debugDataAssociation[index] = -1; } }
    if (dbgErr) { if (modelNum == 0) { debugError[index] = NAN; } }
    if (dbgNorm) { debugNorm[index] = make_float4(0); }

    const float4 & predV_c = labeledPredictedVertMap[index];

    // no prediction
    if (predV_c.z == 0) { return; }

    const float3 predV_m = SE3Transform(T_mc,make_float3(predV_c));

    const float3 predVGrid = obsSdf->getGridCoords(predV_m);
    if (!obsSdf->isInBoundsGradientInterp(predVGrid)) {
        return;
    }

    const float residual = obsSdf->getValueInterpolated(predVGrid)*obsSdf->resolution;

    if (dbgErr) { debugError[index] = residual; }

    const int label = round(predV_c.w);
    const int model = label >> 16;
    const int sdf = label & 65535;
    if (model != modelNum) {
        return;
    }

    if (dbgDA) { debugDataAssociation[index] = label; }

    const int predFrame = labelFrames[sdf];

    // array declarations
    float * de_dtheta = &s[tid*(fullDims+redDims)];
    float * J = &s[tid*(fullDims+redDims) + fullDims];

    const float3 sdfGrad_m = obsSdf->getGradientInterpolated(predVGrid);

    if (dbgNorm) { debugNorm[index] = make_float4(sdfGrad_m,1); }

    atomicAdd(numPredictions,1);

    getErrorJacobianOfModelPoint(de_dtheta,make_float4(predV_m,1),predFrame,sdfGrad_m,fullDims,dependencies,jointTypes,jointAxes,T_fms,T_mfs);

    doPoseGradientReduction(J,de_dtheta,dtheta_dalpha,fullDims,redDims);

    float * JTr = result;
    float * JTJ = &result[redDims];
    float * e = &result[redDims + JTJSize(redDims)];

    //#pragma unroll
//    for (int i=0; i<redDims; i++) {
//        if( J[i]==0.0f)  continue;
//        float v = residual*J[i];
//        atomicAdd(&JTr[i],v);
//        //#pragma unroll
//        for (int j=0; j<=i; j++) {
//            float v2 = J[i]*J[j];
//            atomicAdd(&JTJ[((i*(i+1))>>1) + j],v2);
//        }
//    }
//    atomicAdd(e,0.5*residual*residual);

    computeSquaredLossResult(redDims,residual,J,e,JTr,JTJ);

}

template <bool dbgDA, bool dbgErr, bool dbgNorm>
__global__ void gpu_normEqnsModToObsParamMap(const int fullDims,
                                             const int redDims,
                                             const float4 * labeledPredictedVertMap,
                                             const int width,
                                             const int height,
                                             const int modelNum,
                                             const SE3 T_mc,
                                             const SE3 * T_fms,
                                             const SE3 * T_mfs,
                                             const Grid3D<float> * obsSdf,
                                             const int * labelFrames,
                                             const int * dependencies,
                                             const JointType * jointTypes,
                                             const float3 * jointAxes,
                                             const int * dMapping,
                                             float * result,
                                             int * numPredictions,
                                             int * debugDataAssociation,
                                             float * debugError,
                                             float4 * debugNorm) {

    extern __shared__ float s[];

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    // overflow
    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    const int tid = threadIdx.x + blockDim.x*threadIdx.y;

    if (dbgDA) { if (modelNum == 0) { debugDataAssociation[index] = -1; } }
    if (dbgErr) { if (modelNum == 0) { debugError[index] = NAN; } }
    if (dbgNorm) { debugNorm[index] = make_float4(0); }

    const float4 & predV_c = labeledPredictedVertMap[index];

    // no prediction
    if (predV_c.z == 0) { return; }

    const float3 predV_m = SE3Transform(T_mc,make_float3(predV_c));

    const float3 predVGrid = obsSdf->getGridCoords(predV_m);
    if (!obsSdf->isInBoundsGradientInterp(predVGrid)) {
        return;
    }

    const float residual = obsSdf->getValueInterpolated(predVGrid)*obsSdf->resolution;

    if (dbgErr) { debugError[index] = residual; }

    const int label = round(predV_c.w);
    const int model = label >> 16;
    const int sdf = label & 65535;
    if (model != modelNum) {
        return;
    }

    if (dbgDA) { debugDataAssociation[index] = label; }

    const int predFrame = labelFrames[sdf];

    // array declarations
    float * de_dtheta = &s[tid*(fullDims+redDims)];
    float * J = &s[tid*(fullDims+redDims) + fullDims];

    const float3 sdfGrad_m = obsSdf->getGradientInterpolated(predVGrid);

    if (dbgNorm) { debugNorm[index] = make_float4(sdfGrad_m,1); }

    atomicAdd(numPredictions,1);

    getErrorJacobianOfModelPoint(de_dtheta,make_float4(predV_m,1),predFrame,sdfGrad_m,fullDims,dependencies,jointTypes,jointAxes,T_fms,T_mfs);

    doParamMapping(J,de_dtheta,dMapping,fullDims,redDims);

    float * JTr = result;
    float * JTJ = &result[redDims];
    float * e = &result[redDims + JTJSize(redDims)];

    computeSquaredLossResult(redDims,residual,J,e,JTr,JTJ);

}

__global__ void gpu_splatObsSdf(const float4 * dObsVertMap,
                                const int width,
                                const int height,
                                const SE3 T_cm,
                                const Grid3D<float> * dObsSdf,
                                const float focalLength) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int z = blockIdx.z*blockDim.z + threadIdx.z;

    const float3 & o = dObsSdf->offset;
    const float & resolution = dObsSdf->resolution;

    // TODO: think about this
    //  const float3 center = o + resolution*make_float3( x + 0.5, y + 0.5, z + 0.5);
    const float3 center = SE3Transform(T_cm,o + resolution*make_float3( x , y , z ));

    const int u = round( (focalLength/center.z)*center.x + (width>>1) );
    const int v = round( (focalLength/center.z)*center.y + (height>>1) );

    float & splatVal = dObsSdf->data[x + dObsSdf->dim.x*(y + dObsSdf->dim.y*z)];

    if (u < 0 || u >= width || v < 0 || v >= height) {
        splatVal = truncVal;
    } else if (dObsVertMap[u + v*width].w == 0 || dObsVertMap[u + v*width].z == 0) {
        splatVal = 0.5*truncVal; // TODO: think about this
//    } else {
//        float sdfWorld = (dObsVertMap[u + v*width].z - center.z);
//        float sdf = (sdfWorld)/dObsSdf->resolution;
//        splatVal = fmaxf(0, fminf(truncVal, sdf));
//    }
    } else if (dObsVertMap[u + v*width].z < center.z) {
        splatVal = 0;
    } else {
        splatVal = truncVal;
    }

}

__global__ void gpu_clearObsSdf(const Grid3D<float> * dObsSdf,
                                const float truncationDist) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int z = blockIdx.z*blockDim.z + threadIdx.z;

    dObsSdf->data[x + dObsSdf->dim.x*(y + dObsSdf->dim.y*z)] = truncationDist;
}

__global__ void gpu_computeTruncatedObsDf(const float4 * dObsVertMap,
                                          const int width,
                                          const int height,
                                          const SE3 T_mc,
                                          const Grid3D<float> * dObsSdf,
                                          const float truncationDist) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int z = threadIdx.z;

    if (x >= width-1 || y >= height-1) { return; }

    float4 pA;
    float4 pB;
    float4 pC;
    if (z == 0) {
        pA = dObsVertMap[x   +     y*width];
        pB = dObsVertMap[x+1 +     y*width];
        pC = dObsVertMap[x+1 + (y+1)*width];
    } else {
        pA = dObsVertMap[x   +     y*width];
        pC = dObsVertMap[x+1 + (y+1)*width];
        pB = dObsVertMap[x   + (y+1)*width];
    }
    if (pA.w != 0 && pB.w != 0 && pC.w != 0 ) {

        //printf("%d, %d\n",x,y);

        const float3 pAg = dObsSdf->getGridCoords(make_float3(T_mc*pA));
        const float3 pBg = dObsSdf->getGridCoords(make_float3(T_mc*pB));
        const float3 pCg = dObsSdf->getGridCoords(make_float3(T_mc*pC));

        const float3 minPoint = fminf(pAg,fminf(pBg,pCg));
        const float3 maxPoint = fmaxf(pAg,fmaxf(pBg,pCg));

        const float3 E0 = pAg - pBg;
        const float3 E1 = pCg - pBg;
        float a = dot(E0,E0);
        float b = dot(E0,E1);
        float c = dot(E1,E1);
        float det = a*c-b*b;

        for (int gz=max(0,(int)floor(minPoint.z-truncationDist)); gz< min((int)ceil(maxPoint.z+truncationDist),dObsSdf->dim.z); ++gz) {
            for (int gy=max(0,(int)floor(minPoint.y-truncationDist)); gy< min((int)ceil(maxPoint.y+truncationDist),dObsSdf->dim.y); ++gy) {
                for (int gx=max(0,(int)floor(minPoint.x-truncationDist)); gx< min((int)ceil(maxPoint.x+truncationDist),dObsSdf->dim.x); ++gx) {

                    //printf("> %d, %d, %d\n",gx,gy,gz);
                    float & sdfVal = dObsSdf->data[gx + dObsSdf->dim.x*(gy + dObsSdf->dim.y*gz)];

                    const float3 P = make_float3(gx+0.5,gy+0.5,gz+0.5);
                    const float3 D = pBg - P;
                    float d = dot(E0,D);
                    float e = dot(E1,D);
                    float f = dot(D,D);

                    float s = b*e - c*d;
                    float t = b*d - a*e;

                    int region;
                    if ( s+t <= det) {
                        if ( s < 0 ) {
                            if ( t < 0 ) {
                                region = 4;
                            } else {
                                region = 3;
                            }
                        } else if ( t < 0 ) {
                            region = 5;
                        } else {
                            region = 0;
                        }
                    } else {
                        if ( s < 0 ) {
                            region = 2;
                        } else if ( t < 0) {
                            region = 6;
                        } else {
                            region = 1;
                        }
                    }

                    switch (region) {
                        case 0:
                            {
                                float invDet = 1/det;
                                s*= invDet;
                                t*= invDet;
                            }
                            break;
                        case 1:
                            {
                                float numer = c + e - b - d;
                                if (numer <= 0) {
                                    s = 0;
                                } else {
                                    float denom = a - 2*b + c;
                                    s = ( numer >= denom ? 1 : numer/denom );
                                }
                                t = 1-s;
                            }
                            break;
                        case 2:
                            {
                                float tmp0 = b+d;
                                float tmp1 = c+e;
                                if ( tmp1 > tmp0 ) { // min on edge s+1=1
                                    float numer = tmp1 - tmp0;
                                    float denom = a - 2*b + c;
                                    s = ( numer >= denom ? 1 : numer/denom );
                                    t = 1-s;
                                } else { // min on edge s=0
                                    s = 0;
                                    t = ( tmp1 <= 0 ? 1 : ( e >= 0 ? 0 : -e/c ) );
                                }
                            }
                            break;
                        case 3:
                            s = 0;
                            t = ( e >= 0 ? 0 :
                                           ( -e >= c ? 1 : -e/c ) );
                            break;
                        case 4:
                            if ( d < 0 ) { // min on edge t=0
                                t = 0;
                                s = ( d >= 0 ? 0 :
                                               ( -d >= a ? 1 : -d/a ) );
                            } else { // min on edge s = 0
                                s = 0;
                                t = ( e >= 0 ? 0 :
                                               ( -e >= c ? 1 : -e/c ) );
                            }
                            break;
                        case 5:
                            t = 0;
                            s = ( d >= 0 ? 0 :
                                           ( -d >= a ? 1 : -d/a ) );
                            break;
                        case 6:
                            {
                                float tmp0 = a+d;
                                float tmp1 = b+e;
                                if (tmp0 > tmp1) { // min on edge s+1=1
                                    float numer = c + e - b - d;
                                    float denom = a -2*b + c;
                                    s = ( numer >= denom ? 1 : numer/denom );
                                    t = 1-s;
                                } else { // min on edge t=1
                                    t = 0;
                                    s = ( tmp0 <= 0 ? 1 : ( d >= 0 ? 0 : -d/a ));
                                }
                            }
                            break;
                    }

                    const float3 closestPoint = pBg + s*E0 + t*E1;
                    const float3 v = closestPoint-P;

                    float dist = length(v);
                    float3 unscaledNorm = cross(pAg-pBg,pCg-pBg);
                    if (dot(v,unscaledNorm) < 0) { dist = -dist; }

                    //atomicMin(&sdfVal,length);
                    // TODO
                    //sdfVal = min(sdfVal,list);
                    if (fabs(dist) < fabs(sdfVal)) { sdfVal = dist; }
                    if (fabs(dist) < fabs(sdfVal)) { sdfVal = dist; }
                    if (fabs(dist) < fabs(sdfVal)) { sdfVal = dist; }
                    //printf("%f\n",sdfVal);
                }
            }
        }
    }

}

__global__ void gpu_signTruncatedObsDf(const float4 * dObsVertMap,
                                       const int width,
                                       const int height,
                                       const SE3 T_cm,
                                       const Grid3D<float> * dObsSdf,
                                       const float focalLength) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width-1 || y >= height-1) { return; }

    if (dObsVertMap[x + y*width].w != 0 && dObsVertMap[x+1 + y*width].w != 0 && dObsVertMap[x+1 + (y+1)*width].w != 0 ) {



    }

}

__global__ void gpu_errorModToObs(const float4 * labeledPredVertMap,
                                  const int width,
                                  const int height,
                                  const Grid3D<float> * obsSdf,
                                  float* result) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    // overflow
    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;

    const float4 & predV = labeledPredVertMap[index];

    // no prediction
    if (predV.z == 0) { return; }

    const float3 predVGrid = obsSdf->getGridCoords(make_float3(predV));
    if (!obsSdf->isInBoundsGradientInterp(predVGrid)) {
        return;
    }

    const float err = obsSdf->getValueInterpolated(predVGrid)*obsSdf->resolution;

    //atomicAdd(numPredictions,1);
    atomicAdd(result,0.5*err*err);

}

__global__ void gpu_cullUnobservable(float4 * predVertMap,
                                     const int predWidth,
                                     const int predHeight,
                                     const float4 * obsVertMap,
                                     const int obsWidth,
                                     const int obsHeight) {

    const int predX = blockIdx.x*blockDim.x + threadIdx.x;
    const int predY = blockIdx.y*blockDim.y + threadIdx.y;

    if (predX >= predWidth || predY >= predHeight) { return; }

    const int predIndex = predX + predY*predWidth;

    const int obsX = predX*obsWidth/predWidth;
    const int obsY = predY*obsHeight/predHeight;

    const int obsIndex = obsX + obsY*obsWidth;

    if (obsVertMap[obsIndex].w <= 0            || //obsVertMap[obsIndex].z == 0 ||
        obsVertMap[obsIndex+1].w <= 0          || //obsVertMap[obsIndex+1].z == 0 ||
        obsVertMap[obsIndex+obsWidth].w <= 0   || //obsVertMap[obsIndex+obsWidth].z == 0 ||
        obsVertMap[obsIndex+obsWidth+1].w <= 0 //|| obsVertMap[obsIndex+obsWidth+1].z == 0
            ) {
        predVertMap[predIndex].z = 0;
    }

}


// -=-=-=-=-=-=-=-=-=- interface -=-=-=-=-=-=-=-=-=-
void normEqnsModToObs(const int dimensions,
                      const float4 * dLabeledPredictedVertMap,
                      const int width,
                      const int height,
                      const MirroredModel & model,
                      const SE3 T_gc,
                      float * dResult,
                      int * numPredictions,
                      int * debugDataAssociation,
                      float * debugError,
                      float4 * debugNorm) {

    dim3 block;
    if (height == 1) {
        block.x = 64; block.y = block.z = 1;
    }
    else {
        block.x = 8; block.y = 8; block.z = 1;
    }
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    cudaMemset(dResult,0,(dimensions + JTJSize(dimensions) + 1)*sizeof(float));
    cudaMemset(numPredictions,0,sizeof(int));

    {

        if (debugDataAssociation == 0) {
            if (debugError == 0) {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObs<false,false,false><<<grid,block,64*dimensions*sizeof(float)>>>(dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), T_gc, model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObs<false,false,true><<<grid,block,64*dimensions*sizeof(float)>>> (dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), T_gc, model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
            else {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObs<false,true,false><<<grid,block,64*dimensions*sizeof(float)>>> (dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), T_gc, model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObs<false,true,true><<<grid,block,64*dimensions*sizeof(float)>>>  (dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), T_gc, model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
        }
        else {
            if (debugError == 0) {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObs<true,false,false><<<grid,block,64*dimensions*sizeof(float)>>> (dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), T_gc, model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObs<true,false,true><<<grid,block,64*dimensions*sizeof(float)>>>  (dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), T_gc, model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
            else {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObs<true,true,false><<<grid,block,64*dimensions*sizeof(float)>>>  (dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), T_gc, model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObs<true,true,true><<<grid,block,64*dimensions*sizeof(float)>>>   (dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), T_gc, model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
        }

#ifdef CUDA_ERR_CHECK
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("normEqnsModToObs: %s\n",cudaGetErrorString(err));
        }
#endif
    }

}

void normEqnsModToObsTruncated(const int dimensions,
                               const float4 * dLabeledPredictedVertMap,
                               const int width,
                               const int height,
                               const MirroredModel & model,
                               const float truncationDistance,
                               float * dResult,
                               int * numPredictions,
                               int * debugDataAssociation,
                               float * debugError,
                               float4 * debugNorm) {

    dim3 block;
    if (height == 1) {
        block.x = 64; block.y = block.z = 1;
    }
    else {
        block.x = 8; block.y = 8; block.z = 1;
    }
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    cudaMemset(dResult,0,(dimensions + JTJSize(dimensions) + 1)*sizeof(float));
    cudaMemset(numPredictions,0,sizeof(int));

    {

        if (debugDataAssociation == 0) {
            if (debugError == 0) {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObsTruncated<false,false,false><<<grid,block,64*dimensions*sizeof(float)>>>(dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), truncationDistance, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObsTruncated<false,false,true> <<<grid,block,64*dimensions*sizeof(float)>>>(dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), truncationDistance, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
            else {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObsTruncated<false,true,false> <<<grid,block,64*dimensions*sizeof(float)>>>(dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), truncationDistance, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObsTruncated<false,true,true>  <<<grid,block,64*dimensions*sizeof(float)>>>(dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), truncationDistance, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
        }
        else {
            if (debugError == 0) {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObsTruncated<true,false,false> <<<grid,block,64*dimensions*sizeof(float)>>>(dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), truncationDistance, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObsTruncated<true,false,true>  <<<grid,block,64*dimensions*sizeof(float)>>>(dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), truncationDistance, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
            else {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObsTruncated<true,true,false>  <<<grid,block,64*dimensions*sizeof(float)>>>(dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), truncationDistance, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObsTruncated<true,true,true>   <<<grid,block,64*dimensions*sizeof(float)>>>(dimensions, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), truncationDistance, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
        }

#ifdef CUDA_ERR_CHECK
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("normEqnsModToObs: %s\n",cudaGetErrorString(err));
        }
#endif
    }


}

void normEqnsModToObsReduced(const int dims,
                             const int reductionDims,
                             const float * d_dtheta_dalpha,
                             const float4 * dLabeledPredictedVertMap,
                             const int width,
                             const int height,
                             const MirroredModel & model,
                             float * dResult,
                             int * numPredictions,
                             int * debugDataAssociation,
                             float * debugError,
                             float4 * debugNorm) {

    dim3 block;
    if (height == 1) {
        block.x = 64; block.y = block.z = 1;
    }
    else {
        block.x = 8; block.y = 8; block.z = 1;
    }
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    cudaMemset(dResult,0,(reductionDims + JTJSize(reductionDims) + 1)*sizeof(float));

    {

        if (debugDataAssociation == 0) {
            if (debugError == 0) {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObsReduced<false,false,false><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>>(dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), d_dtheta_dalpha, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObsReduced<false,false,true><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>> (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), d_dtheta_dalpha, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
            else {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObsReduced<false,true,false><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>> (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), d_dtheta_dalpha, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObsReduced<false,true,true><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>>  (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), d_dtheta_dalpha, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
        }
        else {
            if (debugError == 0) {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObsReduced<true,false,false><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>> (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), d_dtheta_dalpha, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObsReduced<true,false,true><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>>  (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), d_dtheta_dalpha, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
            else {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObsReduced<true,true,false><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>>  (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), d_dtheta_dalpha, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObsReduced<true,true,true><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>>   (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), d_dtheta_dalpha, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
        }

#ifdef CUDA_ERR_CHECK
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("normEqnsModToObsReduced: %s\n",cudaGetErrorString(err));
        }
#endif
    }

}

void normEqnsModToObsParamMap(const int dims,
                              const int reductionDims,
                              const int * dMapping,
                              const float4 * dLabeledPredictedVertMap,
                              const int width,
                              const int height,
                              const MirroredModel & model,
                              float * dResult,
                              int * numPredictions,
                              int * debugDataAssociation,
                              float * debugError,
                              float4 * debugNorm) {

    dim3 block;
    if (height == 1) {
        block.x = 64; block.y = block.z = 1;
    }
    else {
        block.x = 8; block.y = 8; block.z = 1;
    }
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    cudaMemset(dResult,0,(reductionDims + JTJSize(reductionDims) + 1)*sizeof(float));

    {

        if (debugDataAssociation == 0) {
            if (debugError == 0) {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObsParamMap<false,false,false><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>>(dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dMapping, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObsParamMap<false,false,true><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>> (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dMapping, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
            else {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObsParamMap<false,true,false><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>> (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dMapping, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObsParamMap<false,true,true><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>>  (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dMapping, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
        }
        else {
            if (debugError == 0) {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObsParamMap<true,false,false><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>> (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dMapping, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObsParamMap<true,false,true><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>>  (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dMapping, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
            else {
                if (debugNorm == 0) {
                    gpu_normEqnsModToObsParamMap<true,true,false><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>>  (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dMapping, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                } else {
                    gpu_normEqnsModToObsParamMap<true,true,true><<<grid,block,64*(dims+reductionDims)*sizeof(float)>>>   (dims, reductionDims, dLabeledPredictedVertMap, width, height, model.getModelID(), model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceTransformsFrameToModel(), model.getDeviceObsSdf(), model.getDeviceSdfFrames(), model.getDeviceDependencies(), model.getDeviceJointTypes(), model.getDeviceJointAxes(), dMapping, dResult, numPredictions, debugDataAssociation, debugError, debugNorm);
                }
            }
        }

#ifdef CUDA_ERR_CHECK
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("normEqnsModToObsReduced: %s\n",cudaGetErrorString(err));
        }
#endif
    }

}

void splatObsSdfZeros(const float4 * dObsVertMap,
                      const int width,
                      const int height,
                      const SE3 & T_cm,
                      const Grid3D<float> * dObsSdf,
                      const uint3 sdfDim,
                      const float focalLength) {

    dim3 block(8,8,4);
    dim3 grid(sdfDim.x / block.x, sdfDim.y / block.y, sdfDim.z / block.z );

    gpu_splatObsSdf<<<grid,block>>>(dObsVertMap,
                                    width,
                                    height,
                                    T_cm,
                                    dObsSdf,
                                    focalLength);

}

void computeTruncatedObsSdf(const float4 * dObsVertMap,
                            const int width,
                            const int height,
                            const SE3 & T_mc,
                            const Grid3D<float> * dObsSdf,
                            const uint3 sdfDim,
                            const float truncationDist) {

    dim3 block(8,8,4);
    dim3 grid(sdfDim.x / block.x, sdfDim.y / block.y, sdfDim.z / block.z );

    gpu_clearObsSdf<<<grid,block>>>(dObsSdf, truncationDist);


    block = dim3(16,8,2);
    grid = dim3( ceil( width / (float)block.x), ceil(height / (float)block.y ), 1);

    gpu_computeTruncatedObsDf<<<grid,block>>>(dObsVertMap,width,height,T_mc,dObsSdf,truncationDist);
    //gpu_signTruncatedObsDf<<<grid,block>>>(dObsVertMap,width,height,T_cm,dObsSdf,focalLength);

}

float errorModToObs(const float4 *dLabeledPredictedVertMap,
                    const int width,
                    const int height,
                    const Grid3D<float> *dObsSdf) {

    dim3 block(16,8);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    static MirroredVector<float> error(1);
    cudaMemset(error.devicePtr(),0,sizeof(float));

    gpu_errorModToObs<<<grid,block>>>(dLabeledPredictedVertMap,width,height,dObsSdf,error.devicePtr());

    error.syncDeviceToHost();
    return error.hostPtr()[0];

}

void cullUnobservable_(float4 * predVertMap,
                       const int predWidth,
                       const int predHeight,
                       const float4 * obsVertMap,
                       const int obsWidth,
                       const int obsHeight,
                       const cudaStream_t stream) {

    dim3 block(8,8,1);
    dim3 grid( ceil( predWidth / (float)block.x), ceil(predHeight / (float)block.y ));

    gpu_cullUnobservable<<<grid,block,0,stream>>>(predVertMap,predWidth,predHeight,
                                                  obsVertMap,obsWidth,obsHeight);

}

}
