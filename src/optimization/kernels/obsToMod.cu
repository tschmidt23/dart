#include "obsToMod.h"

#include "kernel_common.h"
#include "util/mirrored_memory.h"

namespace dart {

static const LossFunctionType lossFunction = HuberLoss;

// -=-=-=-=-=-=-=-=-=- kernels -=-=-=-=-=-=-=-=-=-

template <bool dbgDA, bool dbgErr, bool dbgNorm>
__global__ void gpu_errorAndDataAssociationObsToMod(const float4 * obsVertMap,
                                                    const float4 * obsNormMap,
                                                    const int width,
                                                    const int height,
                                                    const SE3 T_mc,
                                                    const SE3 * T_fms,
                                                    const int * sdfFrames,
                                                    const Grid3D<float> * sdfs,
                                                    const int nSdfs,
                                                    const float distanceThreshold,
                                                    const float normThreshold,
                                                    const float planeOffset,
                                                    const float3 planeNormal,
                                                    int * lastElement,
                                                    DataAssociatedPoint * pts,
                                                    int * debugDataAssociation,
                                                    float * debugError,
                                                    float4 * debugNorm) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const int index = x + y*width;

        if (dbgDA) { debugDataAssociation[index] = -1; }
        if (dbgErr) { debugError[index] = NAN; }
        if (dbgNorm) { debugNorm[index] = make_float4(0); }

        const float4 & xObs_c = obsVertMap[index];
        if (xObs_c.w > 0) {

            const float4 xObs_m = T_mc*xObs_c;

            if (dot(make_float3(xObs_m),planeNormal) >= planeOffset) {

                // calculate distance
                float sdfError = 1e20;
                int grid = -1;

                for (int g=0; g < nSdfs; ++g) {

                    const int f = sdfFrames[g];
                    const float4 xObs_f = T_fms[f]*xObs_m;
                    const Grid3D<float> & sdf = sdfs[g];

                    const float3 xObs_g = sdf.getGridCoords(make_float3(xObs_f));

                    if (!sdf.isInBoundsGradientInterp(xObs_g)) {
                        continue;
                    }

                    const float d = (sdf.getValueInterpolated(xObs_g))*sdf.resolution;

                    //if (fabs(d) < fabs(sdf_error)) {
                    if (d < sdfError) {
                        sdfError = d;
                        grid = g;
                    }
                }

                // skip unassociated points and points beyond the distance threshold
                if (sdfError*sdfError > distanceThreshold*distanceThreshold) { }
                else {

                    const int f = sdfFrames[grid];
                    const float4 xObs_f = T_fms[f]*xObs_m;
                    const Grid3D<float> & sdf = sdfs[grid];
                    const float3 xObs_g = sdf.getGridCoords(make_float3(xObs_f));

                    // TODO: figure out what's going on with the -1
                    const float4 nPred =  -1*(SE3Invert( T_fms[f]*T_mc )*normalize(make_float4(sdf.getGradientInterpolated(xObs_g),0)));

                    if (dbgNorm) { debugNorm[index] = nPred; }

                    float4 v = obsNormMap[index];
                    float3 nObs = make_float3(0,0,0);
                    if (v.w > 0.0) {
                        v.w = 0;
                        nObs = make_float3(v);
                        if (dot(nPred,v) < normThreshold ) {
                            return;
                        }
                    }

                    if (dbgDA) { debugDataAssociation[index] = grid; }
                    if (dbgErr) { debugError[index] = sdfError; }

                    int myElement = atomicAdd(lastElement,1);
                    DataAssociatedPoint dt;
                    dt.index = index;
                    dt.dataAssociation = grid;
                    dt.error = sdfError;
                    pts[myElement] = dt;


                }
            }
        }
    }

}

template <bool dbgDA, bool dbgErr, bool dbgNorm>
__global__ void gpu_errorAndDataAssociationObsToModMultiModel(const float4 * obsVertMap,
                                                              const float4 * obsNormMap,
                                                              const int width,
                                                              const int height,
                                                              const int nModels,
                                                              const SE3 * T_mcs,
                                                              const SE3 * const * T_fms,
                                                              const int * const * sdfFrames,
                                                              const Grid3D<float> * const * sdfs,
                                                              const int * nSdfs,
                                                              const float * distanceThresholds,
                                                              const float * normThresholds,
                                                              const float * planeOffsets,
                                                              const float3 * planeNormals,
                                                              int * lastElement,
                                                              DataAssociatedPoint * * pts,
                                                              int * debugDataAssociation,
                                                              float * debugError,
                                                              float4 * debugNorm) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;

    if (dbgDA) { debugDataAssociation[index] = -1; }
    if (dbgErr) { debugError[index] = NAN; }
    if (dbgNorm) { debugNorm[index] = make_float4(0); }

    const float4 & xObs_c = obsVertMap[index];
    if (xObs_c.w > 0) {

        float sdfError = 1e20;
        int associatedModel = -1;
        int associatedGrid = -1;

        for (int m=0; m<nModels; ++m) {

            const float4 xObs_m = T_mcs[m]*xObs_c;
            const float & planeOffset = planeOffsets[m];
            const float3 & planeNormal = planeNormals[m];
            if (dot(make_float3(xObs_m),planeNormal) >= planeOffset) {

                const int mNSdfs = nSdfs[m];
                const int * mSdfFrames = sdfFrames[m];
                const SE3 * mT_fms = T_fms[m];
                const Grid3D<float> * mSdfs = sdfs[m];

                for (int g=0; g<mNSdfs; ++g) {

                    const int f = mSdfFrames[g];
                    const float4 xObs_f = mT_fms[f]*xObs_m;
                    const Grid3D<float> & sdf = mSdfs[g];

                    //printf("model %d sdf %d is in frame %d\n",m,g,f);

                    //printf("%f ",sdf.resolution);

                    const float3 xObs_g = sdf.getGridCoords(make_float3(xObs_f));

                    if (!sdf.isInBoundsGradientInterp(xObs_g)) {
                        continue;
                    }

                    const float d = (sdf.getValueInterpolated(xObs_g))*sdf.resolution;

                    //printf("%f ",d);

                    // if (fabs(d) < fabs(sdfError) {
                    if (d < sdfError) {
                        //printf(".");

                        if (d*d < distanceThresholds[m]*distanceThresholds[m]) {

                            //printf("*");
                            sdfError = d;
                            associatedGrid = g;
                            associatedModel = m;
                        }
                    }

                }

            }
        }

        if (associatedModel != -1) {

//            const int f = sdfFrames[associatedModel][associatedGrid];
//            const float4 xObs_m = T_mcs[associatedModel]*xObs_c;
//            const float4 xObs_f = T_fms[associatedModel][f]*xObs_m;
//            const Grid3D<float> &sdf = sdfs[associatedModel][associatedGrid];
//            const float3 xObs_g = sdf.getGridCoords(make_float3(xObs_f));

//            const float4 nPred = 1*(SE3Invert( T_fms[associatedModel][f]*T_mcs[associatedModel] )*normalize(make_float4(sdf.getGradientInterpolated(xObs_g),0)));

//            float4 v = obsNormMap[index];
//            float3 nObs = make_float3(0,0,0);
//            if (v.w > 0.0) {
//                v.w = 0;
//                nObs = make_float3(v);
//                if (dot(nPred,v) >= normThresholds[associatedModel]) {

                    if (dbgDA) { debugDataAssociation[index] = ((associatedModel << 16) | associatedGrid); }
                    if (dbgErr) { debugError[index] = sdfError; }
                    if (dbgNorm) { debugNorm[index] = obsNormMap[index]; }

                    int myElement = atomicAdd(&lastElement[associatedModel],1);
                    DataAssociatedPoint * mPts = pts[associatedModel];

                    DataAssociatedPoint dt;
                    dt.index = index;
                    dt.dataAssociation = associatedGrid;
                    dt.error = sdfError;
                    mPts[myElement] = dt;

//                }
//            }

        }
    }


}

template <bool dbgJs>
__global__ void gpu_normEqnsObsToMod(const int dims,
                                     const DataAssociatedPoint * pts,
                                     const float4 * obsVertMap,
                                     const int nPoints,
                                     const SE3 T_mc,
                                     const SE3 * T_fms,
                                     const SE3 * T_mfs,
                                     const int * sdfFrames,
                                     const Grid3D<float> * sdfs,
                                     const int * dependencies,
                                     const JointType * jointTypes,
                                     const float3 * jointAxes,
                                     const float huberDelta,
                                     float * result,
                                     float4 * debugJs) {

    extern __shared__ float s[];

    int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index >= nPoints) {
        return;
    }

    if (dbgJs) { debugJs[index] = make_float4(0); }

    const float4 xObs_m = T_mc*obsVertMap[pts[index].index];

    // array declarations
    float * J = &s[threadIdx.x*dims];

    int obsFrame = sdfFrames[pts[index].dataAssociation];

    const float4 xObs_f = T_fms[obsFrame]*xObs_m;

    // compute SDF gradient
    const int g = pts[index].dataAssociation;
    const Grid3D<float> & sdf = sdfs[g];

    const float3 xObs_g = sdf.getGridCoords(make_float3(xObs_f));
    const float3 sdfGrad_f = sdf.getGradientInterpolated(xObs_g);
    const float3 sdfGrad_m = SE3Rotate(T_mfs[obsFrame],sdfGrad_f);

    getErrorJacobianOfModelPoint(J,xObs_m,obsFrame,sdfGrad_m,dims,dependencies,jointTypes,jointAxes,T_fms,T_mfs);

    if (dbgJs) {
        debugJs[index*dims + 0] = make_float4(1,0,0,1);
        debugJs[index*dims + 1] = make_float4(0,1,0,1);
        debugJs[index*dims + 2] = make_float4(0,0,1,1);
        debugJs[index*dims + 3] = make_float4(        0,-xObs_m.z, xObs_m.y,1);
        debugJs[index*dims + 4] = make_float4( xObs_m.z,        0,-xObs_m.x,1);
        debugJs[index*dims + 5] = make_float4(-xObs_m.y, xObs_m.x,        0,1);
    }

    const float residual = pts[index].error;

    float * JTr = result;
    float * JTJ = &result[dims];
    float * e = &result[dims + JTJSize(dims)];

    switch(lossFunction) {
    case SquaredLoss:
    {
        computeSquaredLossResult(dims,-residual,J,e,JTr,JTJ); // TODO: why negative again?
    }
        break;
    case HuberLoss:
    {
        if (fabs(pts[index].error) < huberDelta ) {
            computeSquaredLossResult(dims,-residual,J,e,JTr,JTJ); // TODO: why negative again?
        }
        else {
            float v = huberDelta;
            if (pts[index].error < 0) {
                v = -v;
            }
            for (int i=0; i<dims; i++) {
                if( J[i]==0.0f)   continue;
                atomicAdd(&JTr[i],v*-J[i]);
                for (int j=0; j<=i; j++) {
                    float v2 = J[i]*J[j];
                    atomicAdd(&JTJ[((i*(i+1))>>1) + j],v2);
                }
            }
            atomicAdd(e,huberDelta * (fabs(pts[index].error) - 0.5*huberDelta));
        }
    }
        break;
    }

}

__global__ void gpu_normEqnsObsToModReduced(const int fullDims,
                                            const int redDims,
                                            const DataAssociatedPoint * pts,
                                            const float4 * obsVertMap,
                                            const int nPoints,
                                            const SE3 T_mc,
                                            const SE3 * T_fms,
                                            const SE3 * T_mfs,
                                            const int * sdfFrames,
                                            const Grid3D<float> * sdfs,
                                            const int * dependencies,
                                            const JointType * jointTypes,
                                            const float3 * jointAxes,
                                            const float huberDelta,
                                            const float * dtheta_dalpha,
                                            float * result) {

    extern __shared__ float s[];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int tid = threadIdx.x + blockDim.x*threadIdx.y;

    if (index >= nPoints) {
        return;
    }

    const float4 xObs_m = T_mc*obsVertMap[pts[index].index];

    // array declarations
    float * de_dtheta = &s[tid*(fullDims+redDims)];
    float * J = &s[tid*(fullDims+redDims) + fullDims];

    int obsFrame = sdfFrames[pts[index].dataAssociation];

    const float4 xObs_f = T_fms[obsFrame]*xObs_m;

    // compute SDF gradient
    const int g = pts[index].dataAssociation;
    const Grid3D<float> & sdf = sdfs[g];

    const float3 xObs_g = sdf.getGridCoords(make_float3(xObs_f));
    const float3 sdfGrad_f = sdf.getGradientInterpolated(xObs_g);
    const float3 sdfGrad_m = make_float3(SE3Rotate(T_mfs[obsFrame],make_float4(sdfGrad_f.x,sdfGrad_f.y,sdfGrad_f.z,0.0)));

    getErrorJacobianOfModelPoint(de_dtheta,xObs_m,obsFrame,sdfGrad_m,fullDims,dependencies,jointTypes,jointAxes,T_fms,T_mfs);

    doPoseGradientReduction(J,de_dtheta,dtheta_dalpha,fullDims,redDims);

    const float residual = pts[index].error;

    float * JTr = result;
    float * JTJ = &result[redDims];
    float * e = &result[redDims + JTJSize(redDims)];

    switch(lossFunction) {
    case SquaredLoss:
    {
        computeSquaredLossResult(redDims,-residual,J,e,JTr,JTJ);
    }
        break;
    case HuberLoss:
    {
        if (fabs(pts[index].error) < huberDelta ) {
            computeSquaredLossResult(redDims,-residual,J,e,JTr,JTJ);
        }
        else {
            float v = huberDelta;
            if (pts[index].error < 0) {
                v = -v;
            }
            for (int i=0; i<redDims; i++) {
                if( J[i]==0.0f)   continue;
                atomicAdd(&JTr[i],v*-J[i]);

            }
            atomicAdd(e,huberDelta * (fabs(pts[index].error) - 0.5*huberDelta));
        }
    }
        break;
    }
}

__global__ void gpu_normEqnsObsToModParamMap(const int fullDims,
                                             const int redDims,
                                             const DataAssociatedPoint * pts,
                                             const float4 * obsVertMap,
                                             const int nPoints,
                                             const SE3 T_mc,
                                             const SE3 * T_fms,
                                             const SE3 * T_mfs,
                                             const int * sdfFrames,
                                             const Grid3D<float> * sdfs,
                                             const int * dependencies,
                                             const JointType * jointTypes,
                                             const float3 * jointAxes,
                                             const float huberDelta,
                                             const int * dMapping,
                                             float * result) {

    extern __shared__ float s[];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int tid = threadIdx.x + blockDim.x*threadIdx.y;

    if (index >= nPoints) {
        return;
    }

    const float4 xObs_m = T_mc*obsVertMap[pts[index].index];

    // array declarations
    float * de_dtheta = &s[tid*(fullDims+redDims)];
    float * J = &s[tid*(fullDims+redDims) + fullDims];

    int obsFrame = sdfFrames[pts[index].dataAssociation];

    const float4 xObs_f = T_fms[obsFrame]*xObs_m;

    // compute SDF gradient
    const int g = pts[index].dataAssociation;
    const Grid3D<float> & sdf = sdfs[g];

    const float3 xObs_g = sdf.getGridCoords(make_float3(xObs_f));
    const float3 sdfGrad_f = sdf.getGradientInterpolated(xObs_g);
    const float3 sdfGrad_m = make_float3(SE3Rotate(T_mfs[obsFrame],make_float4(sdfGrad_f.x,sdfGrad_f.y,sdfGrad_f.z,0.0)));

    getErrorJacobianOfModelPoint(de_dtheta,xObs_m,obsFrame,sdfGrad_m,fullDims,dependencies,jointTypes,jointAxes,T_fms,T_mfs);

    doParamMapping(J,de_dtheta,dMapping,fullDims,redDims);

    const float residual = pts[index].error;

    float * JTr = result;
    float * JTJ = &result[redDims];
    float * e = &result[redDims + JTJSize(redDims)];

    switch(lossFunction) {
    case SquaredLoss:
    {
        computeSquaredLossResult(redDims,-residual,J,e,JTr,JTJ);
    }
        break;
    case HuberLoss:
    {
        if (fabs(pts[index].error) < huberDelta ) {
            computeSquaredLossResult(redDims,-residual,J,e,JTr,JTJ);
        }
        else {
            float v = huberDelta;
            if (pts[index].error < 0) {
                v = -v;
            }
            for (int i=0; i<redDims; i++) {
                if( J[i]==0.0f)   continue;
                atomicAdd(&JTr[i],v*-J[i]);

            }
            atomicAdd(e,huberDelta * (fabs(pts[index].error) - 0.5*huberDelta));
        }
    }
        break;
    }
}


void errorAndDataAssociation(const float4 * dObsVertMap,
                             const float4 * dObsNormMap,
                             const int width,
                             const int height,
                             const MirroredModel & model,
                             const OptimizationOptions & opts,
                             DataAssociatedPoint * dPts,
                             int * dLastElement,
                             int * hLastElement,
                             int * debugDataAssociation,
                             float * debugError,
                             float4 * debugNorm) {

    dim3 block(16,8);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    cudaMemset(dLastElement,0,sizeof(int));

    if (debugDataAssociation == 0) {
        if (debugError == 0) {
            if (debugNorm == 0) {
                gpu_errorAndDataAssociationObsToMod<false,false,false><<<grid,block>>>(dObsVertMap, dObsNormMap, width, height, model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceSdfFrames(), model.getDeviceSdfs(), model.getNumSdfs(), opts.distThreshold[0], opts.normThreshold, opts.planeOffset[0], opts.planeNormal[0], dLastElement, dPts, debugDataAssociation, debugError, debugNorm);
            } else {
                gpu_errorAndDataAssociationObsToMod<false,false,true><<<grid,block>>>(dObsVertMap, dObsNormMap, width, height, model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceSdfFrames(), model.getDeviceSdfs(), model.getNumSdfs(), opts.distThreshold[0], opts.normThreshold, opts.planeOffset[0], opts.planeNormal[0], dLastElement, dPts, debugDataAssociation, debugError, debugNorm);
            }
        } else {
            if (debugNorm == 0) {
                gpu_errorAndDataAssociationObsToMod<false,true,false><<<grid,block>>>(dObsVertMap, dObsNormMap, width, height, model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceSdfFrames(), model.getDeviceSdfs(), model.getNumSdfs(), opts.distThreshold[0], opts.normThreshold, opts.planeOffset[0], opts.planeNormal[0], dLastElement, dPts, debugDataAssociation, debugError, debugNorm);
            } else {
                gpu_errorAndDataAssociationObsToMod<false,true,true><<<grid,block>>>(dObsVertMap, dObsNormMap, width, height, model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceSdfFrames(), model.getDeviceSdfs(), model.getNumSdfs(), opts.distThreshold[0], opts.normThreshold, opts.planeOffset[0], opts.planeNormal[0], dLastElement, dPts, debugDataAssociation, debugError, debugNorm);
            }
        }
    }
    else {
        if (debugError == 0) {
            if (debugNorm == 0) {
                gpu_errorAndDataAssociationObsToMod<true,false,false><<<grid,block>>>(dObsVertMap, dObsNormMap, width, height, model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceSdfFrames(), model.getDeviceSdfs(), model.getNumSdfs(), opts.distThreshold[0], opts.normThreshold, opts.planeOffset[0], opts.planeNormal[0], dLastElement, dPts, debugDataAssociation, debugError, debugNorm);
            } else {
                gpu_errorAndDataAssociationObsToMod<true,false,true><<<grid,block>>>(dObsVertMap, dObsNormMap, width, height, model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceSdfFrames(), model.getDeviceSdfs(), model.getNumSdfs(), opts.distThreshold[0], opts.normThreshold, opts.planeOffset[0], opts.planeNormal[0], dLastElement, dPts, debugDataAssociation, debugError, debugNorm);
            }
        } else {
            if (debugNorm == 0) {
                gpu_errorAndDataAssociationObsToMod<true,true,false><<<grid,block>>>(dObsVertMap, dObsNormMap, width, height, model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceSdfFrames(), model.getDeviceSdfs(), model.getNumSdfs(), opts.distThreshold[0], opts.normThreshold, opts.planeOffset[0], opts.planeNormal[0], dLastElement, dPts, debugDataAssociation, debugError, debugNorm);
            } else {
                gpu_errorAndDataAssociationObsToMod<true,true,true><<<grid,block>>>(dObsVertMap, dObsNormMap, width, height, model.getTransformCameraToModel(), model.getDeviceTransformsModelToFrame(), model.getDeviceSdfFrames(), model.getDeviceSdfs(), model.getNumSdfs(), opts.distThreshold[0], opts.normThreshold, opts.planeOffset[0], opts.planeNormal[0], dLastElement, dPts, debugDataAssociation, debugError, debugNorm);
            }
        }
    }

    cudaMemcpy(hLastElement,dLastElement,sizeof(int),cudaMemcpyDeviceToHost);

}

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
                                       cudaStream_t stream) {

    cudaMemset(lastElements,0,nModels*sizeof(int));

    dim3 block;
    if (height == 1) {
        block.x = 128; block.y = block.z = 1;
    }
    else {
        block.x = 16; block.y = 8; block.z = 1;
    }
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    if (dDebugDataAssociation == 0) {
        if (dDebugError == 0) {
            if (dDebugNorm == 0) {
                gpu_errorAndDataAssociationObsToModMultiModel<false,false,false><<<grid,block,0,stream>>>(dObsVertMap, dObsNormMap, width, height, nModels, T_mcs, T_fms, sdfFrames, sdfs, nSdfs,distanceThresholds, normalThresholds, planeOffsets, planeNormals, lastElements, pts, dDebugDataAssociation, dDebugError, dDebugNorm);
            } else {
                gpu_errorAndDataAssociationObsToModMultiModel<false,false,true><<<grid,block,0,stream>>>(dObsVertMap, dObsNormMap, width, height, nModels, T_mcs, T_fms, sdfFrames, sdfs, nSdfs,distanceThresholds, normalThresholds, planeOffsets, planeNormals, lastElements, pts, dDebugDataAssociation, dDebugError, dDebugNorm);
            }
        } else {
            if (dDebugNorm == 0) {
                gpu_errorAndDataAssociationObsToModMultiModel<false,true,false><<<grid,block,0,stream>>>(dObsVertMap, dObsNormMap, width, height, nModels, T_mcs, T_fms, sdfFrames, sdfs, nSdfs,distanceThresholds, normalThresholds, planeOffsets, planeNormals, lastElements, pts, dDebugDataAssociation, dDebugError, dDebugNorm);
            } else {
                gpu_errorAndDataAssociationObsToModMultiModel<false,true,true><<<grid,block,0,stream>>>(dObsVertMap, dObsNormMap, width, height, nModels, T_mcs, T_fms, sdfFrames, sdfs, nSdfs,distanceThresholds, normalThresholds, planeOffsets, planeNormals, lastElements, pts, dDebugDataAssociation, dDebugError, dDebugNorm);
            }
        }
    } else {
        if (dDebugError == 0) {
            if (dDebugNorm == 0) {
                gpu_errorAndDataAssociationObsToModMultiModel<true,false,false><<<grid,block,0,stream>>>(dObsVertMap, dObsNormMap, width, height, nModels, T_mcs, T_fms, sdfFrames, sdfs, nSdfs,distanceThresholds, normalThresholds, planeOffsets, planeNormals, lastElements, pts, dDebugDataAssociation, dDebugError, dDebugNorm);
            } else {
                gpu_errorAndDataAssociationObsToModMultiModel<true,false,true><<<grid,block,0,stream>>>(dObsVertMap, dObsNormMap, width, height, nModels, T_mcs, T_fms, sdfFrames, sdfs, nSdfs,distanceThresholds, normalThresholds, planeOffsets, planeNormals, lastElements, pts, dDebugDataAssociation, dDebugError, dDebugNorm);
            }
        } else {
            if (dDebugNorm == 0) {
                gpu_errorAndDataAssociationObsToModMultiModel<true,true,false><<<grid,block,0,stream>>>(dObsVertMap, dObsNormMap, width, height, nModels, T_mcs, T_fms, sdfFrames, sdfs, nSdfs,distanceThresholds, normalThresholds, planeOffsets, planeNormals, lastElements, pts, dDebugDataAssociation, dDebugError, dDebugNorm);
            } else {
                gpu_errorAndDataAssociationObsToModMultiModel<true,true,true><<<grid,block,0,stream>>>(dObsVertMap, dObsNormMap, width, height, nModels, T_mcs, T_fms, sdfFrames, sdfs, nSdfs,distanceThresholds, normalThresholds, planeOffsets, planeNormals, lastElements, pts, dDebugDataAssociation, dDebugError, dDebugNorm);
            }
        }
    }

}

void normEqnsObsToMod(const int dims,
                      const float4 * dObsVertMap,
                      const int width,
                      const int height,
                      const MirroredModel & model,
                      const OptimizationOptions & opts,
                      DataAssociatedPoint * dPts,
                      int nElements,
                      float * dResult,
                      float4 * debugJs) {

    std::cout << nElements << " points associated to model " << model.getModelID() << std::endl;

    dim3 block;
    if (height == 1) {
        block.x = 128; block.y = block.z = 1;
    }
    else {
        block.x = 16; block.y = 8; block.z = 1;
    }
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    // initilize system to zero
    cudaMemset(dResult,0,(dims + JTJSize(dims) + 1)*sizeof(float));

    if (nElements > 10) {

        block = dim3(64,1,1);
        grid = dim3(ceil((double)nElements/64),1,1);

        {

            if (debugJs == 0) {
                gpu_normEqnsObsToMod<false><<<grid,block,64*dims*sizeof(float)>>>(dims,
                                                                                  dPts,
                                                                                  dObsVertMap,
                                                                                  nElements,
                                                                                  model.getTransformCameraToModel(),
                                                                                  model.getDeviceTransformsModelToFrame(),
                                                                                  model.getDeviceTransformsFrameToModel(),
                                                                                  model.getDeviceSdfFrames(),
                                                                                  model.getDeviceSdfs(),
                                                                                  model.getDeviceDependencies(),
                                                                                  model.getDeviceJointTypes(),
                                                                                  model.getDeviceJointAxes(),
                                                                                  opts.huberDelta,
                                                                                  dResult,
                                                                                  debugJs);
            } else {
                gpu_normEqnsObsToMod<true><<<grid,block,64*dims*sizeof(float)>>>(dims,
                                                                                 dPts,
                                                                                 dObsVertMap,
                                                                                 nElements,
                                                                                 model.getTransformCameraToModel(),
                                                                                 model.getDeviceTransformsModelToFrame(),
                                                                                 model.getDeviceTransformsFrameToModel(),
                                                                                 model.getDeviceSdfFrames(),
                                                                                 model.getDeviceSdfs(),
                                                                                 model.getDeviceDependencies(),
                                                                                 model.getDeviceJointTypes(),
                                                                                 model.getDeviceJointAxes(),
                                                                                 opts.huberDelta,
                                                                                 dResult,
                                                                                 debugJs);
            }

#ifdef CUDA_ERR_CHECK
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("gpu_normEqnsObsToMod error: %s\n",cudaGetErrorString(err));
            }
#endif
        }

    }

}

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
                             float * dResult) {

    dim3 block;
    if (height == 1) {
        block.x = 128; block.y = block.z = 1;
    }
    else {
        block.x = 16; block.y = 8; block.z = 1;
    }
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    // initilize system to zero
    cudaMemset(dResult,0,(reductionDims + JTJSize(reductionDims) + 1)*sizeof(float));

    if (nElements > 10) {

        block = dim3(64,1,1);
        grid = dim3(ceil((double)nElements/64),1,1);

        {

            gpu_normEqnsObsToModReduced<<<grid,block,64*(dims+reductionDims)*sizeof(float)>>>(dims,
                                                                                              reductionDims,
                                                                                              dPts,
                                                                                              dObsVertMap,
                                                                                              nElements,
                                                                                              model.getTransformCameraToModel(),
                                                                                              model.getDeviceTransformsModelToFrame(),
                                                                                              model.getDeviceTransformsFrameToModel(),
                                                                                              model.getDeviceSdfFrames(),
                                                                                              model.getDeviceSdfs(),
                                                                                              model.getDeviceDependencies(),
                                                                                              model.getDeviceJointTypes(),
                                                                                              model.getDeviceJointAxes(),
                                                                                              opts.huberDelta,
                                                                                              d_dtheta_dalpha,
                                                                                              dResult);

#ifdef CUDA_ERR_CHECK
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("gpu_normEqnsObsToModReduced error: %s\n",cudaGetErrorString(err));
            }
#endif
        }

    }

}

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
                             float * dResult) {

    dim3 block;
    if (height == 1) {
        block.x = 128; block.y = block.z = 1;
    }
    else {
        block.x = 16; block.y = 8; block.z = 1;
    }
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    // initilize system to zero
    cudaMemset(dResult,0,(reductionDims + JTJSize(reductionDims) + 1)*sizeof(float));

    if (nElements > 10) {

        block = dim3(64,1,1);
        grid = dim3(ceil((double)nElements/64),1,1);

        {

            gpu_normEqnsObsToModParamMap<<<grid,block,64*(dims+reductionDims)*sizeof(float)>>>(dims,
                                                                                              reductionDims,
                                                                                              dPts,
                                                                                              dObsVertMap,
                                                                                              nElements,
                                                                                              model.getTransformCameraToModel(),
                                                                                              model.getDeviceTransformsModelToFrame(),
                                                                                              model.getDeviceTransformsFrameToModel(),
                                                                                              model.getDeviceSdfFrames(),
                                                                                              model.getDeviceSdfs(),
                                                                                              model.getDeviceDependencies(),
                                                                                              model.getDeviceJointTypes(),
                                                                                              model.getDeviceJointAxes(),
                                                                                              opts.huberDelta,
                                                                                              dMapping,
                                                                                              dResult);

#ifdef CUDA_ERR_CHECK
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("gpu_normEqnsObsToModReduced error: %s\n",cudaGetErrorString(err));
            }
#endif
        }

    }

}


}
