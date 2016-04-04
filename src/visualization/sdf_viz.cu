#include "sdf_viz.h"

namespace dart {

inline __host__ __device__ unsigned char clamp(int c) {
    return min(max(0,c),255);
}

// h: 0-360
// s: 0 - 1
// v: 0 - 1
inline __host__ __device__ uchar3 hsv2rgb(float h, float s, float v) {
    float c = v*s;
    float hPrime = h/60.0f;
    float x = c*(1 - fabs(fmodf(hPrime,2) - 1));
    float m = v-c;
    int hPrimeInt = hPrime;
    switch (hPrimeInt) {
    case 0:
        return make_uchar3(255*(c+m),255*(x+m),255*(m));
    case 1:
        return make_uchar3(255*(x+m),255*(c+m),255*(m));
    case 2:
        return make_uchar3(255*(m),255*(c+m),255*(x+m));
    case 3:
        return make_uchar3(255*(m),255*(x+m),255*(c+m));
    case 4:
        return make_uchar3(255*(x+m),255*(m),255*(c+m));
    case 5:
        return make_uchar3(255*(c+m),255*(m),255*(x+m));
    }
    return make_uchar3(0,0,0);
}

// -=-=-=-=-=-=-=-=-=- kernels -=-=-=-=-=-=-=-=-=-
template <ColorRamp R>
__global__ void gpu_visualizeModelSdfPlane(uchar3 * img,
                                           const int width,
                                           const int height,
                                           const float2 origin,
                                           const float2 size,
                                           const SE3 T_mc,
                                           const SE3 * T_fms,
                                           const int * sdfFrames,
                                           const Grid3D<float> * sdfs,
                                           const int nSdfs,
                                           const float planeDepth,
                                           const float minVal,
                                           const float maxVal) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uchar3 & imgVal = img[x + y*width];

    float4 pc = make_float4(origin.x + x/(float)width*size.x,
                            origin.y + y/(float)height*size.y,
                            planeDepth,1.0f);

    float4 pm = T_mc * pc;

    float minSdfVal = 1e20;

    for (int s=0; s<nSdfs; ++s) {

        const int f = sdfFrames[s];
        const float4 pf = T_fms[f]*pm;
        const Grid3D<float> &sdf = sdfs[s];

        float3 pSdf = sdf.getGridCoords(make_float3(pf.x,pf.y,pf.z));
        if (!sdf.isInBoundsInterp(pSdf)) {
            continue;
        }

        const float sdfVal = sdf.getValueInterpolated(pSdf);
        minSdfVal = min(minSdfVal,sdfVal);

    }

    if (minSdfVal > 1e19) {
        imgVal = make_uchar3(0,0,0);
    }
    else {

//        printf("%f\n",minSdfVal);

        float normVal = (minSdfVal-minVal)/(maxVal - minVal);

        switch (R) {
        case ColorRampGrayscale:
            imgVal = make_uchar3(clamp(255*normVal),clamp(255*normVal),clamp(255*normVal));
            break;
        case ColorRampHeatMap:
//            if (normVal < 0.25) { imgVal = make_uchar3(0,clamp(255*(normVal/0.25)),255); }
//            else if (normVal < 0.5) { imgVal = make_uchar3(0,255,clamp(255*((0.5-normVal)/0.25))); }
//            else if (normVal < 0.75) { imgVal = make_uchar3(clamp(255*((normVal - 0.5)/0.25)),255,0); }
//            else { imgVal = make_uchar3(255,clamp(255*(1.0-normVal)/0.25),0); }
            imgVal = hsv2rgb(240*max(0.0f,min(1.0f-normVal,1.0f)),0.5,0.8);
//            imgVal.x = clamp(imgVal.x);
//            imgVal.y = clamp(imgVal.y);
//            imgVal.z = clamp(imgVal.z);
            break;
        case ColorRampRedGreen:
//            if (normVal < 0.5) {
//                imgVal = make_uchar3(clamp(200*(1-2*normVal)),
//                                     clamp(30*(1-2*normVal)),
//                                     clamp(30*(1-2*normVal)));
//            }
//            else {
//                imgVal = make_uchar3(clamp(30*(2*normVal - 1)),
//                                     clamp(200*(2*normVal - 1)),
//                                     clamp(30*(2*normVal - 1)));
//            }
//            if (normVal < 0.2) {
//                imgVal = make_uchar3(255, //0,0);
//                                     clamp(255*normVal/0.2),
//                                     clamp(255*normVal/0.2));
//            }
//            else {
//                imgVal = make_uchar3(clamp(255*(1.0-normVal)/0.8),
//                                     255,
//                                     clamp(255*(1.0-normVal/0.8)));
//            }
            if (normVal < 0.2) {
                imgVal = make_uchar3(clamp(255*(0.2-normVal)/0.2),0,0);
            }
            else {
                imgVal = make_uchar3(0,clamp(255*(normVal-0.2)/0.8),0);
            }
            break;
        }
    }

}


template <ColorRamp R>
__global__ void gpu_visualizeModelSdfPlaneProjective(uchar3 * img,
                                                const int width,
                                                const int height,
                                                const SE3 T_mc,
                                                const SE3 * T_fms,
                                                const int * sdfFrames,
                                                const Grid3D<float> * sdfs,
                                                const int nSdfs,
                                                const float planeDepth,
                                                const float focalLength,
                                                const float minVal,
                                                const float maxVal) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uchar3 & imgVal = img[x + y*width];

    float4 pc = make_float4((x-width/2)*planeDepth/focalLength,
                           (y-height/2)*planeDepth/focalLength,
                           planeDepth,1.0f);

    float4 pm = T_mc * pc;

    float minSdfVal = 1e20;

    for (int s=0; s<nSdfs; ++s) {

        const int f = sdfFrames[s];
        const float4 pf = T_fms[f]*pm;
        const Grid3D<float> &sdf = sdfs[s];

        float3 pSdf = sdf.getGridCoords(make_float3(pf.x,pf.y,pf.z));
        if (!sdf.isInBoundsInterp(pSdf)) {
            continue;
        }

        const float sdfVal = sdf.getValueInterpolated(pSdf);
        minSdfVal = min(minSdfVal,sdfVal);

    }

    if (minSdfVal == 1e20) {
        imgVal = make_uchar3(0,0,0);
    }
    else {
        float normVal = (minSdfVal-minVal)/(maxVal - minVal);

        switch (R) {
        case ColorRampGrayscale:
            imgVal = make_uchar3(clamp(255*normVal),clamp(255*normVal),clamp(255*normVal));
            break;
        case ColorRampHeatMap:
            if (normVal < 0.25) { imgVal = make_uchar3(0,clamp(255*(normVal/0.25)),255); }
            else if (normVal < 0.5) { imgVal = make_uchar3(0,255,clamp(255*((0.5-normVal)/0.25))); }
            else if (normVal < 0.75) { imgVal = make_uchar3(clamp(255*((normVal - 0.5)/0.25)),255,0); }
            else { imgVal = make_uchar3(255,clamp(255*(1.0-normVal)/0.25),0); }
            break;
        case ColorRampRedGreen:
//            if (normVal < 0.5) {
//                imgVal = make_uchar3(clamp(200*(1-2*normVal)),
//                                     clamp(30*(1-2*normVal)),
//                                     clamp(30*(1-2*normVal)));
//            }
//            else {
//                imgVal = make_uchar3(clamp(30*(2*normVal - 1)),
//                                     clamp(200*(2*normVal - 1)),
//                                     clamp(30*(2*normVal - 1)));
//            }
            if (normVal < 0.2) {
                imgVal = make_uchar3(255, 0,0);
                                     //clamp(255*normVal/0.2),
                                     //clamp(255*normVal/0.2));
            }
            else {
                imgVal = make_uchar3(0,255,0);//make_uchar3(clamp(255*(1.0-normVal)/0.8),
//                                     255,
//                                     clamp(255*(1.0-normVal/0.8));
            }
            break;
        }
    }

}

template <bool firstModel>
__global__ void gpu_getMultiModelSdfSlice(float * sdfVals,
                                          const int width,
                                          const int height,
                                          const float2 origin,
                                          const float2 size,
                                          const SE3 T_mp,
                                          const SE3 * T_fms,
                                          const int * sdfFrames,
                                          const Grid3D<float> * sdfs,
                                          const int nSdfs) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float4 ptp = make_float4(origin.x + (x+0.5)/(float)width*size.x,
                            origin.y + (y+0.5)/(float)height*size.y,
                            0,1.0f);

    float4 ptm = T_mp * ptp;

    float minSdfVal = 1e20;

    for (int s=0; s<nSdfs; ++s) {

        const int f = sdfFrames[s];
        const float4 pf = T_fms[f]*ptm;
        const Grid3D<float> & sdf = sdfs[s];

        float3 pSdf = sdf.getGridCoords(make_float3(pf.x,pf.y,pf.z));
        if (!sdf.isInBoundsInterp(pSdf)) {
            continue;
        }

        const float sdfVal = sdf.getValueInterpolated(pSdf)*sdf.resolution;
        minSdfVal = min(minSdfVal,sdfVal);

    }

    if (firstModel || minSdfVal < sdfVals[x + y*width]) {
        sdfVals[x + y*width] = minSdfVal;
    }

}

__global__ void gpu_getModelSdfSlice(float * sdfVals,
                                     const int width,
                                     const int height,
                                     const float2 origin,
                                     const float2 size,
                                     const SE3 T_mp,
                                     const SE3 * T_fms,
                                     const int * sdfFrames,
                                     const Grid3D<float> * sdfs,
                                     const int nSdfs) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float4 ptp = make_float4(origin.x + (x+0.5)/(float)width*size.x,
                            origin.y + (y+0.5)/(float)height*size.y,
                            0,1.0f);

    float4 ptm = T_mp * ptp;

    float minSdfVal = 1e20;

    for (int s=0; s<nSdfs; ++s) {

        const int f = sdfFrames[s];
        const float4 pf = T_fms[f]*ptm;
        const Grid3D<float> & sdf = sdfs[s];

        float3 pSdf = sdf.getGridCoords(make_float3(pf.x,pf.y,pf.z));
        if (!sdf.isInBoundsInterp(pSdf)) {
            continue;
        }

        const float sdfVal = sdf.getValueInterpolated(pSdf)*sdf.resolution;
        minSdfVal = min(minSdfVal,sdfVal);

    }

    sdfVals[x + y*width] = minSdfVal;

}

__global__ void gpu_getSdfSlice(float * sdfVals,
                                 const int width,
                                 const int height,
                                 const float2 origin,
                                 const float2 size,
                                 const SE3 T_sp,
                                 const Grid3D<float> * sdf) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float4 ptp = make_float4(origin.x + (x+0.5f)/(float)width*size.x,
                             origin.y + (y+0.5f)/(float)height*size.y,
                             0,1.0f);

    float4 pts = T_sp * ptp;

    float3 pSdf = sdf->getGridCoords(make_float3(pts.x,pts.y,pts.z));
    if (!sdf->isInBoundsInterp(pSdf)) {
        return;
    }

    const float sdfVal = sdf->getValueInterpolated(pSdf)*sdf->resolution;
    sdfVals[x + y*width] = sdfVal;

}

__global__ void gpu_getObservationSdfPlane(float * sdfVals,
                                           const int width,
                                           const int height,
                                           const Grid3D<float> * sdf,
                                           const float planeDepth) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float3 pSdf = make_float3((x/(float)(width-1))*sdf->dim.x,
                              (y/(float)(height-1))*sdf->dim.y,
                              planeDepth*sdf->dim.z);

    if (sdf->isInBoundsInterp(pSdf)) {
        sdfVals[x + y*width] = sdf->getValueInterpolated(pSdf);
    } else {
        sdfVals[x + y*width] = NAN;
    }

}

__global__ void gpu_getObservationSdfPlaneProjective(float * sdfVals,
                                                     const int width,
                                                     const int height,
                                                     const Grid3D<float> * sdf,
                                                     const float planeDepth,
                                                     const float focalLength) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float3 pc = make_float3((x-width/2)*planeDepth/focalLength,
                           (y-height/2)*planeDepth/focalLength,
                           planeDepth);

    float3 pSdf = sdf->getGridCoords(pc);

    if (sdf->isInBoundsInterp(pSdf)) {
        sdfVals[x + y*width] = sdf->getValueInterpolated(pSdf);
    } else {
        sdfVals[x + y*width] = NAN;
    }

}


__global__ void gpu_getModelSdfPlaneProjective(float * sdf,
                                               const int width,
                                               const int height,
                                               const SE3 T_mc,
                                               const SE3 * T_fms,
                                               const int * sdfFrames,
                                               const Grid3D<float> * sdfs,
                                               const int nSdfs,
                                               const float planeDepth,
                                               const float focalLength) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float4 pc = make_float4((x-width/2)*planeDepth/focalLength,
                           (y-height/2)*planeDepth/focalLength,
                           planeDepth,1.0f);

    float4 pm = T_mc * pc;

    float minSdfVal = 1e20;

    for (int s=0; s<nSdfs; ++s) {

        const int f = sdfFrames[s];
        const float4 pf = T_fms[f]*pm;
        const Grid3D<float> &sdf = sdfs[s];

        float3 pSdf = sdf.getGridCoords(make_float3(pf.x,pf.y,pf.z));
        if (!sdf.isInBoundsInterp(pSdf)) {
            continue;
        }

        const float sdfVal = sdf.getValueInterpolated(pSdf);
        minSdfVal = min(minSdfVal,sdfVal);

    }

    sdf[x + y*width] = minSdfVal;
}

__global__ void gpu_getModelSdfGradientPlaneProjective(float3 * grad,
                                                       const int width,
                                                       const int height,
                                                       const SE3 T_mc,
                                                       const SE3 * T_fms,
                                                       const int * sdfFrames,
                                                       const Grid3D<float> * sdfs,
                                                       const int nSdfs,
                                                       const float planeDepth,
                                                       const float focalLength) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float4 pc = make_float4((x-width/2)*planeDepth/focalLength,
                           (y-height/2)*planeDepth/focalLength,
                           planeDepth,1.0f);

    float4 pm = T_mc * pc;

    float minSdfVal = 1e20;
    int minSdf = -1;

    for (int s=0; s<nSdfs; ++s) {

        const int f = sdfFrames[s];
        const float4 pf = T_fms[f]*pm;
        const Grid3D<float> &sdf = sdfs[s];

        float3 pSdf = sdf.getGridCoords(make_float3(pf.x,pf.y,pf.z));
        if (!sdf.isInBoundsGradientInterp(pSdf)) {
            continue;
        }

        const float sdfVal = sdf.getValueInterpolated(pSdf);
        if (sdfVal < minSdfVal) {
            minSdfVal = sdfVal;
            minSdf = s;
        }

    }

    if (minSdf != -1) {
        const int minF = sdfFrames[minSdf];
        const float4 pMinF = T_fms[minF]*pm;
        const float3 pSdf = sdfs[minSdf].getGridCoords(make_float3(pMinF));

        grad[x + y*width] = sdfs[minSdf].getGradientInterpolated(pSdf);
    }

}

__global__ void gpu_visualizeDataAssociationPlane(uchar3 * img,
                                                  const int width,
                                                  const int height,
                                                  const float2 origin,
                                                  const float2 size,
                                                  const SE3 T_mc,
                                                  const SE3 * T_fms,
                                                  const int * sdfFrames,
                                                  const Grid3D<float> * sdfs,
                                                  const int nSdfs,
                                                  const uchar3 * sdfColors,
                                                  const float planeDepth) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uchar3 & imgVal = img[x + y*width];

    float4 pc = make_float4(origin.x + x/(float)width*size.x,
                            origin.y + y/(float)height*size.y,
                            planeDepth,1.0f);

    float4 pm = T_mc * pc;

    float minSdfVal = 1e20;
    int minS = -1;

    for (int s=0; s<nSdfs; ++s) {

        const int f = sdfFrames[s];
        const float4 pf = T_fms[f]*pm;
        const Grid3D<float> &sdf = sdfs[s];

        float3 pSdf = sdf.getGridCoords(make_float3(pf.x,pf.y,pf.z));
        if (!sdf.isInBoundsInterp(pSdf)) {
            continue;
        }

        const float sdfVal = sdf.getValueInterpolated(pSdf);
        if (sdfVal < minSdfVal) {
            minSdfVal = sdfVal;
            minS = s;
        }

    }

    if (minS == -1) {
        imgVal = make_uchar3(0,0,0);
    }
    else {
        imgVal = sdfColors[minS];
    }

}

__global__ void gpu_visualizeDataAssociationPlaneProjective(uchar3 * img,
                                                            const int width,
                                                            const int height,
                                                            const SE3 T_mc,
                                                            const SE3 * T_fms,
                                                            const int * sdfFrames,
                                                            const Grid3D<float> * sdfs,
                                                            const int nSdfs,
                                                            const uchar3 * sdfColors,
                                                            const float planeDepth,
                                                            const float focalLength) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uchar3 &imgVal = img[x + y*width];

    float4 pc = make_float4((x-width/2)*planeDepth/focalLength,
                           (y-height/2)*planeDepth/focalLength,
                           planeDepth,1.0f);

    float4 pm = T_mc * pc;

    float minSdfVal = 1e20;
    int minS = -1;

    for (int s=0; s<nSdfs; ++s) {

        const int f = sdfFrames[s];
        const float4 pf = T_fms[f]*pm;
        const Grid3D<float> &sdf = sdfs[s];

        float3 pSdf = sdf.getGridCoords(make_float3(pf.x,pf.y,pf.z));
        if (!sdf.isInBoundsInterp(pSdf)) {
            continue;
        }

        const float sdfVal = sdf.getValueInterpolated(pSdf);
        if (sdfVal < minSdfVal) {
            minSdfVal = sdfVal;
            minS = s;
        }

    }

    if (minS == -1) {
        imgVal = make_uchar3(0,0,0);
    }
    else {
        imgVal = sdfColors[minS];
    }

}

__global__ void gpu_visualizeDataAssociationPlaneProjective(uchar4 * img,
                                                            const int width,
                                                            const int height,
                                                            const SE3 T_mc,
                                                            const SE3 * T_fms,
                                                            const int * sdfFrames,
                                                            const Grid3D<float> * sdfs,
                                                            const int nSdfs,
                                                            const uchar3 * sdfColors,
                                                            const float planeDepth,
                                                            const float focalLength) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uchar4 & imgVal = img[x + y*width];

    float4 pc = make_float4((x-width/2)*planeDepth/focalLength,
                           (y-height/2)*planeDepth/focalLength,
                           planeDepth,1.0f);

    float4 pm = T_mc * pc;

    float minSdfVal = 1e20;
    int minS = -1;

    for (int s=0; s<nSdfs; ++s) {

        const int f = sdfFrames[s];
        const float4 pf = T_fms[f]*pm;
        const Grid3D<float> &sdf = sdfs[s];

        float3 pSdf = sdf.getGridCoords(make_float3(pf.x,pf.y,pf.z));
        if (!sdf.isInBoundsInterp(pSdf)) {
            continue;
        }

        const float sdfVal = sdf.getValueInterpolated(pSdf);
        if (sdfVal < minSdfVal) {
            minSdfVal = sdfVal;
            minS = s;
        }

    }

    if (minS == -1) {
        imgVal = make_uchar4(0,0,0,0);
    }
    else {
        imgVal = make_uchar4(sdfColors[minS].x,sdfColors[minS].y,sdfColors[minS].z,255);
    }

}

// -=-=-=-=-=-=-=-=-=- host interface functions -=-=-=-=-=-=-=-=-=-
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
                            const ColorRamp ramp) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    switch(ramp) {
    case ColorRampGrayscale:
        gpu_visualizeModelSdfPlane<ColorRampGrayscale><<<grid,block>>>(img,width,height,origin,size,T_mc,T_fms,sdfFrames,sdfs,nSdfs,planeDepth,minVal,maxVal);
        break;
    case ColorRampHeatMap:
        gpu_visualizeModelSdfPlane<ColorRampHeatMap><<<grid,block>>>(img,width,height,origin,size,T_mc,T_fms,sdfFrames,sdfs,nSdfs,planeDepth,minVal,maxVal);
        break;
    case ColorRampRedGreen:
        gpu_visualizeModelSdfPlane<ColorRampRedGreen><<<grid,block>>>(img,width,height,origin,size,T_mc,T_fms,sdfFrames,sdfs,nSdfs,planeDepth,minVal,maxVal);
    }

}

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
                                      const ColorRamp ramp) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    switch(ramp) {
    case ColorRampGrayscale:
        gpu_visualizeModelSdfPlaneProjective<ColorRampGrayscale><<<grid,block>>>(img,width,height,T_mc,T_fms,sdfFrames,sdfs,nSdfs,planeDepth,focalLength,minVal,maxVal);
        break;
    case ColorRampHeatMap:
        gpu_visualizeModelSdfPlaneProjective<ColorRampHeatMap><<<grid,block>>>(img,width,height,T_mc,T_fms,sdfFrames,sdfs,nSdfs,planeDepth,focalLength,minVal,maxVal);
        break;
    case ColorRampRedGreen:
        gpu_visualizeModelSdfPlaneProjective<ColorRampRedGreen><<<grid,block>>>(img,width,height,T_mc,T_fms,sdfFrames,sdfs,nSdfs,planeDepth,focalLength,minVal,maxVal);
    }

}

void getMultiModelSdfSlice(float * sdfSlice,
                           const int width,
                           const int height,
                           const float2 origin,
                           const float2 size,
                           const std::vector<SE3> & T_pm,
                           const std::vector<MirroredModel*> & models) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_getMultiModelSdfSlice<true><<<grid,block>>>(sdfSlice,width,height,origin,size,T_pm[0],
            models[0]->getDeviceTransformsModelToFrame(),
            models[0]->getDeviceSdfFrames(),
            models[0]->getDeviceSdfs(),
            models[0]->getNumSdfs());
    for (int m=1; m<models.size(); ++m) {
        gpu_getMultiModelSdfSlice<false><<<grid,block>>>(sdfSlice,width,height,origin,size,T_pm[m],
                                                        models[m]->getDeviceTransformsModelToFrame(),
                                                        models[m]->getDeviceSdfFrames(),
                                                        models[m]->getDeviceSdfs(),
                                                        models[m]->getNumSdfs());
    }

}

void getModelSdfSlice(float * sdfSlice,
                      const int width,
                      const int height,
                      const float2 origin,
                      const float2 size,
                      const SE3 & T_pm,
                      const MirroredModel & model) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_getModelSdfSlice<<<grid,block>>>(sdfSlice,width,height,origin,size,T_pm,
                                         model.getDeviceTransformsModelToFrame(),
                                         model.getDeviceSdfFrames(),
                                         model.getDeviceSdfs(),
                                         model.getNumSdfs());

}

void getSdfSlice(float * sdfSlice,
                 const int width,
                 const int height,
                 const float2 origin,
                 const float2 size,
                 const SE3 & T_sp,
                 const Grid3D<float> * deviceSdf) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_getSdfSlice<<<grid,block>>>(sdfSlice,width,height,origin,size,T_sp,deviceSdf);

}

void getModelSdfPlaneProjective(float * sdf,
                                const int width,
                                const int height,
                                const SE3 & T_mc,
                                const SE3 * T_fms,
                                const int * sdfFrames,
                                const Grid3D<float> * sdfs,
                                const int nSdfs,
                                const float planeDepth,
                                const float focalLength) {
    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_getModelSdfPlaneProjective<<<grid,block>>>(sdf,width,height,T_mc,T_fms,sdfFrames,sdfs,nSdfs,planeDepth,focalLength);
}

void getModelSdfGradientPlaneProjective(float3 * grad,
                                        const int width,
                                        const int height,
                                        const SE3 & T_mc,
                                        const SE3 * T_fms,
                                        const int * sdfFrames,
                                        const Grid3D<float> * sdfs,
                                        const int nSdfs,
                                        const float planeDepth,
                                        const float focalLength) {
    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_getModelSdfGradientPlaneProjective<<<grid,block>>>(grad,width,height,T_mc,T_fms,sdfFrames,sdfs,nSdfs,planeDepth,focalLength);
}

void getObservationSdfPlane(float * sdfVals,
                            const int width,
                            const int height,
                            const Grid3D<float> * sdf,
                            const float planeDepth) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_getObservationSdfPlane<<<grid,block>>>(sdfVals,
                                               width,
                                               height,
                                               sdf,
                                               planeDepth);

}

void getObservationSdfPlaneProjective(float * sdfVals,
                                      const int width,
                                      const int height,
                                      const Grid3D<float> * sdf,
                                      const float planeDepth,
                                      const float focalLength) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_getObservationSdfPlaneProjective<<<grid,block>>>(sdfVals,
                                                         width,
                                                         height,
                                                         sdf,
                                                         planeDepth,
                                                         focalLength);

}

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
                                   const float planeDepth) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_visualizeDataAssociationPlane<<<grid,block>>>(img,width,height,
                                                      origin,size,
                                                      T_mc,T_fms,sdfFrames,
                                                      sdfs,nSdfs,sdfColors,
                                                      planeDepth);

}

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
                                             const float focalLength) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_visualizeDataAssociationPlaneProjective<<<grid,block>>>(img,width,height,
                                                                T_mc,T_fms,sdfFrames,
                                                                sdfs,nSdfs,sdfColors,
                                                                planeDepth,focalLength);

}

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
                                             const float focalLength) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_visualizeDataAssociationPlaneProjective<<<grid,block>>>(img,width,height,
                                                                T_mc,T_fms,sdfFrames,
                                                                sdfs,nSdfs,sdfColors,
                                                                planeDepth,focalLength);

}

}
