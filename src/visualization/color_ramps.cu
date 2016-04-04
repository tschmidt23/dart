#include "color_ramps.h"
#include "math.h"

namespace dart {

// -=-=-=-=-=-=-=-=-=- helper -=-=-=-=-=-=-=-=-=-
static inline __host__ __device__ unsigned char clamp(int c) {
    return min(max(0,c),255);
}

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
__global__ void gpu_colorRampHeatMap(uchar3 * colored,
                                     const float * vals,
                                     const int width,
                                     const int height,
                                     const float minVal,
                                     const float maxVal) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    uchar3 & imgVal = colored[index];

    if (isnan(vals[index])) {
        imgVal = make_uchar3(0,0,0);
        return;
    }

    const float normVal = (vals[index] - minVal)/(maxVal-minVal);

    if (normVal < 0.25) { imgVal = make_uchar3(0,clamp(255*(normVal/0.25)),255); }
    else if (normVal < 0.5) { imgVal = make_uchar3(0,255,clamp(255*((0.5-normVal)/0.25))); }
    else if (normVal < 0.75) { imgVal = make_uchar3(clamp(255*((normVal - 0.5)/0.25)),255,0); }
    else { imgVal = make_uchar3(255,clamp(255*(1.0-normVal)/0.25),0); }

}

__global__ void gpu_colorRampHeatMap(uchar4 * colored,
                                     const float * vals,
                                     const int width,
                                     const int height,
                                     const float minVal,
                                     const float maxVal) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    uchar4 & imgVal = colored[index];

    if (isnan(vals[index])) {
        imgVal = make_uchar4(0,0,0,0);
        return;
    }

    const float normVal = (vals[index] - minVal)/(maxVal-minVal);

    if (normVal < 0.25) { imgVal = make_uchar4(0,clamp(255*(normVal/0.25)),255,255); }
    else if (normVal < 0.5) { imgVal = make_uchar4(0,255,clamp(255*((0.5-normVal)/0.25)),255); }
    else if (normVal < 0.75) { imgVal = make_uchar4(clamp(255*((normVal - 0.5)/0.25)),255,0,255); }
    else { imgVal = make_uchar4(255,clamp(255*(1.0-normVal)/0.25),0,255); }

}

__global__ void gpu_colorRampHeatMapUnsat(uchar3 * colored,
                                          const float * vals,
                                          const int width,
                                          const int height,
                                          const float minVal,
                                          const float maxVal) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    uchar3 & imgVal = colored[index];

    if (isnan(vals[index])) {
        imgVal = make_uchar3(255,255,255);
        return;
    }

    const float normVal = fmaxf(0,fminf((vals[index] - minVal)/(maxVal-minVal),1));

    const float t = normVal == 1.0 ? 1.0 : fmodf(normVal,0.25)*4;
    uchar3 a, b;
    if (normVal < 0.25) { b = make_uchar3(32,191,139); a = make_uchar3(0x18,0x62,0x93); }
    else if (normVal < 0.5) { b = make_uchar3(241,232,137); a = make_uchar3(32,191,139); }
    else if (normVal < 0.75) { b = make_uchar3(198,132,63); a = make_uchar3(241,232,137); }
    else { b = make_uchar3(0xc0,0x43,0x36); a = make_uchar3(198,132,63); }
    imgVal = make_uchar3((1-t)*a.x + t*b.x,
                         (1-t)*a.y + t*b.y,
                         (1-t)*a.z + t*b.z);

}

__global__ void gpu_colorRampHeatMapUnsat(uchar4 * colored,
                                          const float * vals,
                                          const int width,
                                          const int height,
                                          const float minVal,
                                          const float maxVal) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    uchar4 & imgVal = colored[index];

    if (isnan(vals[index])) {
        imgVal = make_uchar4(0,0,0,0);
        return;
    }

    const float normVal = fmaxf(0,fminf((vals[index] - minVal)/(maxVal-minVal),1));

    const float t = normVal == 1.0 ? 1.0 : fmodf(normVal,0.25)*4;
    uchar3 a, b;
    if (normVal < 0.25) { b = make_uchar3(32,191,139); a = make_uchar3(0x18,0x62,0x93); }
    else if (normVal < 0.5) { b = make_uchar3(241,232,137); a = make_uchar3(32,191,139); }
    else if (normVal < 0.75) { b = make_uchar3(198,132,63); a = make_uchar3(241,232,137); }
    else { b = make_uchar3(0xc0,0x43,0x36); a = make_uchar3(198,132,63); }
    imgVal = make_uchar4((1-t)*a.x + t*b.x,
                         (1-t)*a.y + t*b.y,
                         (1-t)*a.z + t*b.z,255);

}

template <bool showZeroLevel>
__global__ void gpu_colorRampTopographic(uchar4 * colored,
                                         const float * vals,
                                         const int width,
                                         const int height,
                                         const float lineThickness,
                                         const float lineSpacing) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    uchar4 & imgVal = colored[index];

    if (fabs(vals[index]) < 1.5*lineThickness) {
        if (showZeroLevel) {
            float g = clamp(2*255*(fabs(vals[index])-lineThickness)/lineThickness);
            imgVal = make_uchar4(g,g,g,255);
        } else {
            imgVal = make_uchar4(255,255,255,255);
        }
    } else {
        float c = fabs(fmodf(fabs(vals[index])+lineSpacing/2,lineSpacing)-lineSpacing/2);
        if (c < lineThickness ) {
            float g;
            if (showZeroLevel) {
                g = clamp(192+64*c/lineThickness);
            } else {
                g = clamp(64+192*c/lineThickness);
            }
            imgVal = make_uchar4(g,g,g,255);
        }
        else {
            imgVal = make_uchar4(255,255,255,255);
        }
    }

}

template <bool norm>
__global__ void gpu_colorRamp2DGradient(uchar4 * colored,
                                        const float2 * grad,
                                        const int width,
                                        const int height) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    uchar4 &imgVal = colored[index];

    float2 g = grad[index];
    if (norm) { float len = sqrtf(g.x*g.x + g.y*g.y); g = make_float2(g.x/len,g.y/len); }

//    uchar3 rgb = hsv2rgb(180+180*atan2(g.x,g.y)/M_PI,1,1);
    uchar3 rgb = hsv2rgb(180+180*atan2(g.x,g.y)/M_PI,1,1);

    imgVal = make_uchar4(rgb.x,rgb.y,rgb.z,255);
}

template <bool norm>
__global__ void gpu_colorRamp3DGradient(uchar4 * colored,
                                        const float3 * grad,
                                        const int width,
                                        const int height) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    uchar4 & imgVal = colored[index];

    float3 g = grad[index];
    if (norm) { float len = sqrtf(g.x*g.x+g.y*g.y+g.z*g.z); g = make_float3(g.x/len,g.y/len,g.z/len); }

    imgVal = make_uchar4(clamp(128-128*g.x),clamp(128-128*g.y),clamp(128-128*g.z),255);
}

// -=-=-=-=-=-=-=-=-=- interface -=-=-=-=-=-=-=-=-=-
void colorRampHeatMap(uchar3 * colored,
                      const float * vals,
                      const int width,
                      const int height,
                      const float minVal,
                      const float maxVal) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_colorRampHeatMap<<<grid,block>>>(colored,vals,width,height,minVal,maxVal);

}

void colorRampHeatMap(uchar4 * colored,
                      const float * vals,
                      const int width,
                      const int height,
                      const float minVal,
                      const float maxVal) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_colorRampHeatMap<<<grid,block>>>(colored,vals,width,height,minVal,maxVal);

}

void colorRampHeatMapUnsat(uchar3 * colored,
                           const float * vals,
                           const int width,
                           const int height,
                           const float minVal,
                           const float maxVal) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_colorRampHeatMapUnsat<<<grid,block>>>(colored,vals,width,height,minVal,maxVal);

}

void colorRampHeatMapUnsat(uchar4 * colored,
                           const float * vals,
                           const int width,
                           const int height,
                           const float minVal,
                           const float maxVal) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_colorRampHeatMapUnsat<<<grid,block>>>(colored,vals,width,height,minVal,maxVal);

}

void colorRampTopographic(uchar4 * colored,
                          const float * vals,
                          const int width,
                          const int height,
                          const float lineThickness,
                          const float lineSpacing,
                          const bool showZeroLevel) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    if (showZeroLevel) {
        gpu_colorRampTopographic<true><<<grid,block>>>(colored,vals,width,height,lineThickness,lineSpacing);
    } else {
        gpu_colorRampTopographic<false><<<grid,block>>>(colored,vals,width,height,lineThickness,lineSpacing);
    }
}

void colorRamp2DGradient(uchar4 * color,
                         const float2 * grad,
                         const int width,
                         const int height,
                         const bool normalize) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    if (normalize) {
        gpu_colorRamp2DGradient<true><<<grid,block>>>(color,grad,width,height);
    } else {
        gpu_colorRamp2DGradient<false><<<grid,block>>>(color,grad,width,height);

    }

}

void colorRamp3DGradient(uchar4 * color,
                         const float3 * grad,
                         const int width,
                         const int height,
                         const bool normalize) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    if (normalize) {
        gpu_colorRamp3DGradient<true><<<grid,block>>>(color,grad,width,height);
    } else {
        gpu_colorRamp3DGradient<false><<<grid,block>>>(color,grad,width,height);
    }
}


}
