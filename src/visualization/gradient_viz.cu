#include "gradient_viz.h"

#include <helper_math.h>

namespace dart {

inline __host__ __device__ unsigned char clamp(int c) {
    return min(max(0,c),255);
}

inline __host__ __device__ uchar3 hsv2rgb(float h, float s, float v) {
    float c = v*s;
    float hPrime = h/60.0f;
    float x = c*(1 - fabs(fmodf(hPrime,2) - 1));
    float m = c-v;
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
__global__ void gpu_visualizeImageGradient(const float2 * imgGradient, uchar3 * gradientViz, const int width, const int height, const float minMag, const float maxMag) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;

    float2 grad = imgGradient[index];
    const float angle = atan2(grad.y,grad.x);
    const float mag = length(grad);
    grad = grad / mag;

    float h = angle * 180 / M_PI;
    if (h < 0) {
        h += 360;
    }
    const float v = min(max(0.0f,(mag-minMag)/(maxMag-minMag)),1.0f);

    gradientViz[index] = hsv2rgb(h,1.0,v);

}

// -=-=-=-=-=-=-=-=-=- interface -=-=-=-=-=-=-=-=-=-
void visualizeImageGradient(const float2 * imgGradient, uchar3 * gradientViz, const int width, const int height, const float minMag, const float maxMag) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_visualizeImageGradient<<<grid,block>>>(imgGradient,gradientViz,width,height,minMag,maxMag);

}

}
