#include "resampling.h"

#include <iostream>
#include <vector_functions.h>
#include <helper_math.h>

namespace dart {

// -=-=-=-=-=-=-=-=-=- kernels -=-=-=-=-=-=-=-=-=-
template <typename T, int factor>
__global__ void gpu_downsampleNearest(const T * imgIn, const uint2 dimIn, T * imgOut) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= dimIn.x/factor || y >= dimIn.y/factor) {
        return;
    }

    imgOut[x + y*dimIn.x/factor] = imgIn[x*factor + y*factor*dimIn.x];
}

template <int factor>
__global__ void gpu_downsampleAreaAverage(const float * imgIn, const uint2 dimIn, float * imgOut) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= dimIn.x/factor || y >= dimIn.y/factor) {
        return;
    }

    float val = 0;
#pragma unroll
    for (int dy=0; dy < factor; ++dy) {
        for (int dx=0; dx < factor; ++dx) {
            const float &d = imgIn[factor*x+dx + (factor*y+dy)*dimIn.x];
            val += d;
        }
    }

    imgOut[x + y*dimIn.x/factor] = val/(factor*factor);
}


template <int factor>
__global__ void gpu_downsampleAreaAverage(const uchar3 * imgIn, const uint2 dimIn, uchar3 * imgOut) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= dimIn.x/factor || y >= dimIn.y/factor) {
        return;
    }

    int3 val = make_int3(0);
#pragma unroll
    for (int dy=0; dy < factor; ++dy) {
        for (int dx=0; dx < factor; ++dx) {
            const uchar3 &d = imgIn[factor*x+dx + (factor*y+dy)*dimIn.x];
            val.x += d.x;
            val.y += d.y;
            val.z += d.z;
        }
    }

    imgOut[x + y*dimIn.x/factor] = make_uchar3(val.x/(factor*factor),val.y/(factor*factor),val.z/(factor*factor));
}

template <int factor>
__global__ void gpu_downsampleAreaAverage(const uchar4 * imgIn, const uint2 dimIn, uchar4 * imgOut) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= dimIn.x/factor || y >= dimIn.y/factor) {
        return;
    }

    int3 val = make_int3(0);
#pragma unroll
    for (int dy=0; dy < factor; ++dy) {
        for (int dx=0; dx < factor; ++dx) {
            const uchar4 &d = imgIn[factor*x+dx + (factor*y+dy)*dimIn.x];
            val.x += d.x;
            val.y += d.y;
            val.z += d.z;
        }
    }

    imgOut[x + y*dimIn.x/factor] = make_uchar4(val.x/(factor*factor),val.y/(factor*factor),val.z/(factor*factor),255);
}

template <int factor, bool ignoreZero>
__global__ void gpu_downsampleMin(const float * imgIn, const uint2 dimIn, float * imgOut) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= dimIn.x/factor || y >= dimIn.y/factor) {
        return;
    }

    float minVal = 0;
    for (int dy=0; dy < factor; ++dy) {
        for (int dx=0; dx < factor; ++dx) {
            const float &val = imgIn[factor*x+dx + (factor*y+dy)*dimIn.x];
            if (ignoreZero) {
                if (val != 0 && val < minVal) {
                    minVal = val;
                }
            }
            else if (val < minVal) {
                minVal = val;
            }
        }
    }

    imgOut[x + y*dimIn.x/factor] = minVal;
}

// -=-=-=-=-=-=-=-=-=- interface -=-=-=-=-=-=-=-=-=-
void downsampleAreaAverage(const float * imgIn, const uint2 dimIn, float * imgOut, const int factor) {

    dim3 block(16,8,1);
    dim3 grid( ceil( (dimIn.x/factor) / (float)block.x), ceil( (dimIn.y/factor) / (float)block.y ));

    switch (factor) {
    case 2:
        gpu_downsampleAreaAverage<2><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 4:
        gpu_downsampleAreaAverage<4><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 8:
        gpu_downsampleAreaAverage<8><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 16:
        gpu_downsampleAreaAverage<16><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    default:
        std::cout << "downsampling factor " << factor << " not supported" << std::endl;
        break;
    }

}

void downsampleAreaAverage(const uchar3 * imgIn, const uint2 dimIn, uchar3 * imgOut, const int factor) {

    dim3 block(16,8,1);
    dim3 grid( ceil( (dimIn.x/factor) / (float)block.x), ceil( (dimIn.y/factor) / (float)block.y ));

    switch (factor) {
    case 2:
        gpu_downsampleAreaAverage<2><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 4:
        gpu_downsampleAreaAverage<4><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 8:
        gpu_downsampleAreaAverage<8><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 16:
        gpu_downsampleAreaAverage<16><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    default:
        std::cout << "downsampling factor " << factor << " not supported" << std::endl;
        break;
    }

}

void downsampleAreaAverage(const uchar4 * imgIn, const uint2 dimIn, uchar4 * imgOut, const int factor) {

    dim3 block(16,8,1);
    dim3 grid( ceil( (dimIn.x/factor) / (float)block.x), ceil( (dimIn.y/factor) / (float)block.y ));

    switch (factor) {
    case 2:
        gpu_downsampleAreaAverage<2><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 4:
        gpu_downsampleAreaAverage<4><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 8:
        gpu_downsampleAreaAverage<8><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 16:
        gpu_downsampleAreaAverage<16><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    default:
        std::cout << "downsampling factor " << factor << " not supported" << std::endl;
        break;
    }

}

void downsampleNearest(const float * imgIn, const uint2 dimIn, float * imgOut, const int factor) {

    dim3 block(16,8,1);
    dim3 grid( ceil( (dimIn.x/factor) / (float)block.x), ceil( (dimIn.y/factor) / (float)block.y ));

    switch (factor) {
    case 2:
        gpu_downsampleNearest<float,2><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 4:
        gpu_downsampleNearest<float,4><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 8:
        gpu_downsampleNearest<float,8><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 16:
        gpu_downsampleNearest<float,16><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    default:
        std::cout << "downsampling factor " << factor << " not supported" << std::endl;
        break;
    }

}

void downsampleNearest(const uchar3 * imgIn, const uint2 dimIn, uchar3 * imgOut, const int factor) {

    dim3 block(16,8,1);
    dim3 grid( ceil( (dimIn.x/factor) / (float)block.x), ceil( (dimIn.y/factor) / (float)block.y ));

    switch (factor) {
    case 2:
        gpu_downsampleNearest<uchar3,2><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 4:
        gpu_downsampleNearest<uchar3,4><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 8:
        gpu_downsampleNearest<uchar3,8><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    case 16:
        gpu_downsampleNearest<uchar3,16><<<grid,block>>>(imgIn,dimIn,imgOut);
        break;
    default:
        std::cout << "downsampling factor " << factor << " not supported" << std::endl;
        break;
    }

}

void downsampleMin(const float * imgIn, const uint2 dimIn, float * imgOut, const int factor, bool ignoreZero) {

    dim3 block(16,8,1);
    dim3 grid( ceil( (dimIn.x/factor) / (float)block.x), ceil( (dimIn.y/factor) / (float)block.y ));

    switch (factor) {
    case 2:
        if (ignoreZero) {
            gpu_downsampleMin<2,true><<<grid,block>>>(imgIn,dimIn,imgOut);
        } else {
            gpu_downsampleMin<2,false><<<grid,block>>>(imgIn,dimIn,imgOut);
        }
        break;
    case 4:
        if (ignoreZero) {
            gpu_downsampleMin<4,true><<<grid,block>>>(imgIn,dimIn,imgOut);
        } else {
            gpu_downsampleMin<4,false><<<grid,block>>>(imgIn,dimIn,imgOut);
        }
        break;
    case 8:
        if (ignoreZero) {
            gpu_downsampleMin<8,true><<<grid,block>>>(imgIn,dimIn,imgOut);
        } else {
            gpu_downsampleMin<8,false><<<grid,block>>>(imgIn,dimIn,imgOut);
        }
        break;
    case 16:
        if (ignoreZero) {
            gpu_downsampleMin<16,true><<<grid,block>>>(imgIn,dimIn,imgOut);
        } else {
            gpu_downsampleMin<16,false><<<grid,block>>>(imgIn,dimIn,imgOut);
        }
        break;
    default:
        std::cout << "downsampling factor " << factor << " not supported" << std::endl;
        break;
    }

}

}
