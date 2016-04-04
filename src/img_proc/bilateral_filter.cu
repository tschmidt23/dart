#include "bilateral_filter.h"

#include <iostream>
#include <cuda_runtime.h>

namespace dart {

template <typename T>
__global__ void gpu_bilateralFilter(const T * depthIn,
                                    float * depthOut,
                                    const int width,
                                    const int height,
                                    const float domainFactor,
                                    const float rangeFactor) {

    const int twidth = 16;
    const int theight = 16;
    const int swidth = twidth+8;
    const int sheight = theight+8;
    const int nreads = (swidth*sheight)/(twidth*theight);

    __shared__ T sdata[swidth*sheight];

    const int tid = threadIdx.y*blockDim.x + threadIdx.x;

    for (int i=0; i<=nreads; i++) {
        const int readi = tid + i*twidth*theight;
        if (readi < swidth*sheight) {
            const int row = readi / swidth;
            const int col = readi % swidth;
            const int x = blockIdx.x*blockDim.x - 4 + col;
            const int y = blockIdx.y*blockDim.y - 4 + row;
            if (x >= 0 && x < width && y >=0 && y < height) {
                sdata[readi] = depthIn[x + y*width];
            }
        }
    }

    __syncthreads();

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int index = x + y*width;

    const int bx = threadIdx.x + 4;
    const int by = threadIdx.y + 4;

    float d = sdata[bx + by*swidth];

    if (d > 0) {

        float new_d = 0;
        float total_weight = 0;

        int min_dx = max(-4,-x);
        int max_dx = min(4,width-x-1);
        int min_dy = max(-4,-y);
        int max_dy = min(4,height-y-1);

        for (int dy = min_dy; dy <= max_dy; dy++) {
            for (int dx = min_dx; dx <= max_dx; dx++) {

                float dd = sdata[bx + dx + (by+dy)*swidth];

                if ( dd > 0 ) {

                    float rangeDist2 = (d-dd)*(d-dd);
                    float domainDist2 = dx*dx + dy*dy;

                    float weight = expf( -domainDist2*domainFactor - rangeDist2*rangeFactor);

                    new_d += (weight*dd);
                    total_weight += weight;

                }

            }
        }

        depthOut[index] = (new_d / total_weight);

    } else {
        depthOut[index] = 0;
    }

}

template <typename T>
void bilateralFilter(const T * depthIn, float * depthOut, const int width, const int height, const float sigmaDomain, const float sigmaRange) {

    dim3 block(16,16,1);
    dim3 grid( (width) / block.x, (height) / block.y);

    gpu_bilateralFilter<T><<<grid,block>>>(depthIn, depthOut, width, height, 1.0/(2*sigmaDomain*sigmaDomain), 1.0/(2*sigmaRange*sigmaRange));

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    }

}

template void bilateralFilter<float>(const float * depthIn, float * depthOut, const int width, const int height, const float sigmaDomain, const float sigmaRange);
template void bilateralFilter<double>(const double * depthIn, float * depthOut, const int width, const int height, const float sigmaDomain, const float sigmaRange);
template void bilateralFilter<ushort>(const ushort * depthIn, float * depthOut, const int width, const int height, const float sigmaDomain, const float sigmaRange);

}
