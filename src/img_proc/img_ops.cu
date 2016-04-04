#include "img_ops.h"

namespace dart {

// -=-=-=-=-=-=-=-=-=- kernels -=-=-=-=-=-=-=-=-=-
__global__ void gpu_imageSquare(float * out, const float * in, const int width, const int height) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int index = x + y*width;
    out[index] = in[index]*in[index];

}

__global__ void gpu_imageSqrt(float * out, const float * in, const int width, const int height) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int index = x + y*width;
    out[index] = sqrtf(in[index]);

}

template <typename T>
__global__ void gpu_imageFlipX(T * out, const T * in, const int width, const int height) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    out[x + y*width] = in[width - x + y*width];

}

template <typename T>
__global__ void gpu_imageFlipXInPlace(T * img, const int width, const int height) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width/2 || y >= height) {
        return;
    }

    T tmp = img[x + y*width];
    img[x + y*width] = img[width - x + y*width];
    img[width - x + y*width] = tmp;

}

template <typename T>
__global__ void gpu_imageFlipY(T * out, const T * in, const int width, const int height) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    out[x + y*width] = in[x + (height-y)*width];

}

template <typename T>
__global__ void gpu_imageFlipYInPlace(T * img, const int width, const int height) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height/2) {
        return;
    }

    T tmp = img[x + y*width];
    img[x + y*width] = img[x + (height-y)*width];
    img[x + (height-y)*width] = tmp;

}

template <typename T>
__global__ void gpu_unitNormalize(const T * in, T * out, const int width, const int height, const T zeroVal, const T range) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    const int index = x + y*width;
    out[index] = fmaxf(fminf((in[index] - zeroVal)/range,(T)1),(T)0);

}

// -=-=-=-=-=-=-=-=-=- interface -=-=-=-=-=-=-=-=-=-
void imageSquare(float * out, const float * in, const int width, const int height) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil( height / (float)block.y ));

    gpu_imageSquare<<<grid,block>>>(out,in,width,height);
}

void imageSqrt(float * out, const float * in, const int width, const int height) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil( height / (float)block.y ));

    gpu_imageSqrt<<<grid,block>>>(out,in,width,height);

}

template <typename T>
void imageFlipX(T * out, const T * in, const int width, const int height) {

    dim3 block(16,8,1);

    if (out == in) {
        dim3 grid( ceil( width/2 / (float)block.x), ceil( height / (float)block.y ));
        gpu_imageFlipXInPlace<<<grid,block>>>(out,width,height);
    }
    else {
        dim3 grid( ceil( width / (float)block.x), ceil( height / (float)block.y ));
        gpu_imageFlipX<<<grid,block>>>(out,in,width,height);
    }

}

template <typename T>
void imageFlipY(T * out, const T * in, const int width, const int height) {

    dim3 block(16,8,1);

    if (out == in) {
        dim3 grid( ceil( width / (float)block.x), ceil( height/2 / (float)block.y ));
        gpu_imageFlipYInPlace<<<grid,block>>>(out,width,height);
    }
    else {
        dim3 grid( ceil( width / (float)block.x), ceil( height / (float)block.y ));
        gpu_imageFlipY<<<grid,block>>>(out,in,width,height);
    }

}

template <typename T>
void unitNormalize(const T * in, T * out, const int width, const int height, const T zeroVal, const T oneVal) {

    dim3 block(16,16,1);
    dim3 grid( ceil((float)width / block.x), ceil((float)height / block.y));

    T range = oneVal - zeroVal;

    gpu_unitNormalize<<<grid,block>>>(in,out,width,height,zeroVal,range);

    cudaDeviceSynchronize();

}


#define COMPILE_IMAGE_OPS(type) \
    template void imageFlipX<type>(type * out, const type * in, const int width, const int height); \
    template void imageFlipY<type>(type * out, const type * in, const int width, const int height); \
    template void unitNormalize<type>(const type * in, type * out, const int width, const int height, const type zeroVal, const type oneVal);

COMPILE_IMAGE_OPS(float)
COMPILE_IMAGE_OPS(ushort)
//COMPILE_IMAGE_OPS(uchar3)

}
