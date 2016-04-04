#include "distance_transforms.h"
#include <cuda_runtime_api.h>

namespace dart {

// kernels
template <typename Real, bool takeSqrt>
__global__ void gpu_distanceTransform1D(const Real * fIn, Real * fOut, int n, Real * z, int * v) {

    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    for (int q = 1; q <= n-1; q++) {
        Real s = ((fIn[q]+q*q) - (fIn[v[k]]+v[k]*v[k]))/(2*q-2*v[k]);
        while (s <= z[k]) {
            k--;
            s = ((fIn[q]+q*q)-(fIn[v[k]]+v[k]*v[k]))/(2*q-2*v[k]);
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k+1] = +INF;
    }

    k = 0;
    for (int q = 0; q <= n-1; q++) {
        while (z[k+1] < q)
            k++;
        if (takeSqrt)
            fOut[q] = sqrtf((q-v[k])*(q-v[k]) + fIn[v[k]]);
        else
            fOut[q] = (q-v[k])*(q-v[k]) + fIn[v[k]];
    }

}

template <typename Real, bool takeSqrt>
__global__ void gpu_stridedDistanceTransform1D(const Real * fIns, Real * fOuts, const unsigned int len, const unsigned int num,
                                               const unsigned int outerStride, const unsigned int innerStride,
                                               Real * zs, int * vs) {

    const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid >= num)
        return;

    const Real * fIn = &fIns[tid*outerStride];
    Real * fOut = &fOuts[tid*outerStride];

    Real * z = zs + tid*(len+1);
    int * v = vs + tid*len;

    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    for (int q = 1; q <= len-1; q++) {
        Real s = ((fIn[q*innerStride]+q*q) - (fIn[v[k]*innerStride]+v[k]*v[k]))/(2*q-2*v[k]);
        while (s <= z[k]) {
            k--;
            s = ((fIn[q*innerStride]+q*q)-(fIn[v[k]*innerStride]+v[k]*v[k]))/(2*q-2*v[k]);
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k+1] = +INF;
    }

    k = 0;
    for (int q = 0; q <= len-1; q++) {
        while (z[k+1] < q)
            k++;
        if (takeSqrt)
            fOut[q*innerStride] = sqrtf((q-v[k])*(q-v[k]) + fIn[v[k]*innerStride]);
        else
            fOut[q*innerStride] = (q-v[k])*(q-v[k]) + fIn[v[k]*innerStride];
    }

}

template <typename Real, bool takeSqrt>
__global__ void gpu_doublyStridedDistanceTransform1D(const Real * fIns, Real * fOuts, const unsigned int len, const unsigned int maxA, const unsigned int maxB,
                                                     const unsigned int outerStrideA, const unsigned int outerStrideB, const unsigned int innerStride,
                                                     Real * zs, int * vs) {

    const unsigned int a = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y*blockDim.y + threadIdx.y;

    if (a >= maxA || b >= maxB)
        return;

    const Real * fIn = &fIns[a*outerStrideA + b*outerStrideB];
    Real * fOut = &fOuts[a*outerStrideA + b*outerStrideB];

    const unsigned int tid = a + b*maxA;
    Real * z = zs + (tid)*(len+1);
    int * v = vs + tid*len;

    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    for (int q = 1; q <= len-1; q++) {
        Real s = ((fIn[q*innerStride]+q*q) - (fIn[v[k]*innerStride]+v[k]*v[k]))/(2*q-2*v[k]);
        while (s <= z[k]) {
            k--;
            s = ((fIn[q*innerStride]+q*q)-(fIn[v[k]*innerStride]+v[k]*v[k]))/(2*q-2*v[k]);
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k+1] = +INF;
    }

    k = 0;
    for (int q = 0; q <= len-1; q++) {
        while (z[k+1] < q)
            k++;
        if (takeSqrt)
            fOut[q*innerStride] = sqrtf((q-v[k])*(q-v[k]) + fIn[v[k]*innerStride]);
        else
            fOut[q*innerStride] = (q-v[k])*(q-v[k]) + fIn[v[k]*innerStride];
    }

}

template <typename Real>
__global__ void gpu_seedInverseTransform2D(const Real * im, Real * seed, const unsigned int width, const unsigned int height) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int index = x + width*y;

    if (im[index] == 0) {
        seed[index] = INF;
        if (x > 0) {
            if (im[index-1] != 0.0) {
                seed[index] = 0.0;
                return;
            }
        }
        if (x < width - 1 ) {
            if (im[index+1] != 0.0) {
                seed[index] = 0.0;
                return;
            }
        }
        if (y > 0) {
            if (im[index-width] != 0.0) {
                seed[index] = 0.0;
                return;
            }
        }
        if (y < height - 1) {
            if (im[index+width] != 0.0) {
                seed[index] = 0.0;
                return;
            }
        }
    }
    else {
        seed[index] = 0;
    }

}

template <typename Real>
__global__ void gpu_seedInverseTransform3D(const Real * im, Real * seed, const unsigned int width, const unsigned int height, const  unsigned int depth) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth)
        return;

    const int index = x + width*(y + height*z);

    if (im[index] == 0) {
        seed[index] = INF;
        if (x > 0) {
            if (im[index-1] != 0.0) {
                seed[index] = 0.0;
                return;
            }
        }
        if (x < width - 1 ) {
            if (im[index+1] != 0.0) {
                seed[index] = 0.0;
                return;
            }
        }
        if (y > 0) {
            if (im[index-width] != 0.0) {
                seed[index] = 0.0;
                return;
            }
        }
        if (y < height - 1) {
            if (im[index+width] != 0.0) {
                seed[index] = 0.0;
                return;
            }
        }
        if (z > 0) {
            if (im[index-width*height] != 0.0) {
                seed[index] = 0.0;
                return;
            }
        }
        if (z < depth - 1) {
            if (im[index+width*height] != 0.0) {
                seed[index] = 0.0;
                return;
            }
        }
    }
    else {
        seed[index] = 0;
    }

}


template <typename Real>
__global__ void gpu_combineTransforms2D(Real * im, Real * neg, Real * mask, const unsigned int width, const unsigned int height) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int index = x + width*y;

    if (mask[index] == 0) {
        im[index] = -neg[index];
    }

}

template <typename Real>
__global__ void gpu_combineTransforms3D(Real * im, Real * neg, Real * mask, const unsigned int width, const unsigned int height, const unsigned int depth) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth)
        return;

    const int index = x + width*(y + height*z);

    if (mask[index] == 0) {
        im[index] = -neg[index];
    }

}

// host interface functions
template <typename Real>
void distanceTransform1D(const Real * in, Real * out, const unsigned int width, bool takeSqrt) {

    dim3 block(1,1,1);
    dim3 grid(1,1,1);

    Real * z; cudaMalloc(&z,(width+1)*sizeof(Real));
    int * v; cudaMalloc(&v,width*sizeof(int));

    if (takeSqrt) {
        gpu_distanceTransform1D<Real,true><<<grid,block>>>(in,out,width,z,v);
    } else {
        gpu_distanceTransform1D<Real,false><<<grid,block>>>(in,out,width,z,v);
    }
    cudaDeviceSynchronize();

    cudaFree(z);
    cudaFree(v);

}

template <typename Real>
void distanceTransform1D(const Real * in, Real * out, const unsigned int width, bool takeSqrt, Real * zScratch, int * vScratch) {

    dim3 block(1,1,1);
    dim3 grid(1,1,1);

    if (takeSqrt) {
        gpu_distanceTransform1D<Real,true><<<grid,block>>>(in,out,width,zScratch,vScratch);
    } else {
        gpu_distanceTransform1D<Real,false><<<grid,block>>>(in,out,width,zScratch,vScratch);
    }

}

template <typename Real, bool takeSqrt>
void distanceTransform2D(Real * im, Real * scratch, const unsigned int width, const unsigned int height) {

    Real * zScratch; cudaMalloc(&zScratch,(width+1)*(height+1)*sizeof(Real));
    int * vScratch; cudaMalloc(&vScratch,width*height*sizeof(int));

    distanceTransform2D<Real,takeSqrt>(im,scratch,width,height,zScratch,vScratch);

    cudaFree(zScratch);
    cudaFree(vScratch);

}

template <typename Real>
__global__ void gpu_stridedDistanceTransform1Da(const Real * fIns, Real * fOuts, Real * fIntermediates, const unsigned int len, const unsigned int num,
                                                const unsigned int outerStride, const unsigned int innerStride,
                                                Real * zs, int * vs) {

    const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid >= num)
        return;

    const Real * fIn = &fIns[tid*outerStride];
    Real * fOut = &fOuts[tid*outerStride];
    Real * fIntermediate = &fIntermediates[tid*outerStride];

    Real * z = zs + tid*(len+1);
    int * v = vs + tid*len;

    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    for (int q = 1; q <= len-1; q++) {
        Real s = ((fIn[q*innerStride]+q*q) - (fIn[v[k]*innerStride]+v[k]*v[k]))/(2*q-2*v[k]);
        while (s <= z[k]) {
            k--;
            s = ((fIn[q*innerStride]+q*q)-(fIn[v[k]*innerStride]+v[k]*v[k]))/(2*q-2*v[k]);
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k+1] = +INF;
    }

    k = 0;
    for (int q = 0; q <= len-1; q++) {
        while (z[k+1] < q)
            k++;
        fIntermediate[q*innerStride] = fIn[v[k]*innerStride];
        fOut[q*innerStride] = (q-v[k])*(q-v[k]) + fIn[v[k]*innerStride];
    }

}

template <typename Real, bool takeSqrt>
__global__ void gpu_stridedDistanceTransform1Db(const Real * fIns, const Real * fIntermediates, Real * fOuts, const unsigned int len, const unsigned int num,
                                               const unsigned int outerStride, const unsigned int innerStride,
                                               Real * zs, int * vs) {

    const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid >= num)
        return;

    const Real * fIn = &fIns[tid*outerStride];
    const Real * fIntermediate = &fIntermediates[tid*outerStride];
    Real * fOut = &fOuts[tid*outerStride];

    Real * z = zs + tid*(len+1);
    int * v = vs + tid*len;

    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    for (int q = 1; q <= len-1; q++) {
        Real s = ((fIn[q*innerStride]+q*q) - (fIn[v[k]*innerStride]+v[k]*v[k]))/(2*q-2*v[k]);
        while (s <= z[k]) {
            k--;
            s = ((fIn[q*innerStride]+q*q)-(fIn[v[k]*innerStride]+v[k]*v[k]))/(2*q-2*v[k]);
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k+1] = +INF;
    }

    k = 0;
    for (int q = 0; q <= len-1; q++) {
        while (z[k+1] < q)
            k++;
        if (takeSqrt)
            fOut[q*innerStride] = sqrtf((q-v[k])*(q-v[k]) + fIn[v[k]*innerStride] - fIntermediate[v[k]*innerStride]) + fIntermediate[v[k]*innerStride];
        else
            fOut[q*innerStride] = (q-v[k])*(q-v[k]) + fIn[v[k]*innerStride];
    }

}

template <typename Real, bool takeSqrt>
void distanceTransform2D(Real * im, Real * scratch, const unsigned int width, const unsigned int height, Real * zScratch, int * vScratch) {

    dim3 block(64,1,1);
    dim3 grid( ceil( height / (float)block.x), 1, 1);

    Real *fIntermediate;
    cudaMalloc(&fIntermediate,width*height*sizeof(Real));

    // x-direction
    gpu_stridedDistanceTransform1Da<Real><<<grid,block>>>(im,scratch,fIntermediate,width,height,width,1,zScratch,vScratch);
    cudaDeviceSynchronize();

    grid = dim3( ceil( width / (float)block.x), 1, 1);

    // y-direction
    gpu_stridedDistanceTransform1Db<Real,takeSqrt><<<grid,block>>>(scratch,fIntermediate,im,height,width,1,width,zScratch,vScratch);

}

template <typename Real, bool takeSqrt>
void distanceTransform3D(Real * in, Real * out, const unsigned int width, const unsigned int height, const unsigned int depth) {

    Real * zScratch; cudaMalloc(&zScratch,(width+1)*(height+1)*(depth+1)*sizeof(Real));
    int * vScratch; cudaMalloc(&vScratch,width*height*depth*sizeof(int));

    distanceTransform3D<Real,takeSqrt>(in,out,width,height,depth,zScratch,vScratch);

    cudaFree(zScratch);
    cudaFree(vScratch);
}

template <typename Real, bool takeSqrt>
void distanceTransform3D(Real * in, Real * out, const unsigned int width, const unsigned int height, const unsigned int depth, Real * zScratch, int * vScratch) {

    dim3 block(64,1,1);
    dim3 grid( ceil( height*depth / (float)block.x), 1, 1);

    // x-direction
    gpu_stridedDistanceTransform1D<Real,false><<<grid,block>>>(in,out,width,height*depth,width,1,
                                                               zScratch,vScratch);
    cudaDeviceSynchronize();

    // z-direction
    grid = dim3( ceil(width*height / (float)block.x), 1, 1);
    gpu_stridedDistanceTransform1D<Real,false><<<grid,block>>>(out,in,depth,width*height,1,width*height,
                                                               zScratch,vScratch);
    cudaDeviceSynchronize();

    block = dim3(16,16,1);
    grid = dim3(ceil(width / (float)block.x), ceil(depth / (float)block.y), 1);

    // y-direction
    gpu_doublyStridedDistanceTransform1D<Real,takeSqrt><<<grid,block>>>(in,out,height,width,depth,1,width*height,width,
                                                                        zScratch,vScratch);

}

template <typename Real, bool takeSqrt>
void signedDistanceTransform2D(Real * in, Real * out, const unsigned int width, const unsigned int height) {

    Real *zScratch; cudaMalloc(&zScratch,(width+1)*(height+1)*sizeof(Real));
    int *vScratch; cudaMalloc(&vScratch,width*height*sizeof(int));

    signedDistanceTransform2D<Real,takeSqrt>(in,out,width,height,zScratch,vScratch);

    cudaFree(zScratch);
    cudaFree(vScratch);

}

template <typename Real, bool takeSqrt>
void signedDistanceTransform2D(Real * in, Real * out, const unsigned int width, const unsigned int height,
                               Real * zScratch, int *vScratch) {

    Real * imScratch1; cudaMalloc(&imScratch1,width*height*sizeof(Real));
    Real * imScratch2; cudaMalloc(&imScratch2,width*height*sizeof(Real));

    signedDistanceTransform2D<Real,takeSqrt>(in,out,width,height,zScratch,vScratch,imScratch1,imScratch2);

    cudaFree(imScratch1);
    cudaFree(imScratch2);

}

template <typename Real, bool takeSqrt>
void signedDistanceTransform2D(Real * in, Real * out, const unsigned int width, const unsigned int height,
                               Real * zScratch, int * vScratch, Real * imScratch1, Real * imScratch2) {

    dim3 block(16,8,1);
    dim3 grid(ceil(width/(float)block.x),ceil(height/(float)block.y),1);

    Real * seed = out;
    gpu_seedInverseTransform2D<<<grid,block>>>(in,seed,width,height);
    cudaDeviceSynchronize();

    Real * mask = imScratch1;
    cudaMemcpy(mask,in,width*height*sizeof(Real),cudaMemcpyDeviceToDevice);

    distanceTransform2D<Real,takeSqrt>(in,imScratch2,width,height,zScratch,vScratch);
    cudaDeviceSynchronize();

    distanceTransform2D<Real,takeSqrt>(seed,imScratch2,width,height,zScratch,vScratch);
    cudaDeviceSynchronize();

    Real * inDT = in;
    Real * seedDT = seed;
    gpu_combineTransforms2D<<<grid,block>>>(inDT,seedDT,mask,width,height);

    cudaMemcpy(out,inDT,width*height*sizeof(Real),cudaMemcpyDeviceToDevice);

}

template <typename Real, bool takeSqrt>
void signedDistanceTransform3D(Real * in, Real * out, const unsigned int width, const unsigned int height, const unsigned int depth) {

    Real * zScratch; cudaMalloc(&zScratch,(width+1)*(height+1)*(depth+1)*sizeof(Real));
    int * vScratch; cudaMalloc(&vScratch,width*height*depth*sizeof(int));

    signedDistanceTransform3D<Real,takeSqrt>(in,out,width,height,depth,zScratch,vScratch);

    cudaFree(zScratch);
    cudaFree(vScratch);

}

template <typename Real, bool takeSqrt>
void signedDistanceTransform3D(Real * in, Real * out, const unsigned int width, const unsigned int height, const unsigned int depth,
                               Real * zScratch, int * vScratch) {

    Real * imScratch; cudaMalloc(&imScratch,width*height*depth*sizeof(Real));

    signedDistanceTransform3D<Real,takeSqrt>(in,out,width,height,depth,zScratch,vScratch,imScratch);

    cudaFree(imScratch);

}

template <typename Real, bool takeSqrt>
void signedDistanceTransform3D(Real * in, Real * out, const unsigned int width, const unsigned int height, const unsigned int depth,
                               Real * zScratch, int * vScratch, Real * imScratch) {

    dim3 block(8,8,4);
    dim3 grid(ceil(width/(float)block.x),ceil(height/(float)block.y),ceil(depth/(float)block.z));

    Real * seed = out;
    gpu_seedInverseTransform3D<<<grid,block>>>(in,seed,width,height,depth);
    cudaDeviceSynchronize();

    Real * seedDT = imScratch;
    distanceTransform3D<Real,takeSqrt>(seed,seedDT,width,height,depth,zScratch,vScratch);
    cudaDeviceSynchronize();

    Real * inDT = out;
    distanceTransform3D<Real,takeSqrt>(in,inDT,width,height,depth,zScratch,vScratch);
    cudaDeviceSynchronize();

    Real * mask = in;
    gpu_combineTransforms3D<<<grid,block>>>(inDT,seedDT,mask,width,height,depth);

}

#define DECLARE_DISTANCE_TRANSFORM_1D(REAL) \
    template void distanceTransform1D<REAL>(const REAL * in, REAL * out, const unsigned int width, bool takeSqrt); \
    template void distanceTransform1D<REAL>(const REAL * in, REAL * out, const unsigned int width, bool takeSqrt, REAL * zScratch, int * vScratch);
#define DECLARE_DISTANCE_TRANSFORM_2D(REAL,TAKE_SQRT) \
    template void distanceTransform2D<REAL,TAKE_SQRT>(REAL * in, REAL * out, const unsigned int width, const unsigned int height); \
    template void distanceTransform2D<REAL,TAKE_SQRT>(REAL * in, REAL * out, const unsigned int width, const unsigned int height, REAL * zScratch, int * vScratch);
#define DECLARE_DISTANCE_TRANSFORM_3D(REAL,TAKE_SQRT) \
    template void distanceTransform3D<REAL,TAKE_SQRT>(REAL * in, REAL * out, const unsigned int width, const unsigned int height, const unsigned int depth); \
    template void distanceTransform3D<REAL,TAKE_SQRT>(REAL * in, REAL * out, const unsigned int width, const unsigned int height, const unsigned int depth, REAL * zScratch, int * vScratch);
#define DECLARE_SIGNED_DISTANCE_TRANSFORM_2D(REAL,TAKE_SQRT) \
    template void signedDistanceTransform2D<REAL,TAKE_SQRT>(REAL * in, REAL * out, const unsigned int width, const unsigned int height); \
    template void signedDistanceTransform2D<REAL,TAKE_SQRT>(REAL * in, REAL * out, const unsigned int width, const unsigned int height, REAL * zScratch, int * vScratch); \
    template void signedDistanceTransform2D<REAL,TAKE_SQRT>(REAL * in, REAL * out, const unsigned int width, const unsigned int height, REAL * zScratch, int * vScratch, REAL * imScratch1, REAL * imScratch2);
#define DECLARE_SIGNED_DISTANCE_TRANSFORM_3D(REAL,TAKE_SQRT) \
    template void signedDistanceTransform3D<REAL,TAKE_SQRT>(REAL * in, REAL * out, const unsigned int width, const unsigned int height, const unsigned int depth); \
    template void signedDistanceTransform3D<REAL,TAKE_SQRT>(REAL * in, REAL * out, const unsigned int width, const unsigned int height, const unsigned int depth, REAL * zScratch, int * vScratch); \
    template void signedDistanceTransform3D<REAL,TAKE_SQRT>(REAL * in, REAL * out, const unsigned int width, const unsigned int height, const unsigned int depth, REAL * zScratch, int * vScratch, REAL * imScratch);
#define DECLARE_ALL_DISTANCE_TRANSFORMS(REAL,TAKE_SQRT) \
    DECLARE_DISTANCE_TRANSFORM_2D(REAL,TAKE_SQRT) \
    DECLARE_DISTANCE_TRANSFORM_3D(REAL,TAKE_SQRT) \
    DECLARE_SIGNED_DISTANCE_TRANSFORM_2D(REAL,TAKE_SQRT) \
    DECLARE_SIGNED_DISTANCE_TRANSFORM_3D(REAL,TAKE_SQRT)

#define DECLARE_DISTANCE_TRANSFORMS(REAL) \
    DECLARE_DISTANCE_TRANSFORM_1D(REAL)

DECLARE_ALL_DISTANCE_TRANSFORMS(float,false)
DECLARE_ALL_DISTANCE_TRANSFORMS(float,true)
DECLARE_ALL_DISTANCE_TRANSFORMS(double,false)
DECLARE_ALL_DISTANCE_TRANSFORMS(double,true)

DECLARE_DISTANCE_TRANSFORMS(float)
DECLARE_DISTANCE_TRANSFORMS(double)

}
