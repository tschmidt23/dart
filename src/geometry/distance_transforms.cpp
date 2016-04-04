#include "distance_transforms.h"

#include <math.h>

//#include <algorithm>

namespace dart {

// helper functions
template <typename Real, bool takeSqrt>
void stridedDistanceTransform1D(Real* in, Real* out, const unsigned int len, const unsigned int num,
                                const unsigned int outerStride, const unsigned int innerStride,
                                Real* zs, int* vs) {

    for (int i=0; i<num; ++i) {

        Real* fIn = &in[i*outerStride];
        Real* fOut = &out[i*outerStride];

        Real* z = zs + i*(len+1);
        int* v = vs + i*len;

        int k = 0;
        v[0] = 0;
        z[0] = -INF;
        z[1] = +INF;
        for (int q = 1; q <= len-1; q++) {
            Real s  = ((fIn[q*innerStride]+q*q)-(fIn[v[k]*innerStride]+v[k]*v[k]))/(2*q-2*v[k]);
            while (s <= z[k]) {
                k--;
                s  = ((fIn[q*innerStride]+q*q)-(fIn[v[k]*innerStride]+v[k]*v[k]))/(2*q-2*v[k]);
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

}

template <typename Real, bool takeSqrt>
void doublyStridedDistanceTransform1D(Real* in, Real* out, const unsigned int len, const unsigned int maxA, const unsigned int maxB,
                                      const unsigned int outerStrideA, const unsigned int outerStrideB, const unsigned int innerStride,
                                      Real* zs, int* vs) {

    for (int b=0; b<maxB; ++b) {
        for (int a=0; a<maxA; ++a) {

            Real* fIn = &in[a*outerStrideA + b*outerStrideB];
            Real* fOut = &out[a*outerStrideA + b*outerStrideB];

            Real* z = zs + (a + b*maxA)*(len+1);
            int* v = vs + (a + b*maxA)*len;

            int k = 0;
            v[0] = 0;
            z[0] = -INF;
            z[1] = +INF;
            for (int q = 1; q <= len-1; q++) {
                Real s  = ((fIn[q*innerStride]+q*q)-(fIn[v[k]*innerStride]+v[k]*v[k]))/(2*q-2*v[k]);
                while (s <= z[k]) {
                    k--;
                    s  = ((fIn[q*innerStride]+q*q)-(fIn[v[k]*innerStride]+v[k]*v[k]))/(2*q-2*v[k]);
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
                    out[q*innerStride] = sqrtf((q-v[k])*(q-v[k]) + fIn[v[k]*innerStride]);
                else
                    out[q*innerStride] = (q-v[k])*(q-v[k]) + fIn[v[k]*innerStride];
            }
        }
    }

}


// exposed functions
template <typename Real, bool takeSqrt>
void distanceTransform1D(Real* in, Real* out, const unsigned int width) {

    int* vScratch = new int[width];
    Real* zScratch = new Real[width+1];

    distanceTransform1D<Real,takeSqrt>(in,out,width,zScratch,vScratch);

    delete [] vScratch;
    delete [] zScratch;

}

template <typename Real, bool takeSqrt>
void distanceTransform1D(Real* in, Real* out, const unsigned int width, Real* zScratch, int* vScratch) {

    int k = 0;
    vScratch[0] = 0;
    zScratch[0] = -INF;
    zScratch[1] = +INF;
    for (int q = 1; q <= width-1; q++) {
        Real s  = ((in[q]+q*q)-(in[vScratch[k]]+vScratch[k]*vScratch[k]))/(2*q-2*vScratch[k]);
        while (s <= zScratch[k]) {
            k--;
            s  = ((in[q]+q*q)-(in[vScratch[k]]+vScratch[k]*vScratch[k]))/(2*q-2*vScratch[k]);
        }
        k++;
        vScratch[k] = q;
        zScratch[k] = s;
        zScratch[k+1] = +INF;
    }

    k = 0;
    for (int q = 0; q <= width-1; q++) {
        while (zScratch[k+1] < q)
            k++;
        if (takeSqrt)
            out[q] = sqrtf((q-vScratch[k])*(q-vScratch[k]) + in[vScratch[k]]);
        else
            out[q] = (q-vScratch[k])*(q-vScratch[k]) + in[vScratch[k]];
    }

}

template <typename Real, bool takeSqrt>
void distanceTransform2D(Real* im, Real* scratch, const unsigned int width, const unsigned int height) {

    Real* zScratch = new Real[(width+1)*(height+1)];
    int* vScratch = new int[width*height];

    distanceTransform2D<Real,takeSqrt>(im, scratch, width, height, zScratch, vScratch);

    delete [] zScratch;
    delete [] vScratch;
}

template <typename Real, bool takeSqrt>
void distanceTransform2D(Real* im, Real* scratch, const unsigned int width, const unsigned int height, Real* zScratch, int* vScratch) {

    // x-direction
    stridedDistanceTransform1D<Real,false>(im,scratch,width,height,width,1,zScratch,vScratch);

    // y-direction
    stridedDistanceTransform1D<Real,takeSqrt>(scratch,im,height,width,1,height,zScratch,vScratch);

}

template <typename Real, bool takeSqrt>
void distanceTransform3D(Real* in, Real* out, const unsigned int width, const unsigned int height, const unsigned int depth) {

    Real* zScratch = new Real[(width+1)*(height+1)*(depth+1)];
    int* vScratch = new int[width*height*depth];

    distanceTransform3D<Real,takeSqrt>(in,out,width,height,depth,zScratch,vScratch);

    delete [] zScratch;
    delete [] vScratch;

}

template <typename Real, bool takeSqrt>
void distanceTransform3D(Real* in, Real* out, const unsigned int width, const unsigned int height, const unsigned int depth, Real* zScratch, int* vScratch) {

    // x-direction
    stridedDistanceTransform1D<Real,false>(in,out,width,height*depth,width,1,zScratch,vScratch);

    // z-direction
    stridedDistanceTransform1D<Real,false>(out,in,depth,width*height,1,width*height,zScratch,vScratch);

    // y-directions
    doublyStridedDistanceTransform1D<Real,takeSqrt>(in,out,height,width,depth,1,width*height,width,zScratch,vScratch);

}


template <typename Real, bool takeSqrt>
void signedDistanceTransform3D(Real* in, Real* out, const unsigned int width, const unsigned int height, const unsigned int depth) {

    Real* zScratch = new Real[(width+1)*(height+1)*(depth+1)];
    int* vScratch = new int[width*height*depth];

    signedDistanceTransform3D<Real,takeSqrt>(in, out, width, height, depth, zScratch, vScratch);

    delete [] zScratch;
    delete [] vScratch;

}

template <typename Real, bool takeSqrt>
void signedDistanceTransform3D(Real* in, Real* out, const unsigned int width, const unsigned int height, const unsigned int depth,
                               Real* zScratch, int* vScratch) {

    Real* imScratch = new Real[width*height*depth];

    signedDistanceTransform3D<Real,takeSqrt>(in,out,width,height,depth,zScratch,vScratch,imScratch);

    delete [] imScratch;

}

template <typename Real, bool takeSqrt>
void signedDistanceTransform3D(Real* in, Real* out, const unsigned int width, const unsigned int height, const unsigned int depth,
                               Real* zScratch, int* vScratch, Real* imScratch) {

    for (int z=0; z<depth; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {

                const int index = x + width*(y + height*z);

                if (in[index] == 0) {
                    out[index] = INF;
                    if (x > 0) {
                        if (in[index-1] != 0.0) {
                            out[index] = 0.0;
                            continue;
                        }
                    }
                    if (x < width - 1 ) {
                        if (in[index+1] != 0.0) {
                            out[index] = 0.0;
                            continue;
                        }
                    }
                    if (y > 0) {
                        if (in[index-width] != 0.0) {
                            out[index] = 0.0;
                            continue;
                        }
                    }
                    if (y < height - 1) {
                        if (in[index+width] != 0.0) {
                            out[index] = 0.0;
                            continue;
                        }
                    }
                    if (z > 0) {
                        if (in[index-width*height] != 0.0) {
                            out[index] = 0.0;
                            continue;
                        }
                    }
                    if (z < depth - 1) {
                        if (in[index+width*height] != 0.0) {
                            out[index] = 0.0;
                            continue;
                        }
                    }
                }
                else {
                    out[index] = 0;
                }

            }
        }
    }

    distanceTransform3D<Real,takeSqrt>(out,imScratch,width,height,depth,zScratch,vScratch);
    distanceTransform3D<Real,takeSqrt>(in,out,width,height,depth,zScratch,vScratch);

    for (int i=0; i<width*height*depth; i++) {
        if (in[i] == 0)
            out[i] = -imScratch[i];
    }

}

#define DECLARE_DISTANCE_TRANSFORM_1D(REAL,TAKE_SQRT) \
    template void distanceTransform1D<REAL,TAKE_SQRT>(REAL* in, REAL* out, const unsigned int width); \
    template void distanceTransform1D<REAL,TAKE_SQRT>(REAL* in, REAL* out, const unsigned int width, REAL* zScratch, int* vScratch);
#define DECLARE_DISTANCE_TRANSFORM_2D(REAL,TAKE_SQRT) \
    template void distanceTransform2D<REAL,TAKE_SQRT>(REAL* in, REAL* out, const unsigned int width, const unsigned int height); \
    template void distanceTransform2D<REAL,TAKE_SQRT>(REAL* in, REAL* out, const unsigned int width, const unsigned int height, REAL* zScratch, int* vScratch);
#define DECLARE_DISTANCE_TRANSFORM_3D(REAL,TAKE_SQRT) \
    template void distanceTransform3D<REAL,TAKE_SQRT>(REAL* in, REAL* out, const unsigned int width, const unsigned int height, const unsigned int depth); \
    template void distanceTransform3D<REAL,TAKE_SQRT>(REAL* in, REAL* out, const unsigned int width, const unsigned int height, const unsigned int depth, REAL* zScratch, int* vScratch);
#define DECLARE_SIGNED_DISTANCE_TRANSFORM_3D(REAL,TAKE_SQRT) \
    template void signedDistanceTransform3D<REAL,TAKE_SQRT>(REAL* in, REAL* out, const unsigned int width, const unsigned int height, const unsigned int depth); \
    template void signedDistanceTransform3D<REAL,TAKE_SQRT>(REAL* in, REAL* out, const unsigned int width, const unsigned int height, const unsigned int depth, REAL* zScratch, int* vScratch); \
    template void signedDistanceTransform3D<REAL,TAKE_SQRT>(REAL* in, REAL* out, const unsigned int width, const unsigned int height, const unsigned int depth, REAL* zScratch, int* vScratch, REAL* imScratch);
#define DECLARE_ALL_DISTANCE_TRANSFORMS(REAL,TAKE_SQRT) \
    DECLARE_DISTANCE_TRANSFORM_1D(REAL,TAKE_SQRT) \
    DECLARE_DISTANCE_TRANSFORM_2D(REAL,TAKE_SQRT) \
    DECLARE_DISTANCE_TRANSFORM_3D(REAL,TAKE_SQRT) \
    DECLARE_SIGNED_DISTANCE_TRANSFORM_3D(REAL,TAKE_SQRT)

DECLARE_ALL_DISTANCE_TRANSFORMS(float,false)
DECLARE_ALL_DISTANCE_TRANSFORMS(float,true)
DECLARE_ALL_DISTANCE_TRANSFORMS(double,false)
DECLARE_ALL_DISTANCE_TRANSFORMS(double,true)

}
