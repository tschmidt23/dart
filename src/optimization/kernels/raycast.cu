//#include "raycast.h"

#include <vector>
#include <math.h>
#include "geometry/grid_3d.h"
#include "geometry/SE3.h"
#include "util/mirrored_memory.h"

namespace dart {

inline __device__ __host__ bool intersectBox(float3 rayDir, float3 origin, float3 boxmin, float3 boxmax, float2 * t) {
    //double3 rayDir = make_double3(r.x,r.y,r.z);

    // compute intersection of ray with all six bbox planes
    //trying a new fix for axis =0.
    float3 invR = 1.0/rayDir;//rayDir / (rayDir*rayDir);
    //float3 invR = make_float3(invR_.x,invR_.y,invR_.z);
    float3 tbot = invR * (boxmin - origin);
    float3 ttop = invR * (boxmax - origin);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmax(fmax(tmin.x, tmin.y), tmin.z);
    float smallest_tmax = fmin(fmin(tmax.x, tmax.y),  tmax.z);
    //float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    //float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    t->x = largest_tmin;
    t->y = smallest_tmax;
    return   smallest_tmax > largest_tmin;

}

// -=-=-=-=-=-=-=-=-=- kernels -=-=-=-=-=-=-=-=-=-
__global__ void gpu_raycastSDF(float2 fl, float2 pp,
                               const int width, const int height,
                               const SE3 T_mc,
                               const SE3 * T_fms,
                               const SE3 * T_mfs,
                               const int * sdfFrames,
                               const Grid3D<float> * sdfs,
                               const int nSdfs,
                               float4 * points,
                               const float levelSet) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    const SE3 T_cm = dart::SE3Invert(T_mc);

    if (x < width && y < height) {

        const float2 invfl = 1.0f/(fl);
        const float2 uv = make_float2(x + 0.5,y + 0.5);
        const float3 origin = make_float3(0,0,0);
        const float3 raydir = normalize( make_float3( (uv-pp)*invfl,1) );

        float closestT = 10000;
        for (int s=0; s < nSdfs; ++s) {

            const int f = sdfFrames[s];
            const Grid3D<float>& sdf = sdfs[s];

            float2 t;
            float3 origin_f = make_float3(T_fms[f]*T_mc*make_float4(origin,1));
            float3 raydir_f = make_float3(T_fms[f]*T_mc*make_float4(raydir,0));
            float3 raydirNorm_f = normalize(raydir_f);

            float3 sdfMin = sdf.offset;
            float3 sdfMax = sdf.offset + sdf.resolution*make_float3(sdf.dim.x,sdf.dim.y,sdf.dim.z);

            bool intersects = intersectBox(raydirNorm_f,origin_f, sdfMin, sdfMax, &t);

            if (intersects) {

                float3 x_f = raydirNorm_f*t.x + origin_f;
                float3 x_g = fmaxf(make_float3(1,1,1), fminf((x_f - sdf.offset) / sdf.resolution, make_float3(sdf.dim.x-2,sdf.dim.y-2,sdf.dim.z-2)));

                float sdfVal = sdf.getValueInterpolated(x_g)*sdf.resolution;
                float sdfValPrev = sdfVal;

                float tPrev = t.x;

                while (t.x < t.y && t.x < closestT) {

                    x_f = raydirNorm_f*t.x + origin_f;
                    x_g = fmaxf(make_float3(1,1,1) , fminf((x_f - sdf.offset) / sdf.resolution, make_float3(sdf.dim.x-2,sdf.dim.y-2,sdf.dim.z-2)));

                    sdfVal = sdf.getValueInterpolated(x_g)*sdf.resolution;

                    if (sdfValPrev > levelSet && sdfVal <= levelSet) {

                        const float tHit = (t.x - tPrev)*(levelSet - sdfValPrev)/(sdfVal - sdfValPrev) + tPrev;

                        float3 xHit_f = raydirNorm_f*tHit + origin_f;
                        float4 xHit_c = T_cm*T_mfs[f]*make_float4(xHit_f,1);

                        points[x + y*width] = xHit_c;

                        closestT = tHit;

                    }

                    sdfValPrev = sdfVal;
                    tPrev = t.x;
                    t.x += fmax(sdf.resolution / 10, fabs(sdfVal));

                }

            }

        }

    }

}

__global__ void gpu_raycastPrediction(float2 fl,
                                      float2 pp,
                                      const int width,
                                      const int height,
                                      const int modelNum,
                                      const SE3 T_mc,
                                      const SE3 * T_fms,
                                      const SE3 * T_mfs,
                                      const int * sdfFrames,
                                      const Grid3D<float> * sdfs,
                                      const int nSdfs,
                                      float4 * prediction,
                                      const float levelSet) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) { return; }

    const int index = x + y*width;
    if (modelNum == 0) { prediction[index].z = 0; prediction[index].w = -1; }

    const SE3 T_cm = dart::SE3Invert(T_mc);

    if (x < width && y < height) {

        const float2 invfl = 1.0f/(fl);
        const float2 uv = make_float2(x + 0.5,y + 0.5);
        const float3 origin = make_float3(0,0,0);
        const float3 raydir = normalize( make_float3( (uv-pp)*invfl,1) );

        float closestT = 10000;
        for (int s=0; s < nSdfs; ++s) {

            const int f = sdfFrames[s];
            const Grid3D<float>& sdf = sdfs[s];

            float2 t;
            float3 origin_f = make_float3(T_fms[f]*T_mc*make_float4(origin,1));
            float3 raydir_f = make_float3(T_fms[f]*T_mc*make_float4(raydir,0));
            float3 raydirNorm_f = normalize(raydir_f);

            float3 sdfMin = sdf.offset;
            float3 sdfMax = sdf.offset + sdf.resolution*make_float3(sdf.dim.x,sdf.dim.y,sdf.dim.z);

            bool intersects = intersectBox(raydirNorm_f,origin_f, sdfMin, sdfMax, &t);

            if (intersects) {

                float3 x_f = raydirNorm_f*t.x + origin_f;
                float3 x_g = fmaxf(make_float3(1,1,1), fminf((x_f - sdf.offset) / sdf.resolution, make_float3(sdf.dim.x-2,sdf.dim.y-2,sdf.dim.z-2)));

                float sdfVal = sdf.getValueInterpolated(x_g)*sdf.resolution;
                float sdfValPrev = sdfVal;

                float tPrev = t.x;

                while (t.x < t.y && t.x < closestT) {

                    x_f = raydirNorm_f*t.x + origin_f;
                    x_g = fmaxf(make_float3(1,1,1) , fminf((x_f - sdf.offset) / sdf.resolution, make_float3(sdf.dim.x-2,sdf.dim.y-2,sdf.dim.z-2)));

                    sdfVal = sdf.getValueInterpolated(x_g)*sdf.resolution;

                    if (sdfValPrev > levelSet && sdfVal <= levelSet) {

                        const float tHit = (t.x - tPrev)*(levelSet - sdfValPrev)/(sdfVal - sdfValPrev) + tPrev;

                        float3 xHit_f = raydirNorm_f*tHit + origin_f;
                        float4 xHit_c = T_cm*T_mfs[f]*make_float4(xHit_f,1);

                        float currentZ = prediction[index].z;
                        if (currentZ == 0 || xHit_c.z < currentZ) {
                            int id = (modelNum << 16) + s;
                            xHit_c.w = id;
                            prediction[index] = xHit_c;
                        }

                        closestT = tHit;

                    }

                    sdfValPrev = sdfVal;
                    tPrev = t.x;
                    t.x += fmax(sdf.resolution / 10, fabs(sdfVal));

                }

            }

        }

    }

}


__global__ void gpu_raycastPredictionDebug(float2 fl,
                                           float2 pp,
                                           const int width,
                                           const int height,
                                           const int modelNum,
                                           const SE3 T_mc,
                                           const SE3 * T_fms,
                                           const SE3  * T_mfs,
                                           const int* sdfFrames,
                                           const Grid3D<float> * sdfs,
                                           const int nSdfs,
                                           float4 * prediction,
                                           const float levelSet,
                                           unsigned char * boxIntersections) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) { return; }

    const int index = x + y*width;
    if (modelNum == 0) { prediction[index].z = 0; boxIntersections[index] = 0; }

    const SE3 T_cm = dart::SE3Invert(T_mc);

    if (x < width && y < height) {

        const float2 invfl = 1.0f/(fl);
        const float2 uv = make_float2(x + 0.5,y + 0.5);
        const float3 origin = make_float3(0,0,0);
        const float3 raydir = normalize( make_float3( (uv-pp)*invfl,1) );

        float closestT = 10000;
        for (int s=0; s < nSdfs; ++s) {

            const int f = sdfFrames[s];
            const Grid3D<float>& sdf = sdfs[s];

            float2 t;
            float3 origin_f = make_float3(T_fms[f]*T_mc*make_float4(origin,1));
            float3 raydir_f = make_float3(T_fms[f]*T_mc*make_float4(raydir,0));
            float3 raydirNorm_f = normalize(raydir_f);

            float3 sdfMin = sdf.offset;
            float3 sdfMax = sdf.offset + sdf.resolution*make_float3(sdf.dim.x,sdf.dim.y,sdf.dim.z);

            bool intersects = intersectBox(raydirNorm_f,origin_f, sdfMin, sdfMax, &t);

            if (intersects) {

                boxIntersections[index] = 1;

                float3 x_f = raydirNorm_f*t.x + origin_f;
                float3 x_g = fmaxf(make_float3(1,1,1), fminf((x_f - sdf.offset) / sdf.resolution, make_float3(sdf.dim.x-2,sdf.dim.y-2,sdf.dim.z-2)));

                float sdfVal = sdf.getValueInterpolated(x_g)*sdf.resolution;
                float sdfValPrev = sdfVal;

                float tPrev = t.x;

                while (t.x < t.y && t.x < closestT) {

                    x_f = raydirNorm_f*t.x + origin_f;
                    x_g = fmaxf(make_float3(1,1,1) , fminf((x_f - sdf.offset) / sdf.resolution, make_float3(sdf.dim.x-2,sdf.dim.y-2,sdf.dim.z-2)));

                    sdfVal = sdf.getValueInterpolated(x_g)*sdf.resolution;

                    if (sdfValPrev > levelSet && sdfVal <= levelSet) {

                        const float tHit = (t.x - tPrev)*(levelSet - sdfValPrev)/(sdfVal - sdfValPrev) + tPrev;

                        float3 xHit_f = raydirNorm_f*tHit + origin_f;
                        float4 xHit_c = T_cm*T_mfs[f]*make_float4(xHit_f,1);

                        float currentZ = prediction[index].z;
                        if (currentZ == 0 || xHit_c.z < currentZ) {
                            int id = (modelNum << 16) + s;
                            xHit_c.w = id;
                            prediction[index] = xHit_c;
                        }

                        closestT = tHit;

                    }

                    sdfValPrev = sdfVal;
                    tPrev = t.x;
                    t.x += fmax(sdf.resolution / 10, fabs(sdfVal));

                }

            }

        }

    }

}

__global__ void gpu_raycastPredictionDebugRay(float2 fl,
                                              float2 pp,
                                              const int x,
                                              const int y,
                                              const int width,
                                              const int modelNum,
                                              const SE3 T_mc,
                                              const SE3 * T_fms,
                                              const SE3 * T_mfs,
                                              const int * sdfFrames,
                                              const Grid3D<float> * sdfs,
                                              const int nSdfs,
                                              float4 * prediction,
                                              const float levelSet,
                                              float3 * boxIntersects,
                                              float2 * raySteps,
                                              const int maxRaySteps)
{

    const int index = x + y*width;
    if (modelNum == 0) { prediction[index].z = 0; }

    printf("-=-=-=-=-=-=-=-=-\nmodel %d\n-=-=-=-=-=-=-=-=-\n",modelNum);

    const SE3 T_cm = dart::SE3Invert(T_mc);

    const float2 invfl = 1.0f/(fl);
    const float2 uv = make_float2(x + 0.5,y + 0.5);
    const float3 origin = make_float3(0,0,0);
    const float3 raydir = normalize( make_float3( (uv-pp)*invfl,1) );

    int step = 0;

    float closestT = 10000;
    for (int s=0; s < nSdfs; ++s) {

        printf("sdf %d\n",s);

        const int f = sdfFrames[s];
        const Grid3D<float>& sdf = sdfs[s];

        float2 t;
        float3 origin_f = make_float3(T_fms[f]*T_mc*make_float4(origin,1));
        float3 raydir_f = make_float3(T_fms[f]*T_mc*make_float4(raydir,0));
        float3 raydirNorm_f = normalize(raydir_f);

        float3 sdfMin = sdf.offset;
        float3 sdfMax = sdf.offset + sdf.resolution*make_float3(sdf.dim.x,sdf.dim.y,sdf.dim.z);

        bool intersects = intersectBox(raydirNorm_f,origin_f, sdfMin, sdfMax, &t);

        if (intersects) {

            boxIntersects[s] = make_float3(t.x,t.y,1);

            printf("\tsdf is intersected\n");

            float3 x_f = raydirNorm_f*t.x + origin_f;
            float3 x_g = fmaxf(make_float3(1,1,1), fminf((x_f - sdf.offset) / sdf.resolution, make_float3(sdf.dim.x-2,sdf.dim.y-2,sdf.dim.z-2)));

            float sdfVal = sdf.getValueInterpolated(x_g)*sdf.resolution;
            float sdfValPrev = sdfVal;

            float tPrev = t.x;

            while (t.x < t.y && t.x < closestT) {


                x_f = raydirNorm_f*t.x + origin_f;
                x_g = fmaxf(make_float3(1,1,1) , fminf((x_f - sdf.offset) / sdf.resolution, make_float3(sdf.dim.x-2,sdf.dim.y-2,sdf.dim.z-2)));

                sdfVal = sdf.getValueInterpolated(x_g)*sdf.resolution;

                if (step < maxRaySteps) {
                    raySteps[step] = make_float2(t.x,sdfVal);
                    ++step;
                }

                printf("\tt.x=%f\n",t.x);
                printf("\tsdfVal=%f\n",sdfVal);

                if (sdfValPrev > levelSet && sdfVal <= levelSet) {

                    printf("\tcrossed\n");

                    const float tHit = (t.x - tPrev)*(levelSet - sdfValPrev)/(sdfVal - sdfValPrev) + tPrev;

                    float3 xHit_f = raydirNorm_f*tHit + origin_f;
                    float4 xHit_c = T_cm*T_mfs[f]*make_float4(xHit_f,1);

                    float currentZ = prediction[index].z;
                    if (currentZ == 0 || xHit_c.z < currentZ) {
                        int id = (modelNum << 16) + s;
                        xHit_c.w = id;
                        prediction[index] = xHit_c;
                    }

                    closestT = tHit;

                    printf("\tclosestT now %f\n",closestT);

                }

                sdfValPrev = sdfVal;
                tPrev = t.x;
                t.x += fmax(sdf.resolution / 10, fabs(sdfVal));

            }

        }

        else {
            boxIntersects[s] = make_float3(5,5,5);
        }


    }

    printf("took %d steps\n",step);

}

// -=-=-=-=-=-=-=-=-=- interface -=-=-=-=-=-=-=-=-=-
void raycastPrediction(float2 fl,
                       float2 pp,
                       const int width,
                       const int height,
                       const int modelNum,
                       const SE3 T_mc,
                       const SE3 * T_fms,
                       const SE3 * T_mfs,
                       const int * sdfFrames,
                       const Grid3D<float> * sdfs,
                       const int nSdfs,
                       float4 * prediction,
                       const float levelSet,
                       cudaStream_t stream) {

    dim3 block = dim3(8,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_raycastPrediction<<<grid,block,0,stream>>>(fl,pp,width,height,modelNum,
                                                   T_mc,T_fms,T_mfs,sdfFrames,sdfs,nSdfs,
                                                   prediction, levelSet);

//    gpu_raycastPredictionDebug<<<grid,block>>>(fl,pp,width,height,modelNum,
//                                          T_mc,T_fms,T_mfs,sdfFrames,sdfs,nSdfs,
//                                          prediction, levelSet, boxIntersect);

}

void raycastPredictionDebugRay(float2 fl,
                               float2 pp,
                               const int x,
                               const int y,
                               const int width,
                               const int modelNum,
                               const SE3 T_mc,
                               const SE3 * T_fms,
                               const SE3 * T_mfs,
                               const int * sdfFrames,
                               const Grid3D<float> * sdfs,
                               const int nSdfs,
                               float4 * prediction,
                               const float levelSet,
                               float3 * boxIntersects,
                               float2 * raySteps,
                               const int maxRaySteps) {

    dim3 block(1,1,1);
    dim3 grid(1,1,1);

    gpu_raycastPredictionDebugRay<<<grid,block>>>(fl,pp,x,y,width,modelNum,T_mc,T_fms,T_mfs,sdfFrames,sdfs,nSdfs,prediction,levelSet,boxIntersects,raySteps,maxRaySteps);

}

}
