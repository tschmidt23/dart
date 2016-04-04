#include "sdf.h"
#include "geometry/geometry.h"

namespace dart {

void projectToSdfSurface(const Grid3D<float> & sdf, float3 & pointGrid, const float threshold, const int maxIters) {

    int iter = 0;
    for (iter; iter < maxIters; ++iter) {

        float dist = sdf.getValueInterpolated(pointGrid);
        if (abs(dist) < threshold) {
            //std::cout << "reached threshold in " << iter << " iters" << std::endl;
            return;
        }

        float3 grad = sdf.getGradientInterpolated(pointGrid);
        pointGrid = pointGrid - dist*normalize(grad);

    }

}

void analyticMeshSdf(Grid3D<float> & sdf, const Mesh & mesh) {

    if (mesh.nFaces > sdf.dim.x*sdf.dim.y*sdf.dim.z) {
        for (int z=0; z<sdf.dim.z; ++z) {
            for (int y=0; y<sdf.dim.y; ++y) {
                for (int x=0; x<sdf.dim.x; ++x) {

                    float & minDist = sdf.data[(z*sdf.dim.y + y)*sdf.dim.x + x];

                    float3 c = (make_float3(x,y,z) + sdf.offset)*sdf.resolution;
                    for (int f=0; f<mesh.nFaces; ++f) {
                        const int3 & face = mesh.faces[f];
                        const float3 & A = mesh.vertices[face.x];
                        const float3 & B = mesh.vertices[face.y];
                        const float3 & C = mesh.vertices[face.z];
                        float dist = distancePointTriangle(c,A,B,C);
                        if ( dist < fabs(minDist) ) {
//                            float3 unscaledNorm = cross(A-B,C-B);
//                            if (dot(c-B,unscaledNorm) < 0) { dist = -dist; }
                            minDist = dist;
                        }
                    }

                }
            }
        }
    } else {

        for (int f=0; f<mesh.nFaces; ++f) {
            if ( f % 100 == 0) { std::cout << f << " / " << mesh.nFaces << std::endl; }
            const int3 & face = mesh.faces[f];
            const float3 & A = mesh.vertices[face.x];
            const float3 & B = mesh.vertices[face.y];
            const float3 & C = mesh.vertices[face.z];

#pragma omp parallel for
            for (int z=0; z<sdf.dim.z; ++z) {
                for (int y=0; y<sdf.dim.y; ++y) {
                    for (int x=0; x<sdf.dim.x; ++x) {

                        float & minDist = sdf.data[(z*sdf.dim.y + y)*sdf.dim.x + x];
                        float3 c = sdf.getWorldCoords(make_float3(x+0.5,y+0.5,z+0.5));

                        float dist = distancePointTriangle(c,A,B,C);
                        if ( dist < fabs(minDist) ) {
//                            float3 unscaledNorm = cross(A-B,C-B);
//                            if (dot(B - c,unscaledNorm) < 0) { dist = -dist; }
                            minDist = dist;
                        }

                    }
                }
            }
        }

    }

}

void analyticBoxSdf(Grid3D<float> & sdf, SE3 T_bg, const float3 boxMin, const float3 boxMax) {

    for (int z=0; z<sdf.dim.z; ++z) {
        for (int y=0; y<sdf.dim.y; ++y) {
            for (int x=0; x<sdf.dim.x; ++x) {

                float & minDist = sdf.data[(z*sdf.dim.y + y)*sdf.dim.x + x];

                const float3 center_g = sdf.getWorldCoords(make_float3(x,y,z) + make_float3(0.5,0.5,0.5));
                const float3 center_b = SE3Transform(T_bg,center_g);

                float dist;
                if ( center_b.x < boxMin.x ) {
                    if (center_b.y < boxMin.y) {
                        if (center_b.z < boxMin.z) { // ---
                            dist = length(center_b - boxMin);
                        } else if (center_b.z > boxMax.z) { // --+
                            dist = length( center_b - make_float3(boxMin.x,boxMin.y,boxMax.z));
                        } else { // --=
                            dist = length(make_float2(center_b.x,center_b.y) - make_float2(boxMin.x,boxMin.y));
                        }
                    } else if (center_b.y > boxMax.y) {
                        if (center_b.z < boxMin.z) { // -+-
                            dist = length( center_b - make_float3(boxMin.x,boxMax.y,boxMin.z));
                        } else if (center_b.z > boxMax.z) { // -++
                            dist = length( center_b - make_float3(boxMin.x,boxMax.y,boxMax.z));
                        } else { // -+=
                            dist = length(make_float2(center_b.x,center_b.y) - make_float2(boxMin.x,boxMax.y));
                        }
                    } else {
                        if (center_b.z < boxMin.z) { // -=-
                            dist = length( make_float2(center_b.x,center_b.z) - make_float2(boxMin.x,boxMin.z));
                        } else if (center_b.z > boxMax.z) { // -=+
                            dist = length( make_float2(center_b.x,center_b.z) - make_float2(boxMin.x,boxMax.z));
                        } else { // -==
                            dist = boxMin.x - center_b.x;
                        }
                    }
                } else if (center_b.x > boxMax.x) {
                    if (center_b.y < boxMin.y) {
                        if (center_b.z < boxMin.z) { // +--
                            dist = length( center_b - make_float3(boxMax.x,boxMin.y,boxMin.z));
                        } else if (center_b.z > boxMax.z) { // +-+
                            dist = length( center_b - make_float3(boxMax.x,boxMin.y,boxMax.z));
                        } else { // +-=
                            dist = length(make_float2(center_b.x,center_b.y) - make_float2(boxMax.x,boxMin.y));
                        }
                    } else if (center_b.y > boxMax.y) {
                        if (center_b.z < boxMin.z) { // ++-
                            dist = length( center_b - make_float3(boxMax.x,boxMax.y,boxMin.z));
                        } else if (center_b.z > boxMax.z) { // +++
                            dist = length( center_b - boxMax);
                        } else { // ++=
                            dist = length(make_float2(center_b.x,center_b.y) - make_float2(boxMax.x,boxMax.y));
                        }
                    } else {
                        if (center_b.z < boxMin.z) { // +=-
                            dist = length( make_float2(center_b.x,center_b.z) - make_float2(boxMax.x,boxMin.z));
                        } else if (center_b.z > boxMax.z) { // +=+
                            dist = length( make_float2(center_b.x,center_b.z) - make_float2(boxMax.x,boxMax.z));
                        } else { // +==
                            dist = center_b.x - boxMax.x;
                        }
                    }
                } else {
                    if (center_b.y < boxMin.y) {
                        if (center_b.z < boxMin.z) { // =--
                            dist = length( make_float2(center_b.y,center_b.z) - make_float2(boxMin.y,boxMin.z));
                        } else if (center_b.z > boxMax.z) { // =-+
                            dist = length( make_float2(center_b.y,center_b.z) - make_float2(boxMin.y,boxMax.z));
                        } else { // =-=
                            dist = boxMin.y - center_b.y;
                        }
                    } else if (center_b.y > boxMax.y) {
                        if (center_b.z < boxMin.z) { // =+-
                            dist = length( make_float2(center_b.y,center_b.z) - make_float2(boxMax.y,boxMin.z));
                        } else if (center_b.z > boxMax.z) { // =++
                            dist = length( make_float2(center_b.y,center_b.z) - make_float2(boxMax.y,boxMax.z));
                        } else { // =+=
                            dist = center_b.y - boxMax.y;
                        }
                    } else {
                        if (center_b.z < boxMin.z) { // ==-
                            dist = boxMin.z - center_b.z;
                        } else if (center_b.z > boxMax.z) { // ==+
                            dist = center_b.z - boxMax.z;
                        } else { // ===
                            const float3 dMin = fabs(center_b - boxMin);
                            const float3 dMax = fabs(center_b - boxMax);
                            dist = -std::min(std::min(dMin.x,dMax.x),std::min(std::min(dMin.y,dMax.y),std::min(dMin.z,dMax.z)));
                        }
                    }
                }
                dist /= sdf.resolution;

                if (fabs(dist) < fabs(minDist)) {
                    minDist = dist;
                }

//                const float3 dMin = fabs(center_b - boxMin);
//                const float3 dMax = fabs(center_b - boxMax);

//                const float dist = std::min(std::min(dMin.x,dMax.x),std::min(std::min(dMin.y,dMax.y),std::min(dMin.z,dMax.z)));

//                if (dist > fabs(minDist)) { continue; }

//                if (center_b.x > boxMin.x && center_b.x < boxMax.x &&
//                    center_b.y > boxMin.y && center_b.y < boxMax.y &&
//                    center_b.z > boxMin.z && center_b.z < boxMax.z) {
//                    minDist = -dist;
//                } else {
//                    minDist = dist;
//                }

            }
        }
    }

}

void analyticSphereSdf(Grid3D<float> & sdf, SE3 T_sg, const float sphereRadius) {

    for (int z=0; z<sdf.dim.z; ++z) {
        for (int y=0; y<sdf.dim.y; ++y) {
            for (int x=0; x<sdf.dim.x; ++x) {

                float & minDist = sdf.data[(z*sdf.dim.y + y)*sdf.dim.x + x];

                const float3 center_g = sdf.getWorldCoords(make_float3(x,y,z) + make_float3(0.5,0.5,0.5));
                const float3 center_s = SE3Transform(T_sg,center_g);

                const float dist = (length(center_s) - sphereRadius) / sdf.resolution;

                if (fabs(dist) < fabs(minDist)) {
                    minDist = dist;
                }

            }
        }
    }

}



}
