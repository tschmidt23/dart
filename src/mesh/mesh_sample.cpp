#include "mesh_sample.h"

#include <algorithm>
#include <cstdlib>

#include <iostream>

#include <vector_functions.h>
#include <helper_math.h>

namespace dart {

void sampleMesh(std::vector<float3> & sampledPoints, const Mesh & mesh, const float sampleDensity) {

    // compute cumulative and total surface area
    std::vector<double> cumulativeSurfaceArea(mesh.nFaces);
    for (int f=0; f<mesh.nFaces; ++f) {

        const int3 & face = mesh.faces[f];
        const float3 & A = mesh.vertices[face.x];
        const float3 & B = mesh.vertices[face.y];
        const float3 & C = mesh.vertices[face.z];

        double a = length(A - B);
        double b = length(B - C);
        double c = length(C - A);

        double s = (a + b + c)/2;

        double surfaceArea = sqrt(s*(s-a)*(s-b)*(s-c));

        if (std::isnan(surfaceArea)) {
            surfaceArea = 0;
        }

        if (f == 0) { cumulativeSurfaceArea[f] = surfaceArea; }
        else { cumulativeSurfaceArea[f] = cumulativeSurfaceArea[f-1] + surfaceArea; }

    }

    double & totalSurfaceArea = cumulativeSurfaceArea.back();

    const int nSamplePoints = round(totalSurfaceArea*sampleDensity);

    sampledPoints.resize(nSamplePoints);

    // sample points
    for (int p=0; p<nSamplePoints; ++p) {

        // pick a face
        double r0 = rand()*totalSurfaceArea/RAND_MAX;
        int f = std::lower_bound(cumulativeSurfaceArea.begin(),cumulativeSurfaceArea.end(),r0) - cumulativeSurfaceArea.begin();

        int3 & face = mesh.faces[f];
        float * A = (float*)&mesh.vertices[face.x];
        float * B = (float*)&mesh.vertices[face.y];
        float * C = (float*)&mesh.vertices[face.z];
        float * sample = (float*)&sampledPoints[p];

        // pick a point
        float r1 = rand() / (float) RAND_MAX;
        float r2 = rand() / (float) RAND_MAX;

        for (int i=0; i<3; ++i) {
            sample[i] = (1-sqrtf(r1))*A[i] +
                        sqrtf(r1)*(1-r2)*B[i] +
                        sqrtf(r1)*r2*C[i];
        }

    }

}

}
