#ifndef SDF_H
#define SDF_H

#include "geometry/SE3.h"
#include "grid_3d.h"
#include "mesh/mesh.h"

namespace dart {

void projectToSdfSurface(const Grid3D<float> & sdf, float3 & pointGrid, const float threshold = 1e-5, const int maxIters = 100);

void analyticMeshSdf(Grid3D<float> & sdf, const Mesh & mesh);

void analyticBoxSdf(Grid3D<float> & sdf, SE3 T_bg, const float3 boxMin, const float3 boxMax);

void analyticSphereSdf(Grid3D<float> & sdf, SE3 T_sg, const float sphereRadius);

}

#endif // SDF_H
