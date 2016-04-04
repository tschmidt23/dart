#ifndef MESH_SAMPLE_H
#define MESH_SAMPLE_H

#include <vector>
#include "mesh.h"

namespace dart {

void sampleMesh(std::vector<float3> & sampledPoints, const Mesh & mesh, const float sampleDensity);

}

#endif // MESH_SAMPLE_H
