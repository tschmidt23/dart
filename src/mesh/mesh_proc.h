#ifndef MESH_PROC_H
#define MESH_PROC_H

#include "mesh.h"
#include "geometry/SE3.h"

namespace dart {

void removeDuplicatedVertices(Mesh & mesh);

void removeDegenerateFaces(Mesh & mesh);

void removeDisconnectedComponent(Mesh & src, Mesh & split, int startV);

void transformMesh(Mesh & mesh, const SE3 & transform);

void scaleMesh(Mesh & mesh, const float3 scale);

}

#endif // MESH_PROC_H
