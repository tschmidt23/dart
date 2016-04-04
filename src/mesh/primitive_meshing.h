#ifndef PRIMITIVE_MESHING_H
#define PRIMITIVE_MESHING_H

#include "mesh.h"

namespace dart {

Mesh * generateUnitIcosphereMesh(const int splits);

Mesh * generateCylinderMesh(const int slices);

Mesh * generateCubeMesh();

}

#endif // PRIMITIVE_MESHING_H
