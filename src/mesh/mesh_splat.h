#ifndef MESH_SPLAT_H
#define MESH_SPLAT_H

#include "mesh.h"
#include "geometry/grid_3d.h"

namespace dart {

void splatSolidMesh(const Mesh & mesh, Grid3D<float> & sdf);

}

#endif // MESH_SPLAT_H
