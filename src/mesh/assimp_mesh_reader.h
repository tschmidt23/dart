#ifndef ASSIMP_MESH_READER_H
#define ASSIMP_MESH_READER_H

#include "util/model_renderer.h"

namespace dart {

class AssimpMeshReader : public dart::MeshReader {
public:
    dart::Mesh * readMesh(const std::string filename) const;
};

}

#endif // ASSIMP_MESH_READER_H
