#ifndef MODEL_RENDERER_H
#define MODEL_RENDERER_H

#include <map>
#include <string>
#include <vector>
#include <sys/types.h>
#include <GL/glew.h>

#include "mesh/mesh.h"
#include "dart_types.h"
//#include "models.h"

namespace dart {

class MeshReader {
public:
    virtual Mesh * readMesh(const std::string filename) const = 0;
};

class ModelRenderer {
public:
    ModelRenderer(MeshReader * meshReader);
    ~ModelRenderer();

    /**
    Returns the mesh number for a given mesh. If the mesh has not been seen before, a new mesh number is
    assigned and the mesh is loaded into memory.
    @param meshFilename The name of the file in which the mesh is stored.
    @return The unique mesh identifier if successful, otherwise -1.
    */
    int getMeshNumber(const std::string meshFilename);

    void renderPrimitive(const GeomType type) const;

    /**
    Renders a mesh that is already loaded into memory.
    @param meshNum The unique identifier associated with the mesh.
    @see getMeshNumber()
    */
    void renderMesh(const uint meshNum) const;

    const Mesh & getMesh(const uint meshNum) const;

private:

    // primitive data
    GLuint _primitiveVBOs[NumPrimitives];
    GLuint _primitiveNBOs[NumPrimitives];
    GLuint _primitiveIBOs[NumPrimitives];
    int _nPrimitiveFaces[NumPrimitives];

    // mesh data
    std::map<std::string,uint> _meshNumbers;
    std::vector<GLuint> _meshVBOs;
    std::vector<GLuint> _meshNBOs;
    std::vector<GLuint> _meshIBOs;
    std::vector<int> _nMeshFaces;
    std::vector<Mesh *> _meshes;

    // mesh loader
    MeshReader * _meshReader;
};

}

#endif // MODEL_RENDERER_H
