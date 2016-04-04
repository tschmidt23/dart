#ifndef MESH_H
#define MESH_H

#include <string.h>
#include <vector_types.h>

namespace dart {

class Mesh {
public:

    /**
    * The default mesh constructor.
    * All data members are initialized to null pointers, and will need to be allocated later.
    */
    Mesh() : faces(0), vertices(0), normals(0), nVertices(0), nFaces(0) { }

    /**
    * Constructs a mesh with preallocated mesh data.
    * The arrays for vertices, normals, and faces are allocated to the appropriate size, but the values are undefined.
    * @param numVertices The number of vertices in the mesh.
    * @param numFaces The number of faces in the mesh.
    */
    Mesh(const int numVertices, const int numFaces) :
        faces(new int3[numFaces]),
        vertices(new float3[numVertices]),
        normals(new float3[numVertices]),
        nVertices(numVertices),
        nFaces(numFaces) { }

    Mesh(const Mesh & copy) :
        faces(new int3[copy.nFaces]),
        vertices(new float3[copy.nVertices]),
        normals(new float3[copy.nVertices]),
        nVertices(copy.nVertices),
        nFaces(copy.nFaces) {
        memcpy(faces,copy.faces,copy.nFaces*sizeof(int3));
        memcpy(vertices,copy.vertices,copy.nVertices*sizeof(float3));
        memcpy(normals,copy.normals,copy.nVertices*sizeof(float3));
    }

    ~Mesh() { delete faces; delete vertices; delete normals; }

    void writeToObjFile(const char * filename);

    /**
    * The face data array. Each entry defines a face consisting of three indices into the vertex array.
    */
    int3 * faces;

    /**
    * The vertex data array. Each entry defines a vertex.
    */
    float3 * vertices;

    /**
    * The normal data array. Each entry defines the normal of the corresponding entry in the vertex data array.
    */
    float3 * normals;

    /**
    * The number of vertices in the mesh.
    */
    int nVertices;

    /**
    * The number of faces in the mesh.
    */
    int nFaces;
};

}

#endif // MESH_H
