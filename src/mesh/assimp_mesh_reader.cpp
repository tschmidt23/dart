#include "assimp_mesh_reader.h"

#include <iostream>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <vector_types.h>
#include <vector_functions.h>

namespace dart {

dart::Mesh * AssimpMeshReader::readMesh(const std::string filename) const {

    const struct aiScene * scene = aiImportFile(filename.c_str(),0); //aiProcess_JoinIdenticalVertices);
    if (scene == 0) {
        std::cerr << "error: " << aiGetErrorString() << std::endl;
        return 0;
    }

    if (scene->mNumMeshes != 1) {
        std::cerr << "there are " << scene->mNumMeshes << " meshes in " << filename << std::endl;
        aiReleaseImport(scene);
        return 0;
    }

    aiMesh * aiMesh = scene->mMeshes[0];

    dart::Mesh * mesh = new dart::Mesh(aiMesh->mNumVertices,aiMesh->mNumFaces);

    for (int f=0; f<mesh->nFaces; ++f) {
        aiFace& face = aiMesh->mFaces[f];
        if (face.mNumIndices != 3) {
            std::cerr << filename << " is not a triangle mesh" << std::endl;
            delete mesh;
            aiReleaseImport(scene);
            return 0;
        }
        mesh->faces[f] = make_int3(face.mIndices[0],face.mIndices[1],face.mIndices[2]);
    }
    for (int v=0; v<mesh->nVertices; ++v) {
        aiVector3D & vertex = aiMesh->mVertices[v];
        mesh->vertices[v] = make_float3(vertex.x,vertex.y,vertex.z);
        aiVector3D & normal = aiMesh->mNormals[v];
        mesh->normals[v] = make_float3(normal.x,normal.y,normal.z);
    }

    aiReleaseImport(scene);

    return mesh;

}

}
