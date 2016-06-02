#include "assimp_mesh_reader.h"

#include <iostream>
#include <assimp/cimport.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <vector_types.h>
#include <vector_functions.h>

namespace dart {

dart::Mesh * AssimpMeshReader::readMesh(const std::string filename) const {

    // define import flags for assimp
    const unsigned int import_flags = aiProcess_Triangulate | aiProcess_SortByPType;
    // aiProcess_Triangulate: triangulate ploygones such that all vertices have 3 points
    // aiProcess_SortByPType: split meshes with mixed types into separate meshes

    const struct aiScene * scene = aiImportFile(filename.c_str(), import_flags); //aiProcess_JoinIdenticalVertices);

    if (scene == 0) {
        std::cerr << "error: " << aiGetErrorString() << std::endl;
        aiReleaseImport(scene);
        return 0;
    }

    // find first triangle mesh
    aiMesh * aiMesh = NULL;
    // stop searching if end of list or if first mesh is found (aiMesh!=NULL)
    for(unsigned int i = 0; i<scene->mNumMeshes && aiMesh==NULL; i++) {
        if(scene->mMeshes[i]->mPrimitiveTypes == aiPrimitiveType_TRIANGLE)
            aiMesh = scene->mMeshes[i];
        else
            std::cout<<"ignoring non-triangle mesh in "<<filename<<std::endl;
    }

    dart::Mesh * mesh = NULL;
    if(aiMesh != NULL) {
        mesh = new dart::Mesh(aiMesh->mNumVertices,aiMesh->mNumFaces);

        for (int f=0; f<mesh->nFaces; ++f) {
            aiFace& face = aiMesh->mFaces[f];
            mesh->faces[f] = make_int3(face.mIndices[0],face.mIndices[1],face.mIndices[2]);
        }
        for (int v=0; v<mesh->nVertices; ++v) {
            aiVector3D & vertex = aiMesh->mVertices[v];
            mesh->vertices[v] = make_float3(vertex.x,vertex.y,vertex.z);
            aiVector3D & normal = aiMesh->mNormals[v];
            mesh->normals[v] = make_float3(normal.x,normal.y,normal.z);
        }
    } // if aiMesh!=NULL
    else {
        std::cerr << "there are no triangle meshes in " << filename << std::endl;
    }

    aiReleaseImport(scene);

    return mesh;

}

}
