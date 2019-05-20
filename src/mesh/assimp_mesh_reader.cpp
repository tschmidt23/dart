#include "assimp_mesh_reader.h"

#include <iostream>
#include <assimp/cimport.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <assert.h>

namespace dart {

template<typename T>
size_t vec_byte_size(const typename std::vector<T>& vec) {
    return sizeof(T) * vec.size();
}

aiMatrix4x4 getFullT(const std::vector<aiMatrix4x4> &transforms) {
    // identity transform
    aiMatrix4x4 T;
    // concatenate all transformations
    for(auto t : transforms)
        T *= t;
    return T;
}

void getMesh(const aiScene* const scene, const aiNode* const node,
             std::vector<float3> &verts, std::vector<float3> &normals, std::vector<int3> &faces, std::vector<aiMatrix4x4> transforms)
{
    // start to count mesh ID from here
    const unsigned int offset = verts.size();

    // we need to skip the transformation from the scene to the root frame
    if(node->mParent != NULL)
        transforms.push_back(node->mTransformation);

    const aiMatrix4x4 T = getFullT(transforms);

    for(unsigned int m=0; m<node->mNumMeshes; m++) {
        const aiMesh* const mesh = scene->mMeshes[node->mMeshes[m]];
        if(mesh->mPrimitiveTypes == aiPrimitiveType_TRIANGLE) {
            // transform each vertice
            for(unsigned int v=0; v<mesh->mNumVertices; v++) {
                const aiVector3D vert = T*mesh->mVertices[v];
                verts.push_back(make_float3(vert.x, vert.y, vert.z));
            }

            // transform each normal
            for(unsigned int v=0; v<mesh->mNumVertices && mesh->HasNormals(); v++) {
                const aiVector3D norm = T*mesh->mNormals[v];
                normals.push_back(make_float3(norm.x, norm.y, norm.z));
            }

            // get faces and store vertex ID with offsets for each new mesh
            for(unsigned int f=0; f<mesh->mNumFaces; f++) {
                const aiFace face = mesh->mFaces[f];
                assert(face.mNumIndices == 3);
                faces.push_back(make_int3(offset + face.mIndices[0],
                                          offset + face.mIndices[1],
                                          offset + face.mIndices[2]));
            }
        } // check aiPrimitiveType_TRIANGLE
    }

    // recursively for each child node
    for(unsigned int i=0; i<node->mNumChildren; i++) {
        const aiNode* const child = node->mChildren[i];
        getMesh(scene, child, verts, normals, faces, transforms);
    }
}

dart::Mesh * AssimpMeshReader::readMesh(const std::string filename) const {

    // define import flags for assimp
    const unsigned int import_flags = aiProcess_Triangulate |
                                      aiProcess_SortByPType |
                                      aiProcess_JoinIdenticalVertices;
    // aiProcess_Triangulate: triangulate ploygones such that all vertices have 3 points
    // aiProcess_SortByPType: split meshes with mixed types into separate meshes
    // aiProcess_JoinIdenticalVertices: remove vertices with same coordinates

    const struct aiScene * scene = aiImportFile(filename.c_str(), import_flags);

    if (scene == 0) {
        std::cerr << "error: " << aiGetErrorString() << std::endl;
        aiReleaseImport(scene);
        return 0;
    }

    std::vector<int3> faces;
    std::vector<float3> verts;
    std::vector<float3> normals;
    std::vector<aiMatrix4x4> transforms;

    // recursively obtain vertices, normals and faces from meshes
    const aiNode* const rnode = scene->mRootNode;
    getMesh(scene, rnode, verts, normals, faces, transforms);

    // copy mesh  data to DART mesh
    dart::Mesh* mesh = new dart::Mesh(verts.size(), faces.size());
    memcpy(mesh->vertices, verts.data(), vec_byte_size(verts));
    memcpy(mesh->normals, normals.data(), vec_byte_size(normals));
    memcpy(mesh->faces, faces.data(), vec_byte_size(faces));

    aiReleaseImport(scene);

    return mesh;
}

}
