#include "model_renderer.h"

#include <iostream>
#include "mesh/primitive_meshing.h"
#include <GL/glx.h>

namespace dart {

static const int cylinderSlices = 20;
static const int sphereSplits = 2;

ModelRenderer::ModelRenderer(MeshReader * meshReader) {

    // check if a GLContext is available
    GLXContext glxContext = glXGetCurrentContext();
    if (glxContext == NULL) {
        std::cerr << "gl context is null; please intitialize a gl context before instantiation of dart::Tracker" << std::endl;
    }

    // generate the primitive meshes
    Mesh * primitiveMeshes[NumPrimitives];
    primitiveMeshes[PrimitiveSphereType] = generateUnitIcosphereMesh(sphereSplits);
    primitiveMeshes[PrimitiveCylinderType] = generateCylinderMesh(cylinderSlices);
    primitiveMeshes[PrimitiveCubeType] = generateCubeMesh();

    // copy data to buffers
    for (int i=0; i<NumPrimitives; ++i) {

        Mesh * primitive = primitiveMeshes[i];

        _primitiveVBOs[i].Reinitialise(pangolin::GlArrayBuffer,primitive->nVertices,GL_FLOAT,3,GL_STATIC_DRAW);
        _primitiveVBOs[i].Upload(primitive->vertices,primitive->nVertices*sizeof(float3));

        _primitiveNBOs[i].Reinitialise(pangolin::GlArrayBuffer,primitive->nVertices,GL_FLOAT,3,GL_STATIC_DRAW);
        _primitiveNBOs[i].Upload(primitive->vertices,primitive->nVertices*sizeof(float3));

        _primitiveIBOs[i].Reinitialise(pangolin::GlElementArrayBuffer,primitive->nFaces*3,GL_UNSIGNED_INT,3,GL_STATIC_DRAW);
        _primitiveIBOs[i].Upload(primitive->faces,primitive->nFaces*sizeof(int3));

        _nPrimitiveFaces[i] = primitive->nFaces;

        delete primitive;

    }

    // initialize mesh reader
    _meshReader = meshReader;

}

ModelRenderer::~ModelRenderer() {

    // free meshes
    for (int i=0; i<_meshes.size(); ++i) {
        delete _meshes[i];
    }

    // free mesh reader
    delete _meshReader;
}

int ModelRenderer::getMeshNumber(const std::string meshFilename) {

    if (_meshReader == 0) {
        std::cerr << "mesh reader was not set; therefore, meshes may not be read." << std::endl;
        return -1;
    }

    if (_meshNumbers.find(meshFilename) != _meshNumbers.end()) {
        return _meshNumbers[meshFilename];
    }

    Mesh * mesh = _meshReader->readMesh(meshFilename);

    if (mesh == 0) {
        std::cerr << "could not read " + meshFilename << std::endl;
        return -1;
    }

    int meshNum = _meshNumbers.size();
    _meshNumbers[meshFilename] = meshNum;

    _meshVBOs.emplace_back(pangolin::GlArrayBuffer, mesh->nVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    _meshVBOs.back().Upload(mesh->vertices,mesh->nVertices*sizeof(float3));

    _meshNBOs.emplace_back(pangolin::GlArrayBuffer, mesh->nVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    _meshNBOs.back().Upload(mesh->normals,mesh->nVertices*sizeof(float3));

    _meshIBOs.emplace_back(pangolin::GlElementArrayBuffer, mesh->nFaces*3, GL_UNSIGNED_INT, 3, GL_STATIC_DRAW);
    _meshIBOs.back().Upload(mesh->faces,mesh->nFaces*sizeof(int3));

    _nMeshFaces.push_back(mesh->nFaces);

    _meshes.push_back(mesh);

    return meshNum;

}

void ModelRenderer::renderPrimitive(const GeomType type) const {

    _primitiveVBOs[type].Bind();
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    _primitiveNBOs[type].Bind();
    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_FLOAT,0,0);

    _primitiveIBOs[type].Bind();
    glDrawElements(GL_TRIANGLES,3*_nPrimitiveFaces[type],GL_UNSIGNED_INT,0);

    _primitiveNBOs[type].Unbind();
    _primitiveVBOs[type].Unbind();
    _primitiveIBOs[type].Unbind();

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

}

void ModelRenderer::renderMesh(const uint meshNum) const {

    _meshNBOs[meshNum].Bind();
    glNormalPointer(GL_FLOAT,0,0);

    _meshVBOs[meshNum].Bind();
    glVertexPointer(3, GL_FLOAT, 0, 0);

    _meshIBOs[meshNum].Bind();

    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

    glDrawElements(GL_TRIANGLES,3*_nMeshFaces[meshNum],GL_UNSIGNED_INT,0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    _meshNBOs[meshNum].Unbind();
    _meshVBOs[meshNum].Unbind();
    _meshIBOs[meshNum].Unbind();

}

const Mesh & ModelRenderer::getMesh(const uint meshNum) const {

    return *_meshes[meshNum];

}

}
