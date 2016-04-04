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

    // generate the buffers
    glGenBuffersARB(NumPrimitives,_primitiveVBOs);
    glGenBuffersARB(NumPrimitives,_primitiveNBOs);
    glGenBuffersARB(NumPrimitives,_primitiveIBOs);

    // copy data to buffers
    for (int i=0; i<NumPrimitives; ++i) {

        Mesh * primitive = primitiveMeshes[i];

        glBindBufferARB(GL_ARRAY_BUFFER_ARB,_primitiveVBOs[i]);
        glBufferDataARB(GL_ARRAY_BUFFER_ARB,primitive->nVertices*sizeof(float3),primitive->vertices,GL_STATIC_DRAW_ARB);

        glBindBufferARB(GL_ARRAY_BUFFER_ARB,_primitiveNBOs[i]);
        glBufferDataARB(GL_ARRAY_BUFFER_ARB,primitive->nVertices*sizeof(float3),primitive->normals,GL_STATIC_DRAW_ARB);

        glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,_primitiveIBOs[i]);
        glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB,primitive->nFaces*sizeof(int3),primitive->faces,GL_STATIC_DRAW_ARB);

        _nPrimitiveFaces[i] = primitive->nFaces;

        delete primitive;

    }

    // initialize mesh reader
    _meshReader = meshReader;

}

ModelRenderer::~ModelRenderer() {

    // free buffers
    glDeleteBuffersARB(NumPrimitives,_primitiveVBOs);
    glDeleteBuffersARB(NumPrimitives,_primitiveNBOs);
    glDeleteBuffersARB(NumPrimitives,_primitiveIBOs);

    if (_meshNumbers.size() > 0) {
        glDeleteBuffersARB(_meshNumbers.size(),_meshVBOs.data());
        glDeleteBuffersARB(_meshNumbers.size(),_meshNBOs.data());
        glDeleteBuffersARB(_meshNumbers.size(),_meshIBOs.data());
    }

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

    _meshVBOs.resize(meshNum + 1);
    _meshNBOs.resize(meshNum + 1);
    _meshIBOs.resize(meshNum + 1);

    glGenBuffersARB(1,&_meshVBOs[meshNum]);
    glGenBuffersARB(1,&_meshNBOs[meshNum]);
    glGenBuffersARB(1,&_meshIBOs[meshNum]);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB,_meshVBOs[meshNum]);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB,mesh->nVertices*sizeof(float3),mesh->vertices,GL_STATIC_DRAW_ARB);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB,_meshNBOs[meshNum]);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB,mesh->nVertices*sizeof(float3),mesh->normals,GL_STATIC_DRAW_ARB);

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,_meshIBOs[meshNum]);
    glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB,mesh->nFaces*sizeof(int3),mesh->faces,GL_STATIC_DRAW_ARB);

    _nMeshFaces.push_back(mesh->nFaces);

    _meshes.push_back(mesh);

    return meshNum;

}

void ModelRenderer::renderPrimitive(const GeomType type) const {

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _primitiveNBOs[type]);
    glNormalPointer(GL_FLOAT,0,0);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _primitiveVBOs[type]);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, _primitiveIBOs[type]);

    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

    glDrawElements(GL_TRIANGLES,3*_nPrimitiveFaces[type],GL_UNSIGNED_INT,0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB,0);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,0);

}

void ModelRenderer::renderMesh(const uint meshNum) const {

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _meshNBOs[meshNum]);
    glNormalPointer(GL_FLOAT,0,0);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _meshVBOs[meshNum]);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, _meshIBOs[meshNum]);

    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

    glDrawElements(GL_TRIANGLES,3*_nMeshFaces[meshNum],GL_UNSIGNED_INT,0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB,0);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,0);

}

const Mesh & ModelRenderer::getMesh(const uint meshNum) const {

    return *_meshes[meshNum];

}

}
