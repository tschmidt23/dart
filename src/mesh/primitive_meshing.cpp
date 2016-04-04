#include "primitive_meshing.h"

#include <string.h>
#include <map>
#include <vector>
#include <stdlib.h>
#include <math.h>

#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

namespace dart {

Mesh * generateUnitIcosphereMesh(const int splits) {

    std::vector<float3> * vertVec = new std::vector<float3>();
    std::vector<int3> * faceVec = new std::vector<int3>();

    // generate initial 12 vertices
    float t = (1.0f + sqrtf(5.0f)) / 2.0f;

    vertVec->push_back(normalize(make_float3(-1, t, 0)));
    vertVec->push_back(normalize(make_float3( 1, t, 0)));
    vertVec->push_back(normalize(make_float3(-1,-1, 0)));
    vertVec->push_back(normalize(make_float3( 1,-t, 0)));

    vertVec->push_back(normalize(make_float3( 0,-1, t)));
    vertVec->push_back(normalize(make_float3( 0, 1, t)));
    vertVec->push_back(normalize(make_float3( 0,-1,-t)));
    vertVec->push_back(normalize(make_float3( 0, 1,-t)));

    vertVec->push_back(normalize(make_float3( t, 0,-1)));
    vertVec->push_back(normalize(make_float3( t, 0, 1)));
    vertVec->push_back(normalize(make_float3(-t, 0,-1)));
    vertVec->push_back(normalize(make_float3(-t, 0, 1)));

    // generate intitial 20 faces
    faceVec->push_back(make_int3( 0,11, 5));
    faceVec->push_back(make_int3( 0, 5, 1));
    faceVec->push_back(make_int3( 0, 1, 7));
    faceVec->push_back(make_int3( 0, 7,10));
    faceVec->push_back(make_int3( 0,10,11));

    faceVec->push_back(make_int3( 1, 5, 9));
    faceVec->push_back(make_int3( 5,11, 4));
    faceVec->push_back(make_int3(11,10, 2));
    faceVec->push_back(make_int3(10, 7, 6));
    faceVec->push_back(make_int3( 7, 1, 8));

    faceVec->push_back(make_int3( 3, 9, 4));
    faceVec->push_back(make_int3( 3, 4, 2));
    faceVec->push_back(make_int3( 3, 2, 6));
    faceVec->push_back(make_int3( 3, 6, 8));
    faceVec->push_back(make_int3( 3, 8, 9));

    faceVec->push_back(make_int3( 4, 9, 5));
    faceVec->push_back(make_int3( 2, 4,11));
    faceVec->push_back(make_int3( 6, 2,10));
    faceVec->push_back(make_int3( 8, 6, 7));
    faceVec->push_back(make_int3( 9, 8, 1));

    // map of already split vertices
    std::map<int64_t,int> splitVerts;

    for (int i=0; i<splits; ++i) {
        std::vector<int3> * newFaces = new std::vector<int3>();

        for (unsigned int f = 0; f < faceVec->size(); ++f) {

            const int v1 = faceVec->at(f).x;
            const int v2 = faceVec->at(f).y;
            const int v3 = faceVec->at(f).z;
            int64_t key;
            std::map<int64_t,int>::iterator it;
            int p12, p23, p31;

            // edge 12
            key = (v1 < v2) ? (((int64_t)v1 << 32) | v2) : (((int64_t)v2 << 32) | v1);
            it = splitVerts.find(key);
            if (it != splitVerts.end()) { // check if we've already split this edge
                p12 = it->second;
            }
            else {
                p12 = vertVec->size();
                splitVerts[key] = p12;
                vertVec->push_back(normalize(vertVec->at(v1) + vertVec->at(v2)));
            }

            // edge 23
            key = (v2 < v3) ? (((int64_t)v2 << 32) | v3) : (((int64_t)v3 << 32) | v2);
            it = splitVerts.find(key);
            if (it != splitVerts.end()) { // check if we've already split this edge
                p23 = it->second;
            }
            else {
                p23 = vertVec->size();
                splitVerts[key] = p23;
                vertVec->push_back(normalize(vertVec->at(v2) + vertVec->at(v3)));
            }

            // edge 31
            key = (v3 < v1) ? (((int64_t)v3 << 32) | v1) : (((int64_t)v1 << 32) | v3);
            it = splitVerts.find(key);
            if (it != splitVerts.end()) { // check if we've already split this edge
                p31 = it->second;
            }
            else {
                p31 = vertVec->size();
                splitVerts[key] = p31;
                vertVec->push_back(normalize(vertVec->at(v3) + vertVec->at(v1)));
            }

            // add new faces
            newFaces->push_back(make_int3(v1,p12,p31));
            newFaces->push_back(make_int3(v2,p23,p12));
            newFaces->push_back(make_int3(v3,p31,p23));
            newFaces->push_back(make_int3(p12,p23,p31));

        }

        delete faceVec;
        faceVec = newFaces;
    }

    // construct mesh
    Mesh * mesh = new Mesh(vertVec->size(),faceVec->size());
    memcpy(mesh->vertices,vertVec->data(),vertVec->size()*sizeof(float3));
    memcpy(mesh->normals,vertVec->data(),vertVec->size()*sizeof(float3));
    memcpy(mesh->faces,faceVec->data(),faceVec->size()*sizeof(int3));

    // memory cleanup
    delete vertVec;
    delete faceVec;

    return mesh;

}

Mesh * generateCylinderMesh(const int slices) {

    const int nVertices = 4*slices + 2;
    const int nFaces = 4*slices;

    Mesh * mesh = new Mesh(nVertices,nFaces);

    // generate vertices and normals
    for (int i=0; i<slices; ++i) {
        const float angle = i*2.0*M_PI/slices;
        mesh->vertices[i + 0*slices] = make_float3(cos(angle),sin(angle),0.0f);
        mesh->normals [i + 0*slices] = mesh->vertices[i];
        mesh->vertices[i + 1*slices] = make_float3(cos(angle),sin(angle),1.0f);
        mesh->normals [i + 1*slices] = mesh->vertices[i];
        mesh->vertices[i + 2*slices] = mesh->vertices[i + 0*slices];
        mesh->normals [i + 2*slices] = make_float3(0,0,-1);
        mesh->vertices[i + 3*slices] = mesh->vertices[i + 1*slices];
        mesh->normals [i + 3*slices] = make_float3(0,0,1);
    }
    mesh->vertices[4*slices  ] = make_float3(0,0,0);
    mesh->normals [4*slices  ] = make_float3(0,0,-1);
    mesh->vertices[4*slices+1] = make_float3(0,0,1);
    mesh->normals [4*slices+1] = make_float3(0,0,1);

    // generate indices for the sides
    for (int i=0; i<slices-1; ++i) {
        mesh->faces[2*i  ] = make_int3(i,i+1,i+slices+1);
        mesh->faces[2*i+1] = make_int3(i,i+slices+1,i+slices);
    }
    mesh->faces[2*(slices-1)  ] = make_int3(slices-1,0,slices);
    mesh->faces[2*(slices-1)+1] = make_int3(slices-1,slices,2*slices-1);

    // generate indices for the end caps
    for (int i=0; i<slices-1; ++i) {
        mesh->faces[2*slices + 2*i  ] = make_int3(2*slices + i,     2*slices+i+1,4*slices  );
        mesh->faces[2*slices + 2*i+1] = make_int3(3*slices + i + 1, 3*slices+i,    4*slices+1);
    }
    mesh->faces[2*slices+2*(slices-1)  ] = make_int3(2*slices + slices-1,2*slices,4*slices);
    mesh->faces[2*slices+2*(slices-1)+1] = make_int3(3*slices, 3*slices+slices-1,4*slices+1);

    return mesh;
}

Mesh * generateCubeMesh() {

    Mesh * mesh = new Mesh(24,12);

    // initialize normals
    for (int n=0; n<4; ++n) {
        mesh->normals[ 0 + n] = make_float3( 0, 0, 1);
        mesh->normals[ 4 + n] = make_float3( 0, 0,-1);
        mesh->normals[ 8 + n] = make_float3( 0, 1, 0);
        mesh->normals[12 + n] = make_float3( 0,-1, 0);
        mesh->normals[16 + n] = make_float3( 1, 0, 0);
        mesh->normals[20 + n] = make_float3(-1, 0, 0);
    }

    // initialize faces
    for (int f=0; f<6; ++f) {
        mesh->faces[2*f  ] = make_int3(4*f, 4*f+1, 4*f+2);
        mesh->faces[2*f+1] = make_int3(4*f, 4*f+2, 4*f+3);
    }

    // initialize vertices
    mesh->vertices[ 0] = make_float3( 0.5, 0.5, 0.5);
    mesh->vertices[ 1] = make_float3( 0.5,-0.5, 0.5);
    mesh->vertices[ 2] = make_float3(-0.5,-0.5, 0.5);
    mesh->vertices[ 3] = make_float3(-0.5, 0.5, 0.5);

    mesh->vertices[ 4] = make_float3( 0.5, 0.5,-0.5);
    mesh->vertices[ 5] = make_float3( 0.5,-0.5,-0.5);
    mesh->vertices[ 6] = make_float3(-0.5,-0.5,-0.5);
    mesh->vertices[ 7] = make_float3(-0.5, 0.5,-0.5);

    mesh->vertices[ 8] = make_float3( 0.5, 0.5, 0.5);
    mesh->vertices[ 9] = make_float3( 0.5, 0.5,-0.5);
    mesh->vertices[10] = make_float3(-0.5, 0.5,-0.5);
    mesh->vertices[11] = make_float3(-0.5, 0.5, 0.5);

    mesh->vertices[12] = make_float3( 0.5,-0.5, 0.5);
    mesh->vertices[13] = make_float3( 0.5,-0.5,-0.5);
    mesh->vertices[14] = make_float3(-0.5,-0.5,-0.5);
    mesh->vertices[15] = make_float3(-0.5,-0.5, 0.5);

    mesh->vertices[16] = make_float3( 0.5, 0.5, 0.5);
    mesh->vertices[17] = make_float3( 0.5, 0.5,-0.5);
    mesh->vertices[18] = make_float3( 0.5,-0.5,-0.5);
    mesh->vertices[19] = make_float3( 0.5,-0.5, 0.5);

    mesh->vertices[20] = make_float3(-0.5, 0.5, 0.5);
    mesh->vertices[21] = make_float3(-0.5, 0.5,-0.5);
    mesh->vertices[22] = make_float3(-0.5,-0.5,-0.5);
    mesh->vertices[23] = make_float3(-0.5,-0.5, 0.5);

    return mesh;

}

}
