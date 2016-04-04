#include "mesh_proc.h"

#include <iostream>
#include <map>
#include <set>
#include <stack>
#include <vector>

#include <string.h>

#include <vector_functions.h>

namespace dart {

void removeDuplicatedVertices(Mesh & mesh) {
    const int originalVerts = mesh.nVertices;
    std::vector<float3> newVerts;
    std::vector<float3> newNorms;
    std::map<int,int> vertMapping;
    for (int i=0; i<mesh.nVertices; ++i) {
        int v;
        for (v=0; v<newVerts.size(); ++v) {
            if (mesh.vertices[i].x == newVerts[v].x && mesh.vertices[i].y == newVerts[v].y && mesh.vertices[i].z == newVerts[v].z) {
                vertMapping[i] = v;
                break;
            }
        }
        if (v == newVerts.size()) {
            vertMapping[i] = newVerts.size();
            newVerts.push_back(mesh.vertices[i]);
            newNorms.push_back(mesh.normals[i]);
        }
    }
    for (int f=0; f<mesh.nFaces; ++f) {
        mesh.faces[f].x = vertMapping[mesh.faces[f].x];
        mesh.faces[f].y = vertMapping[mesh.faces[f].y];
        mesh.faces[f].z = vertMapping[mesh.faces[f].z];
    }
    mesh.nVertices = newVerts.size();
    delete [] mesh.vertices;
    delete [] mesh.normals;
    mesh.vertices = new float3[mesh.nVertices];
    mesh.normals = new float3[mesh.nVertices];
    memcpy(mesh.vertices,newVerts.data(),mesh.nVertices*sizeof(float3));
    memcpy(mesh.normals,newNorms.data(),mesh.nVertices*sizeof(float3));
    std::cout << originalVerts << " vertices mapped down to " << newVerts.size() << std::endl;
}

void removeDegenerateFaces(Mesh & mesh) {

    const int originalFaces = mesh.nFaces;

    std::vector<int3> nondegenerateFaces;
    for (int f=0; f<mesh.nFaces; ++f) {
        const int3 & face = mesh.faces[f];
        if (face.x != face.y && face.y != face.z && face.z != face.x) {
            nondegenerateFaces.push_back(face);
        }
    }

    delete [] mesh.faces;
    mesh.nFaces = nondegenerateFaces.size();
    mesh.faces = new int3[mesh.nFaces];
    memcpy(mesh.faces,nondegenerateFaces.data(),mesh.nFaces*sizeof(int3));
    std::cout << "removed " << (originalFaces-mesh.nFaces) << " faces" << std::endl;

}

void removeDisconnectedComponent(Mesh & src, Mesh & split, int startV) {

    // build map from verts to faces
    std::vector<std::vector<int> > vertFaces(src.nVertices);
    for (int i=0; i<src.nFaces; ++i) {
        const int3 & f = src.faces[i];
        vertFaces[f.x].push_back(i);
        vertFaces[f.y].push_back(i);
        vertFaces[f.z].push_back(i);
    }

    std::set<int> disconnectedVerts;
    std::set<int> disconnectedFaces;
    disconnectedVerts.insert(startV);

    std::stack<int> toExpandList;
    toExpandList.push(startV);
    while (toExpandList.size() > 0) {
        int toExpand = toExpandList.top();
//        std::cout << "expanding " << toExpand << std::endl;
        toExpandList.pop();
        std::vector<int> & faces = vertFaces[toExpand];
        for (int i=0; i<faces.size(); ++i) {
            disconnectedFaces.insert(faces[i]);
            const int3 & f = src.faces[faces[i]];
            if (disconnectedVerts.find(f.x) == disconnectedVerts.end()) { disconnectedVerts.insert(f.x); toExpandList.push(f.x); }
            if (disconnectedVerts.find(f.y) == disconnectedVerts.end()) { disconnectedVerts.insert(f.y); toExpandList.push(f.y); }
            if (disconnectedVerts.find(f.z) == disconnectedVerts.end()) { disconnectedVerts.insert(f.z); toExpandList.push(f.z); }
        }
    }

    std::cout << "found connected component of " << disconnectedVerts.size() << "/" << src.nVertices << " vertices" << std::endl;

    {
        split.nVertices = disconnectedVerts.size();
        split.nFaces = disconnectedFaces.size();

        split.vertices = new float3[split.nVertices];
        split.normals = new float3[split.nVertices];
        split.faces = new int3[split.nFaces];

        std::map<int,int> vertMapping;
        int i = 0;
        for (std::set<int>::iterator it = disconnectedVerts.begin(); it != disconnectedVerts.end(); ++it) {
            int v = *it;
            vertMapping[v] = i;
            split.vertices[i] = src.vertices[v];
            ++i;
        }

        i = 0;
        for (std::set<int>::iterator it = disconnectedFaces.begin(); it != disconnectedFaces.end(); ++it) {
            int3 &face = src.faces[*it];
            split.faces[i] = make_int3(vertMapping[face.x],vertMapping[face.y],vertMapping[face.z]);
            ++i;
        }
    }

    std::vector<float3> vertsLeft;
    std::vector<int> vertMapping(src.nVertices,-1);

    for (int i=0; i<src.nVertices; ++i) {
        if (disconnectedVerts.find(i) == disconnectedVerts.end()) {
            vertMapping[i] = vertsLeft.size();
            vertsLeft.push_back(src.vertices[i]);
        }
    }

    std::vector<int3> facesLeft;
    for (int f=0; f<src.nFaces; ++f) {
        int x = src.faces[f].x;
        int y = src.faces[f].y;
        int z = src.faces[f].z;

        if (vertMapping[x] >= 0 && vertMapping[y] >= 0 && vertMapping[z] >= 0) {
            facesLeft.push_back(make_int3(vertMapping[x],vertMapping[y],vertMapping[z]));
        }
    }

    src.nVertices = vertsLeft.size();
    src.nFaces = facesLeft.size();

    delete [] src.vertices;
    delete [] src.normals;
    delete [] src.faces;
    src.vertices = new float3[vertsLeft.size()];
    src.normals = new float3[vertsLeft.size()];
    src.faces = new int3[facesLeft.size()];

    memcpy(src.vertices,vertsLeft.data(),vertsLeft.size()*sizeof(float3));
    memcpy(src.faces,facesLeft.data(),facesLeft.size()*sizeof(int3));
}

void transformMesh(Mesh & mesh, const SE3 & transform) {

    for (int v=0; v<mesh.nVertices; ++v) {
        mesh.vertices[v] = SE3Transform(transform,mesh.vertices[v]);
        mesh.normals[v] = SE3Rotate(transform,mesh.normals[v]);
    }

}

void scaleMesh(Mesh & mesh, const float3 scale) {

    for (int v=0; v<mesh.nVertices; ++v) {
        mesh.vertices[v].x *= scale.x;
        mesh.vertices[v].y *= scale.y;
        mesh.vertices[v].z *= scale.z;
    }

}

}
