#include "mesh.h"

#include <iostream>
#include <fstream>

namespace dart {

void Mesh::writeToObjFile(const char * filename) {

    std::ofstream stream(filename);
    std::cout << "writing " << nVertices << " vertices to " << filename << std::endl;
    for (int v=0; v<nVertices; ++v) {
        stream << "v  " << vertices[v].x << " " << vertices[v].y << " " << vertices[v].z << std::endl;
        stream << "vn " << normals[v].x << " " << normals[v].y << " " << normals[v].z << std::endl;
    }
    for (int f=0; f<nFaces; ++f) {
        if (faces[f].x < 0 || faces[f].x >= nVertices || faces[f].y < 0 || faces[f].y >= nVertices || faces[f].z < 0 || faces[f].z >= nVertices) {
            std::cout << "face " << f << " has out-of-bounds vertices" << std::endl;
        }
        if (faces[f].x == faces[f].y || faces[f].y == faces[f].z || faces[f].z == faces[f].x) {
            std::cout << "face " << f << " has repeated vertices" << std::endl;
        }
        stream << "f " << faces[f].x+1 << "//" << faces[f].x+1 << " " << faces[f].y+1 << "//" << faces[f].y+1  << " " << faces[f].z+1 << "//" << faces[f].z+1 << std::endl;
    }
    stream.close();

}

}
