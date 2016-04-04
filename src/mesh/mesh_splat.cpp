#include "mesh_splat.h"

#include <list>
#include <set>
#include <string.h>

namespace dart {

// TODO: work out the bugs in this
void splatSolidMesh(const dart::Mesh & mesh, dart::Grid3D<float> & sdf) {

    // determine which faces are candidates for intersection at each voxel
    std::list<int> * candidateFaces = new std::list<int>[sdf.dim.x*sdf.dim.y*sdf.dim.z];

    for (int f = 0; f<mesh.nFaces; f++) {

        const float3& p1 = mesh.vertices[mesh.faces[f].x];
        const float3& p2 = mesh.vertices[mesh.faces[f].y];
        const float3& p3 = mesh.vertices[mesh.faces[f].z];

        float min_x = std::min(std::min(p1.x,p2.x),p3.x);
        float min_y = std::min(std::min(p1.y,p2.y),p3.y);
        float min_z = std::min(std::min(p1.z,p2.z),p3.z);

        float max_x = std::max(std::max(p1.x,p2.x),p3.x);
        float max_y = std::max(std::max(p1.y,p2.y),p3.y);
        float max_z = std::max(std::max(p1.z,p2.z),p3.z);

        if (min_x == max_x) {
            if (min_x == sdf.offset.x) {
                max_x += 1e-6;
            } else {
                min_x -= 1e-6;
            }
        }
        if (min_y == max_y) {
            if (min_y == sdf.offset.y) {
                max_y += 1e-6;
            } else {
                min_x -= 1e-6;
            }
        }
        if (min_z == max_z) {
            if (min_z == sdf.offset.z) {
                max_z += 1e-6;
            } else {
                min_z -= 1e-6;
            }
        }

        for (int z = std::max(0.0,floor((min_z-sdf.offset.z)/sdf.resolution)); z < std::min((double)sdf.dim.z,ceil((max_z-sdf.offset.z)/sdf.resolution)); z++) {
            for (int y = std::max(0.0,floor((min_y-sdf.offset.y)/sdf.resolution)); y < std::min((double)sdf.dim.y,ceil((max_y-sdf.offset.y)/sdf.resolution)); y++) {
                for (int x = std::max(0.0,floor((min_x-sdf.offset.x)/sdf.resolution)); x < std::min((double)sdf.dim.x,ceil((max_x-sdf.offset.x)/sdf.resolution)); x++) {
                    candidateFaces[x + y*sdf.dim.x + z*sdf.dim.x*sdf.dim.y].push_back(f);
                }
            }
        }

    }

    char * votes = new char[sdf.dim.x*sdf.dim.y*sdf.dim.z];
    memset(votes,0,sdf.dim.x*sdf.dim.y*sdf.dim.z*sizeof(char));

    const float halo = 1e-6;

    // cast x rays
    for (int z = 0; z < sdf.dim.z; z++) {
        for (int y = 0; y < sdf.dim.y; y++) {

            bool filling = false;
            std::set<int> intersecting_faces;

            for (int x = 0; x < sdf.dim.x; x++) {

                bool filled = filling;

                std::list<int> & candidates = candidateFaces[x + y*sdf.dim.x + z*sdf.dim.x*sdf.dim.y];

                const float3 r0 = make_float3(x*sdf.resolution + sdf.offset.x,(y+0.5)*sdf.resolution + sdf.offset.y,(z+0.5)*sdf.resolution + sdf.offset.z);
                const float3 r1 = make_float3((x+1)*sdf.resolution + sdf.offset.x,(y+0.5)*sdf.resolution + sdf.offset.y,(z+0.5)*sdf.resolution + sdf.offset.z);

                for (std::list<int>::iterator it = candidates.begin(); it != candidates.end(); it++) {

                    const int f = *it;

                    // check if we've already intersected this face
                    if (intersecting_faces.find(f) != intersecting_faces.end()) { continue; }

                    // check plane intersection
                    const float3 & p1 = mesh.vertices[mesh.faces[f].x];
                    const float3 & p2 = mesh.vertices[mesh.faces[f].y];
                    const float3 & p3 = mesh.vertices[mesh.faces[f].z];
                    const float3 u = p2-p1;
                    const float3 v = p3-p1;
                    const float3 n = cross(u,v);

                    const float denom = dot(n,(r1-r0));
                    if (denom == 0) { continue; }
                    const float r = dot(n,(p1-r0))/denom;

                    if (r < 0 || r > 1) { continue; }

                    // check triangle intersection
                    const float3 r_int = r0 + r*(r1-r0);
                    const float3 w = r_int - p1;

                    const float st_denom = dot(u,v)*dot(u,v) - dot(u,u)*dot(v,v);
                    const float s = (dot(u,v)*dot(w,v)-dot(v,v)*dot(w,u))/st_denom;
                    const float t = (dot(u,v)*dot(w,u)-dot(u,u)*dot(w,v))/st_denom;

                    if (s < -halo || t < -halo || s + t > (1 + halo)) { continue; }

                    // if it's passed all these tests, it must be intersecting
                    intersecting_faces.insert(f);
                    filling = !filling;
                    filled = true;

                }

                if (filled) {
                    votes[x + y*sdf.dim.x + z*sdf.dim.x*sdf.dim.y]++;
                }
            }

        }
    }

    // cast y rays
    for (int z = 0; z < sdf.dim.z; z++) {
        for (int x = 0; x< sdf.dim.x; x++) {

            bool filling = false;
            std::set<int> intersecting_faces;

            for (int y = 0; y < sdf.dim.y; y++) {

                bool filled = filling;

                std::list<int> &candidates = candidateFaces[x + y*sdf.dim.x + z*sdf.dim.x*sdf.dim.y];

                const float3 r0 = make_float3((x+0.5)*sdf.resolution + sdf.offset.x,y*sdf.resolution + sdf.offset.y,(z+0.5)*sdf.resolution + sdf.offset.z);
                const float3 r1 = make_float3((x+0.5)*sdf.resolution + sdf.offset.x,(y+1)*sdf.resolution + sdf.offset.y,(z+0.5)*sdf.resolution + sdf.offset.z);

                for (std::list<int>::iterator it = candidates.begin(); it != candidates.end(); it++) {

                    const int f = *it;

                    // check if we've already intersected this face
                    if (intersecting_faces.find(f) != intersecting_faces.end()) { continue; }

                    // check plane intersection
                    const float3 & p1 = mesh.vertices[mesh.faces[f].x];
                    const float3 & p2 = mesh.vertices[mesh.faces[f].y];
                    const float3 & p3 = mesh.vertices[mesh.faces[f].z];
                    const float3 u = p2-p1;
                    const float3 v = p3-p1;
                    const float3 n = cross(u,v);

                    const float denom = dot(n,(r1-r0));
                    if (denom == 0) { continue; }
                    const float r = dot(n,(p1-r0))/denom;

                    if (r < 0 || r > 1) { continue; }

                    // check triangle intersection
                    const float3 r_int = r0 + r*(r1-r0);
                    const float3 w = r_int - p1;

                    const float st_denom = dot(u,v)*dot(u,v) - dot(u,u)*dot(v,v);
                    const float s = (dot(u,v)*dot(w,v)-dot(v,v)*dot(w,u))/st_denom;
                    const float t = (dot(u,v)*dot(w,u)-dot(u,u)*dot(w,v))/st_denom;

                    if (s < -halo || t < -halo || s + t > (1 + halo)) { continue; }

                    // if it's passed all these tests, it must be intersecting
                    intersecting_faces.insert(f);
                    filling = !filling;
                    filled = true;

                }

                if (filled) {
                    votes[x + y*sdf.dim.x + z*sdf.dim.x*sdf.dim.y]++;
                }
            }

        }
    }

    // cast z rays
    for (int y = 0; y < sdf.dim.y; y++) {
        for (int x = 0; x< sdf.dim.x; x++) {

            bool filling = false;
            std::set<int> intersecting_faces;

            for (int z = 0; z < sdf.dim.z; z++) {

                bool filled = filling;

                std::list<int> & candidates = candidateFaces[x + y*sdf.dim.x + z*sdf.dim.x*sdf.dim.y];

                const float3 r0 = make_float3((x+0.5)*sdf.resolution + sdf.offset.x,(y+0.5)*sdf.resolution + sdf.offset.y,z*sdf.resolution + sdf.offset.z);
                const float3 r1 = make_float3((x+0.5)*sdf.resolution + sdf.offset.x,(y+0.5)*sdf.resolution + sdf.offset.y,(z+1)*sdf.resolution + sdf.offset.z);

                for (std::list<int>::iterator it = candidates.begin(); it != candidates.end(); it++) {

                    const int f = *it;

                    // check if we've already intersected this face
                    if (intersecting_faces.find(f) != intersecting_faces.end()) { continue; }

                    // check plane intersection
                    const float3 & p1 = mesh.vertices[mesh.faces[f].x];
                    const float3 & p2 = mesh.vertices[mesh.faces[f].y];
                    const float3 & p3 = mesh.vertices[mesh.faces[f].z];
                    const float3 u = p2-p1;
                    const float3 v = p3-p1;
                    const float3 n = cross(u,v);

                    const float denom = dot(n,(r1-r0));
                    if (denom == 0) { continue; }
                    const float r = dot(n,(p1-r0))/denom;

                    if (r < 0 || r > 1) { continue; }

                    // check triangle intersection
                    const float3 r_int = r0 + r*(r1-r0);
                    const float3 w = r_int - p1;

                    const float st_denom = dot(u,v)*dot(u,v) - dot(u,u)*dot(v,v);
                    const float s = (dot(u,v)*dot(w,v)-dot(v,v)*dot(w,u))/st_denom;
                    const float t = (dot(u,v)*dot(w,u)-dot(u,u)*dot(w,v))/st_denom;

                    if (s < -halo || t < -halo || s + t > (1 + halo)) { continue; }

                    // if it's passed all these tests, it must be intersecting
                    intersecting_faces.insert(f);
                    filling = !filling;
                    filled = true;

                }

                if (filled) {
                    votes[x + y*sdf.dim.x + z*sdf.dim.x*sdf.dim.y]++;
                }
            }

        }
    }

    for (int z = 0; z < sdf.dim.z; z++) {
        for (int y = 0; y < sdf.dim.y; y++) {
            for (int x = 0; x < sdf.dim.x; x++) {
                if (votes[x + y*sdf.dim.x + z*sdf.dim.x*sdf.dim.y] >= 2) {
                    sdf.data[x + y*sdf.dim.x + z*sdf.dim.x*sdf.dim.y] = 0;
                }
            }
        }
    }

    delete [] candidateFaces;
    delete [] votes;

}

}
