#include "host_only_model.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>
#include <stdio.h>
#include <cstdarg>
#include <matheval.h>
#include <limits>
#include <sys/stat.h>
#include <fstream>

#include "geometry/distance_transforms.h"
#include "geometry/geometry.h"
#include "geometry/sdf.h"
#include "mesh/mesh_proc.h"
#include "mesh/mesh_splat.h"
#include "util/string_format.h"

namespace dart {

HostOnlyModel::HostOnlyModel() : Model(6) {
    _nFrames = 1;

    _parents.push_back(-1);
    _children.push_back(std::vector<int>());
    _frameGeoms.push_back(std::vector<int>());

    _T_mf.push_back(SE3());
    _T_fm.push_back(SE3());

}

HostOnlyModel::~HostOnlyModel() { }

float HostOnlyModel::evaluateExpression(const std::string & expression) {
    char * cStr = new char[expression.size() + 1];
    memcpy(cStr,expression.c_str(),expression.size());
    cStr[expression.size()] = 0;
    void * f  = evaluator_create(cStr);
    assert(f);
    char * * varNames;
    int varCount;
    evaluator_get_variables(f,&varNames,&varCount);
    double varValues[varCount];
    for (int i=0; i<varCount; ++i) {
        varValues[i] = getSizeParam(std::string(varNames[i]));
    }
    float val = evaluator_evaluate(f,varCount,varNames,varValues);

    delete [] cStr;
    evaluator_destroy(f);

    return val;
}

int HostOnlyModel::addFrame(int parent, JointType type,
                            std::string posX, std::string posY, std::string posZ,
                            std::string orX, std::string orY, std::string orZ,
                            std::string axisX, std::string axisY, std::string axisZ,
                            std::string jointMin, std::string jointMax, std::string jointName) {

    std::vector<std::string> frameExpressions(11);
    frameExpressions[0] = posX;
    frameExpressions[1] = posY;
    frameExpressions[2] = posZ;
    frameExpressions[3] = orX;
    frameExpressions[4] = orY;
    frameExpressions[5] = orZ;
    frameExpressions[6] = axisX;
    frameExpressions[7] = axisY;
    frameExpressions[8] = axisZ;
    frameExpressions[9] = jointMin;
    frameExpressions[10] = jointMax;
    _frameParamExpressions.push_back(frameExpressions);

    int frameNum = getNumFrames();

    _children.push_back(std::vector<int>());
    _parents.push_back(parent);
    _children[parent].push_back(frameNum);
    _frameGeoms.push_back(std::vector<int>());

    _jointTypes.push_back(type);

    _jointLimits.push_back(make_float2(evaluateExpression(jointMin),
                                       evaluateExpression(jointMax)));
    _jointNames.push_back(jointName);
    ++_dimensionality;

    for (int f=0; f<frameNum; ++f) {
        _dependencies.insert(_dependencies.begin() + ((f+1)*getNumJoints()-1) ,1,0);
    }
    for (int j=0; j<(getNumJoints()-1); ++j) {
        _dependencies.push_back(_dependencies[parent*getNumJoints() + j]);
    }
    _dependencies.push_back(1);

    float3 pos = make_float3(evaluateExpression(posX),
                             evaluateExpression(posY),
                             evaluateExpression(posZ));
    _positions.push_back(pos);

    float3 orientation = make_float3(evaluateExpression(orX),
                            evaluateExpression(orY),
                            evaluateExpression(orZ));
    _orientations.push_back(orientation);

    float3 axis = make_float3(evaluateExpression(axisX),
                              evaluateExpression(axisY),
                              evaluateExpression(axisZ));
    _axes.push_back(axis);

    _T_mf.push_back(SE3Fromse3(se3(pos.x,pos.y,pos.z,0,0,0))*dart::SE3Fromse3(dart::se3(0,0,0,
                                               orientation.x,orientation.y,orientation.z)));
    _T_fm.push_back(SE3Invert(_T_mf[_nFrames]));

    if (_modelVersion == 0) {
        _T_pf.push_back(SE3Fromse3(se3(pos.x, pos.y, pos.z, 0,0,0))*
                        SE3Fromse3(se3(0,0,0,orientation.x, orientation.y, orientation.z)));
    } else {
        _T_pf.push_back(SE3FromTranslation(make_float3(pos.x,pos.y,pos.z))*
                        SE3FromEuler(make_float3(orientation.z,orientation.y,orientation.x)));
    }
    _nFrames++;

    return _nFrames-1;

}

void HostOnlyModel::addGeometry(int frame, GeomType geometryType, std::string sx, std::string sy, std::string sz, std::string tx, std::string ty, std::string tz, std::string rx, std::string ry, std::string rz, unsigned char red, unsigned char green, unsigned char blue, const std::string meshFilename) {

    std::vector<std::string> geomExpressions;
    geomExpressions.push_back(sx);
    geomExpressions.push_back(sy);
    geomExpressions.push_back(sz);
    geomExpressions.push_back(tx);
    geomExpressions.push_back(ty);
    geomExpressions.push_back(tz);
    geomExpressions.push_back(rx);
    geomExpressions.push_back(ry);
    geomExpressions.push_back(rz);
    _geomParamExpressions.push_back(geomExpressions);

    addGeometry(frame, geometryType,
                evaluateExpression(sx),
                evaluateExpression(sy),
                evaluateExpression(sz),
                evaluateExpression(tx),
                evaluateExpression(ty),
                evaluateExpression(tz),
                evaluateExpression(rx),
                evaluateExpression(ry),
                evaluateExpression(rz),
                red,green,blue,
                meshFilename);

}

void HostOnlyModel::addGeometry(int frame, GeomType geometryType, float sx, float sy, float sz, float tx, float ty, float tz, float rx, float ry, float rz, unsigned char red, unsigned char green, unsigned char blue, const std::string meshFilename) {

    if (frame >= _nFrames ) { return; }

    const int geomNumber = getNumGeoms();

    // update mesh data if geometry is a mesh
    if (geometryType == MeshType) {
        if (_renderer == 0) {
            std::cerr << "model renderer has no mesh reader; mesh models may not be used" << std::endl;
            return;
        }
        uint meshNumber = _renderer->getMeshNumber(meshFilename);
        _meshFilenames[geomNumber] = meshFilename;
        setMeshNumber(geomNumber,meshNumber);
    }

    _geomTypes.push_back(geometryType);
    _geomColors.push_back(make_uchar3(red,green,blue));
    _geomScales.push_back(make_float3(sx,sy,sz));
    if (_modelVersion == 0) {
        _geomTransforms.push_back(SE3Fromse3(se3(tx,ty,tz,rx,ry,rz)));
    } else {
        _geomTransforms.push_back(SE3FromTranslation(tx,ty,tz)*SE3FromEuler(make_float3(rz,ry,rx)));
    }

    _frameGeoms[frame].push_back(geomNumber);

    // update frame color if this is the first geom
//    if (_frameGeoms[frame].size() == 1) {
        _sdfColors.push_back(make_uchar3(red,green,blue));
//    }

}

void HostOnlyModel::computeStructure() {

    for (int f=0; f<getNumFrames()-1; ++f) {
        _positions[f].x = evaluateExpression(_frameParamExpressions[f][0]);
        _positions[f].y = evaluateExpression(_frameParamExpressions[f][1]);
        _positions[f].z = evaluateExpression(_frameParamExpressions[f][2]);
        _orientations[f].x = evaluateExpression(_frameParamExpressions[f][3]);
        _orientations[f].y = evaluateExpression(_frameParamExpressions[f][4]);
        _orientations[f].z = evaluateExpression(_frameParamExpressions[f][5]);
        _axes[f].x = evaluateExpression(_frameParamExpressions[f][6]);
        _axes[f].y = evaluateExpression(_frameParamExpressions[f][7]);
        _axes[f].z = evaluateExpression(_frameParamExpressions[f][8]);
        _jointLimits[f] = make_float2(evaluateExpression(_frameParamExpressions[f][9]),
                                      evaluateExpression(_frameParamExpressions[f][10]));
    }

    for (int joint = 0; joint <getNumJoints(); ++joint) {
        if (_modelVersion == 0) {
            _T_pf[joint] = SE3Fromse3(se3(_positions[joint].x, _positions[joint].y, _positions[joint].z, 0,0,0))*
                           SE3Fromse3(se3(0,0,0,_orientations[joint].x, _orientations[joint].y, _orientations[joint].z));
        } else {
            _T_pf[joint] = SE3FromTranslation(make_float3(_positions[joint].x, _positions[joint].y, _positions[joint].z))*
                           SE3FromEuler(make_float3(_orientations[joint].z, _orientations[joint].y, _orientations[joint].x));
        }
    }

    for (int g=0; g<getNumGeoms(); ++g) {
        float3 scale = make_float3(evaluateExpression(_geomParamExpressions[g][0]),
                evaluateExpression(_geomParamExpressions[g][1]),
                evaluateExpression(_geomParamExpressions[g][2]));
        _geomScales[g] = scale;
        if (_modelVersion == 0) {
            _geomTransforms[g] = SE3Fromse3(se3(evaluateExpression(_geomParamExpressions[g][3]),
                                                evaluateExpression(_geomParamExpressions[g][4]),
                                                evaluateExpression(_geomParamExpressions[g][5]),
                                                evaluateExpression(_geomParamExpressions[g][6]),
                                                evaluateExpression(_geomParamExpressions[g][7]),
                                                evaluateExpression(_geomParamExpressions[g][8])));
        } else {
            _geomTransforms[g] = SE3FromTranslation(evaluateExpression(_geomParamExpressions[g][3]),
                                                    evaluateExpression(_geomParamExpressions[g][4]),
                                                    evaluateExpression(_geomParamExpressions[g][5]))*
                                 SE3FromEuler(make_float3(evaluateExpression(_geomParamExpressions[g][8]),
                                                          evaluateExpression(_geomParamExpressions[g][7]),
                                                          evaluateExpression(_geomParamExpressions[g][6])));
        }
    }

}

// TODO
void HostOnlyModel::voxelize(float resolution, float padding, std::string cacheFile) {

    _frameSdfNumbers.clear();
    _sdfFrames.clear();
    _sdfs.clear();
    _sdfColors.clear();

    int n = 0;
    for (int i=0; i<_nFrames; i++) {
        if (getFrameNumGeoms(i) == 0) {
            _frameSdfNumbers.push_back(-1);
            continue;
        }

        std::string filename = dart::stringFormat("%s.sdf%02d.res%06f.sdf",cacheFile.c_str(),n,resolution);
        struct stat buffer;

        _sdfColors.push_back(getGeometryColor(getFrameGeoms(i)[getFrameNumGeoms(i)-1]));
        _sdfs.push_back(Grid3D<float>());
        if (cacheFile != "" && stat(filename.c_str(), &buffer) == 0) {
            std::cout << "sdf cached at " << filename << std::endl;
            std::ifstream stream;
            stream.open(filename.c_str(),std::ios_base::in | std::ios_base::binary);
            stream.read((char*)&_sdfs[n].dim,sizeof(int3));
            stream.read((char*)&_sdfs[n].offset,sizeof(float3));
            stream.read((char*)&_sdfs[n].resolution,sizeof(float));
            const int sdfSize = _sdfs[n].dim.x*_sdfs[n].dim.y*_sdfs[n].dim.z;
            _sdfs[n].data = new float[sdfSize];
            stream.read((char*)_sdfs[n].data,sdfSize*sizeof(float));
            stream.close();
        } else {
            voxelizeFrame(_sdfs[n],i,0.0,1e20,resolution,padding);

            //        distanceTransform3D(_voxel_grids[n].data,_voxel_grids[n].dim.x,_voxel_grids[n].dim.y,_voxel_grids[n].dim.z,true);
            //        cpu::signedDistanceTransform3D(_voxel_grids[n].data,_voxel_grids[n].dim.x,_voxel_grids[n].dim.y,_voxel_grids[n].dim.z,true);
            //        gpu::signedDistanceTransform3D<float,true>(_voxel_grids[n].data,_voxel_grids[n].dim);
            //        sdkStopTimer(&dt_timer);

            const int nVoxels = _sdfs[n].dim.x*_sdfs[n].dim.y*_sdfs[n].dim.z;
#if CUDA_BUILD
            float * sdfIn; cudaMalloc(&sdfIn,nVoxels*sizeof(float));
            float * sdfOut; cudaMalloc(&sdfOut,nVoxels*sizeof(float));
            cudaMemcpy(sdfIn,_sdfs[n].data,nVoxels*sizeof(float),cudaMemcpyHostToDevice);
#else
            float * sdfIn = _sdfs[n].data;
            float * sdfOut = new float[nVoxels];
#endif // CUDA_BUILD

            signedDistanceTransform3D<float,true>(sdfIn,sdfOut,_sdfs[n].dim.x,_sdfs[n].dim.y,_sdfs[n].dim.z);

#if CUDA_BUILD
            cudaMemcpy(_sdfs[n].data,sdfOut,nVoxels*sizeof(float),cudaMemcpyDeviceToHost);
            cudaFree(sdfIn);
            cudaFree(sdfOut);
#else
            memcpy(_sdfs[n].data,sdfOut,nVoxels*sizeof(float));
#endif // CUDA_BUILD

            if (cacheFile != "") {
                std::ofstream stream;
                stream.open(filename.c_str(),std::ios_base::out | std::ios_base::binary);
                stream.write((char *)&_sdfs[n].dim,sizeof(int3));
                stream.write((char *)&_sdfs[n].offset,sizeof(float3));
                stream.write((char *)&_sdfs[n].resolution,sizeof(float));
                stream.write((char *)_sdfs[n].data,nVoxels*sizeof(float));
                stream.close();
            }

        }

        _sdfFrames.push_back(i);
        _frameSdfNumbers.push_back(n);

        ++n;
    }

}

void HostOnlyModel::voxelize2(float resolution, float padding, std::string cacheFile) {

    _frameSdfNumbers.clear();
    _sdfFrames.clear();
    _sdfs.clear();
    _sdfColors.clear();

    int n = 0;
    for (int i=0; i<_nFrames; i++) {
        if (getFrameNumGeoms(i) == 0) {
            _frameSdfNumbers.push_back(-1);
            continue;
        }

        _sdfColors.push_back(getGeometryColor(getFrameGeoms(i)[getFrameNumGeoms(i)-1]));
        _sdfs.push_back(Grid3D<float>());


        std::string filename = dart::stringFormat("%s.sdf%02d.res%06f.sdf",cacheFile.c_str(),n,resolution);
        struct stat buffer;
        if ((cacheFile != "") && stat (filename.c_str(), &buffer) == 0) {
            std::cout << "sdf cached at " << filename << std::endl;
            std::ifstream stream;
            stream.open(filename.c_str(),std::ios_base::in | std::ios_base::binary);
            stream.read((char *)&_sdfs[n].dim,sizeof(int3));
            stream.read((char *)&_sdfs[n].offset,sizeof(float3));
            stream.read((char *)&_sdfs[n].resolution,sizeof(float));
            const int sdfSize = _sdfs[n].dim.x*_sdfs[n].dim.y*_sdfs[n].dim.z;
            _sdfs[n].data = new float[sdfSize];
            stream.read((char *)_sdfs[n].data,sdfSize*sizeof(float));
            stream.close();

        } else {

            voxelizeFrame(_sdfs[n],i,1.0,-1.0,resolution,padding);
            const int sdfSize = _sdfs[n].dim.x*_sdfs[n].dim.y*_sdfs[n].dim.z;
            float * signs = new float[sdfSize];
            memcpy(signs,_sdfs[n].data,sdfSize*sizeof(float));

            for (int k=0; k<_sdfs[n].dim.x*_sdfs[n].dim.y*_sdfs[n].dim.z; ++k) {
                _sdfs[n].data[k] = 1e20;
            }
            for (int g=0; g<getFrameNumGeoms(i); ++g) {
                int geomNum = getFrameGeoms(i)[g];

                switch (getGeometryType(geomNum)) {
                    case MeshType:
                        {
                            std::cout << "computing analytic mesh SDF" << std::endl;
                            int meshNum = getMeshNumber(geomNum);
                            const Mesh & mesh = getMesh(meshNum);
                            Mesh transformedMesh(mesh);
                            transformMesh(transformedMesh,getGeometryTransform(geomNum));
                            analyticMeshSdf(_sdfs[n],transformedMesh);
                        } break;
                    case PrimitiveCubeType:
                        {
                            std::cout << "computing analytic cube SDF" << std::endl;
                            const float3 scale = getGeometryScale(geomNum);
                            analyticBoxSdf(_sdfs[n],getGeometryTransform(geomNum),-0.5*scale,0.5*scale);
                        }
                    case PrimitiveSphereType:
                        {
                            std::cout << "computing analytic sphere SDF" << std::endl;
                            const float radius = getGeometryScale(geomNum).x;
                            assert(radius == getGeometryScale(geomNum).y);
                            assert(radius == getGeometryScale(geomNum).z);
                            analyticSphereSdf(_sdfs[n],getGeometryTransform(geomNum),radius);
                        }
                        break;
                }
            }
//            for (int k=0; k<_sdfs[n].dim.x*_sdfs[n].dim.y*_sdfs[n].dim.z; ++k) {
//                float & dist = _sdfs[n].data[k];
//                //dist = sqrt(dist/resolution);
//                dist = signs[k] == 0 ? -sqrt(dist)/resolution : sqrt(dist)/resolution;
//            }

            if (cacheFile != "") {
                std::ofstream stream;
                stream.open(filename.c_str(),std::ios_base::out | std::ios_base::binary);
                stream.write((char*)&_sdfs[n].dim,sizeof(int3));
                stream.write((char*)&_sdfs[n].offset,sizeof(float3));
                stream.write((char*)&_sdfs[n].resolution,sizeof(float));
                stream.write((char*)_sdfs[n].data,sdfSize*sizeof(float));
                stream.close();
            }
        }

        _sdfFrames.push_back(i);
        _frameSdfNumbers.push_back(n);

        ++n;
    }

}

void HostOnlyModel::setGeometryColor(int geomNumber, unsigned char color[]) {
    _geomColors[geomNumber] = make_uchar3(color[0],color[1],color[2]);
}

void HostOnlyModel::setGeometryColor(int geomNumber, unsigned char red, unsigned char green, unsigned char blue) {
    _geomColors[geomNumber] = make_uchar3(red,green,blue);
}


void HostOnlyModel::setJointLimits(const int joint, const float min, const float max) {

    if (joint < 0 || joint >= getNumJoints()) { return; }

    _jointLimits[joint] = make_float2(min,max);
}

void HostOnlyModel::setArticulation(const float * pose) {

    // compute transforms from frame to model
    int j = 6;
    for (int f=1; f<getNumFrames(); ++f) {

        const float p  = std::min(std::max(getJointMin(j-6),pose[j]),getJointMax(j-6));

        const int joint = f-1;
        SE3 T_pf = getTransformJointAxisToParent(joint);
        switch(_jointTypes[joint]) {
            case RotationalJoint:
                T_pf = T_pf*SE3Fromse3(se3(0, 0, 0,
                                           p*_axes[joint].x, p*_axes[joint].y, p*_axes[joint].z));
                ++j;
                break;
            case PrismaticJoint:
                T_pf = T_pf*SE3Fromse3(se3(p*_axes[joint].x, p*_axes[joint].y, p*_axes[joint].z,
                        0, 0, 0));
                ++j;
                break;
        }
        const int parent = getFrameParent(f);
        _T_mf[f] = _T_mf[parent]*T_pf;
        _T_fm[f] = SE3Invert(_T_mf[f]);
    }

}

void HostOnlyModel::setPose(const Pose & pose) {

    _T_cm = pose.getTransformModelToCamera();
    _T_mc = pose.getTransformCameraToModel();

    // compute transforms from frame to model
    int j = 0;
    for (int f=1; f<getNumFrames(); ++f) {

        const float p = std::min(std::max(getJointMin(j),pose.getArticulation()[j]),getJointMax(j));

        const int joint = f-1;
        SE3 T_pf = getTransformJointAxisToParent(joint);
        switch(_jointTypes[joint]) {
        case RotationalJoint:
            T_pf = T_pf*SE3Fromse3(se3(0, 0, 0,
                                                   p*_axes[joint].x, p*_axes[joint].y, p*_axes[joint].z));
            ++j;
            break;
        case PrismaticJoint:
            T_pf = T_pf*SE3Fromse3(se3(p*_axes[joint].x, p*_axes[joint].y, p*_axes[joint].z,
                                                   0, 0, 0));
            ++j;
            break;
        }
        const int parent = getFrameParent(f);
        _T_mf[f] = _T_mf[parent]*T_pf;
        _T_fm[f] = SE3Invert(_T_mf[f]);
    }

}

void HostOnlyModel::setGeometryScale(const int geom, const float3 scale) {
    _geomScales[geom] = scale;
}

void HostOnlyModel::setGeometryTransform(const int geom, const SE3 &T) {
    _geomTransforms[geom] = T;
}

void HostOnlyModel::setLinkParent(const int link, const int parent) {
    _parents[link] = parent;
}

void HostOnlyModel::setVoxelGrid(Grid3D<float> &grid, const int link) {

    const int sdfNum = getFrameSdfNumber(link);
    _sdfs[sdfNum] = grid;

}

void HostOnlyModel::voxelizeFrame(Grid3D<float> &sdf, const int frame, const float fg, const float bg, const float resolution, const float padding, const float Rc) {

    float min_x, min_y, min_z, max_x, max_y, max_z;
    min_x = min_y = min_z = std::numeric_limits<float>::infinity();
    max_x = max_y = max_z = -std::numeric_limits<float>::infinity();

    // compute geometry extrema (int geometric units)
    for (int g = 0; g < getFrameNumGeoms(frame); g++) {
        int geom = getFrameGeoms(frame)[g];

        float p[9];
        const float3 geomScale = getGeometryScale(geom);
        p[0] = geomScale.x;
        p[1] = geomScale.y;
        p[2] = geomScale.z;
        const se3 geomT = se3FromSE3(getGeometryTransform(geom));
        memcpy(&p[3],geomT.p,6*sizeof(float));

        float o[3];
        float s[3];

        switch(getGeometryType(geom)) {
        case PrimitiveSphereType:
            if (p[6] == 0  && p[7] == 0 && p[8] == 0) { // ellipsoid is not rotated
                o[0] = p[3]-p[0];
                o[1] = p[4]-p[1];
                o[2] = p[5]-p[2];
                s[0] = 2*p[0];
                s[1] = 2*p[1];
                s[2] = 2*p[2];
            }
            else {  // ellipsoid is rotated
                aabbEllipsoid(&p[0],&p[3],&p[6],o,s);
            }
            break;
        case PrimitiveCylinderType:
            if (p[6] == 0 && p[7] == 0 && p[8] == 0) { // cylinder is not rotated
                o[0] = p[3]-p[0];
                o[1] = p[4]-p[1];
                o[2] = p[5];
                s[0] = 2*p[0];
                s[1] = 2*p[1];
                s[2] = p[2];
            }
            else { // cylinder is rotated
                aabbEllipticCylinder(&p[0],p[2],&p[3],&p[6],o,s);
            }
            break;
        case PrimitiveCubeType:
            if (p[6] == 0 && p[7] == 0 && p[8] == 0) {
                o[0] = p[3]-0.5*p[0];
                o[1] = p[4]-0.5*p[1];
                o[2] = p[5]-0.5*p[2];
                s[0] = p[0];
                s[1] = p[1];
                s[2] = p[2];
            }
            else { // rectangular prism is rotated
                aabbRectangularPrism(&p[0],&p[3],&p[6],o,s);
            }
            break;
        case MeshType:
            const Mesh& mesh = _renderer->getMesh(getMeshNumber(geom));
            Mesh transformedMesh(mesh);
            scaleMesh(transformedMesh,make_float3(p[0],p[1],p[2]));

//            se3 T_se3(p[3],p[4],p[5],p[6],p[7],p[8]);
//            SE3 T_SE3 = SE3Fromse3(T_se3);

            SE3 T = SE3Fromse3(se3(0,0,0,p[6],p[7],p[8]))*SE3Fromse3(se3(p[3],p[4],p[5],0,0,0));

            transformMesh(transformedMesh,getGeometryTransform(geom));

//            for (int v=0; v<mesh.nVertices; ++v) {
//                const float3 &vert = mesh.vertices[v];
//                transformedMesh.vertices[v] = SE3Transform(T_SE3,make_float3(p[0]*vert.x,p[1]*vert.y,p[2]*vert.z));
//                transformedMesh.normals[v] = SE3Rotate(T_SE3,mesh.normals[v]);
//            }
//            memcpy(transformedMesh.faces,mesh.faces,mesh.nFaces*sizeof(int3));

            float xMax, yMax, zMax;
            o[0] = o[1] = o[2] = std::numeric_limits<float>::infinity();
            xMax = yMax = zMax = -std::numeric_limits<float>::infinity();

            for (int v=0; v<transformedMesh.nVertices; ++v) {
                const float3& vertex = transformedMesh.vertices[v];
                o[0] = std::min(o[0],vertex.x);
                o[1] = std::min(o[1],vertex.y);
                o[2] = std::min(o[2],vertex.z);
                xMax = std::max(xMax,vertex.x);
                yMax = std::max(yMax,vertex.y);
                zMax = std::max(zMax,vertex.z);
            }

            s[0] = xMax - o[0];
            s[1] = yMax - o[1];
            s[2] = zMax - o[2];

            break;
        }

        min_x = std::min(min_x,o[0]);
        max_x = std::max(max_x,o[0]+s[0]);
        min_y = std::min(min_y,o[1]);
        max_y = std::max(max_y,o[1]+s[1]);
        min_z = std::min(min_z,o[2]);
        max_z = std::max(max_z,o[2]+s[2]);
    }

    min_x -= padding;
    min_y -= padding;
    min_z -= padding;
    max_x += padding;
    max_y += padding;
    max_z += padding;

    // compute grid size (in voxel units)
    uint3 dim;
    dim.x = ceil((max_x-min_x)/resolution);
    dim.y = ceil((max_y-min_y)/resolution);
    dim.z = ceil((max_z-min_z)/resolution);

    std::cout << " voxel dimensions are " << dim.x << ", " << dim.y << ", " << dim.z << std::endl;

    sdf.dim = dim;
    sdf.data = new float[dim.x*dim.y*dim.z];
    for (int z=0; z<dim.z; z++)
        for (int y=0; y<dim.y; y++)
            for (int x=0; x<dim.x; x++)
                sdf.data[x + y*dim.x + z*dim.x*dim.y] = bg;
    sdf.resolution = resolution;
    sdf.offset = make_float3(min_x,min_y,min_z);

    for (int g = 0; g < getFrameNumGeoms(frame); g++) {
        int geom = getFrameGeoms(frame)[g];
        float p[9];
        const float3 geomScale = getGeometryScale(geom);
        p[0] = geomScale.x;
        p[1] = geomScale.y;
        p[2] = geomScale.z;
        const se3 geomT = se3FromSE3(getGeometryTransform(geom));
        memcpy(&p[3],geomT.p,6*sizeof(float));

        // compute offset (in geometric units)
        float off_x = (min_x) + 0.5*resolution;
        float off_y = (min_y) + 0.5*resolution;
        float off_z = (min_z) + 0.5*resolution;

        switch(getGeometryType(geom)) {
        case PrimitiveSphereType:
            if (p[6] == 0  && p[7] == 0 && p[8] == 0) { // ellipsoid is not rotated
                off_x -= p[3];
                off_y -= p[4];
                off_z -= p[5];
                for (int z = 0; z < dim.z; z++) {
                    float vox_z = z*resolution + off_z; // voxel z position (in geometric units)
                    float C = vox_z * vox_z / (p[2]*p[2]);
                    for (int y = 0; y < dim.y; y++) {
                        float vox_y = y*resolution + off_y; // voxel y position (in geometric units)
                        float B = vox_y * vox_y / (p[1]*p[1]);
                        for (int x = 0; x < dim.x; x++) {
                            float vox_x = x*resolution + off_x; // voxel x position (in geometric units)
                            float A = vox_x*vox_x / (p[0]*p[0]);
                            float vox_r = sqrt(A + B + C);
//                            if (fabs(vox_r - 1.0) < 0.5*sqrt(1/(p[0]*p[0]) + 1/(p[1]*p[1]) + 1/(p[2]*p[2]))*resolution)
//                                vg.data[x + y*dim.x + z*dim.x*dim.y] = fg;
                            if (vox_r < 1.0)
                                sdf.data[x + y*dim.x + z*dim.x*dim.y] = fg;
                        }
                    }
                }
            }
            else {  // ellipsoid is rotated
                SE3 T_vox_geom, T_geom_vox;
                T_vox_geom = SE3Fromse3(se3(p[3],p[4],p[5],p[6],p[7],p[8]));
                T_geom_vox = SE3Invert(T_vox_geom);

                float4 voxPt, geomPt;
                voxPt = geomPt = make_float4(0,0,0,1);

                for (int z = 0; z < dim.z; z++) {
                    voxPt.z = z*resolution + off_z;
                    for (int y = 0; y < dim.y; y++) {
                        voxPt.y = y*resolution + off_y;
                        for (int x = 0; x < dim.x; x++) {
                            voxPt.x = x*resolution + off_x;
                            geomPt = T_geom_vox * voxPt;
                            float A = geomPt.x*geomPt.x/(p[0]*p[0]);
                            float B = geomPt.y*geomPt.y/(p[1]*p[1]);
                            float C = geomPt.z*geomPt.z/(p[2]*p[2]);
                            float geom_r = sqrt(A + B + C);
                            if (geom_r < 1.0)
                                sdf.data[x + y*dim.x + z*dim.x*dim.y] = fg;
                        }
                    }
                }
            }
            break;
        case PrimitiveCylinderType:
            if (p[6] == 0  && p[7] == 0 && p[8] == 0) { // cylinder is not rotated
                for (int z = 0; z < dim.z; z++) {
                    float vox_z = z*resolution + off_z; // voxel z position (in geometric units)
                    for (int y = 0; y < dim.y; y++) {
                        float vox_y = y*resolution + off_y; // voxel y position (in geometric units)
                        float A = vox_y * vox_y / (p[1]*p[1]);
                        for (int x = 0; x < dim.x; x++) {
                            float vox_x = x*resolution + off_x; // voxel x position (in geometric units)
                            float B = vox_x * vox_x / (p[0]*p[0]);
                            float vox_r = sqrt(A + B);
//                            if (fabs(vox_r-1) < 0.5*sqrt(1/(p[0]*p[0]) + 1/(p[1]*p[1]))*resolution && vox_z >= 0 && vox_z < p[2])
//                                vg.data[x + y*dim.x + z*dim.x*dim.y] = fg;
                            if (vox_r < 1.0 && vox_z >= 0 && vox_z < p[2])
                                sdf.data[x + y*dim.x + z*dim.x*dim.y] = fg;
                        }
                    }
                }
            }
            else { // cylinder is rotated
                SE3 R_vox_geom, R_geom_vox;
                R_vox_geom = SE3Fromse3(se3(0,0,0,p[6],p[7],p[8]));
                R_geom_vox = SE3Invert(R_vox_geom);

                float4 voxPt, geomPt;
                voxPt = geomPt = make_float4(0,0,0,1);

                for (int z = 0; z < dim.z; z++) {
                    voxPt.z = z*resolution + off_z;
                    for (int y = 0; y < dim.y; y++) {
                        voxPt.y = y*resolution + off_y;
                        for (int x = 0; x < dim.x; x++) {
                            voxPt.x = x*resolution + off_x;
                            geomPt = R_geom_vox * voxPt;
                            float A = geomPt.x*geomPt.x/(p[0]*p[0]);
                            float B = geomPt.y*geomPt.y/(p[1]*p[1]);
                            float geom_r = sqrt(A + B);
                            if (geom_r < 1.0 && geomPt.z >= 0 && geomPt.z < p[2])
                                sdf.data[x +y*dim.x + z*dim.x*dim.y] = fg;
                        }
                    }
                }
            }
            break;
        case PrimitiveCubeType:
            if (p[6] == 0 && p[7] == 0 && p[8] == 0) {
                for (int z = 0; z < dim.z; z++) {
                    float vox_z = z*resolution + off_z; // voxel z position (in geometric units)
                    if (vox_z > 0.5*p[2] || vox_z < -0.5*p[2])
                        continue;
                    for (int y = 0; y < dim.y; y++) {
                        float vox_y = y*resolution + off_y; // voxel y position (in geometric units)
                        if (vox_y > 0.5*p[1] || vox_y < -0.5*p[1])
                            continue;
                        for (int x = 0; x < dim.x; x++) {
                            float vox_x = x*resolution + off_x; // voxel x position (in geometric units)
                            if (vox_x > 0.5*p[0] || vox_x < -0.5*p[0])
                                continue;
                            sdf.data[x + y*dim.x + z*dim.x*dim.y] = fg;
                        }
                    }
                }
            }
            else { // rectangular prism is rotated
                SE3 R_vox_geom, R_geom_vox;
                R_vox_geom = SE3Fromse3(se3(0,0,0,p[6],p[7],p[8]));
                R_geom_vox = SE3Invert(R_vox_geom);

                float4 voxPt, geomPt;
                voxPt = geomPt = make_float4(0,0,0,1);

                for (int z = 0; z < dim.z; z++) {
                    voxPt.z = z*resolution + off_z;
                    for (int y = 0; y < dim.y; y++) {
                        voxPt.y = y*resolution + off_y;
                        for (int x = 0; x < dim.x; x++) {
                            voxPt.x = x*resolution + off_x;
                            geomPt = R_geom_vox * voxPt;
                            if (geomPt.x > -0.5*p[0] && geomPt.x < 0.5*p[0] && geomPt.y > -0.5*p[1] && geomPt.y < 0.5*p[1] && geomPt.z > -0.5*p[2] && geomPt.z < 0.5*p[2])
                                sdf.data[x + y*dim.x + z*dim.x*dim.y] = fg;
                        }
                    }
                }
            }
            break;
        case MeshType:
            // TODO: wasteful to transform this twice
            const Mesh & mesh = _renderer->getMesh(getMeshNumber(geom));
            Mesh transformedMesh(mesh);
            scaleMesh(transformedMesh,make_float3(p[0],p[1],p[2]));

//            se3 T_se3(p[3],p[4],p[5],p[6],p[7],p[8]);
//            SE3 T_SE3 = SE3Fromse3(T_se3);

            SE3 T = SE3Fromse3(se3(0,0,0,p[6],p[7],p[8]))*SE3Fromse3(se3(p[3],p[4],p[5],0,0,0));

            transformMesh(transformedMesh,getGeometryTransform(geom));

            splatSolidMesh(transformedMesh, sdf);

        }
    }

}

}
