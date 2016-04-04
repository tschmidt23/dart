#include "tracker.h"

#include "mesh/mesh_proc.h"
#include "mesh/mesh_sample.h"
#include "mesh/primitive_meshing.h"
#include "util/cuda_utils.h"
#include "util/dart_io.h"
#include "util/string_format.h"
#include "geometry/plane_fitting.h"
#if ASSIMP_BUILD
#include "mesh/assimp_mesh_reader.h"
#endif // ASSIMP_BUILD
#if CUDA_BUILD
#include <cuda_gl_interop.h>
#endif // CUDA_BUILD

#include <GL/glx.h>

namespace dart {

Tracker::Tracker() : _depthSource(0), _pcSource(0), _optimizer(0),
    _T_mcs(0), _T_fms(0), _sdfFrames(0), _sdfs(0), _nSdfs(0), _distanceThresholds(0),
    _normalThresholds(0), _planeOffsets(0), _planeNormals(0), _dampingMatrices(0) {

    glewInit();

#if ASSIMP_BUILD
    Model::initializeRenderer(new AssimpMeshReader());
#else
    Model::initializeRenderer();
#endif // ASSIMP_BUILD

    cudaGLSetGLDevice(0);
    cudaDeviceReset();

}

Tracker::~Tracker() {

    for (int m=0; m<_mirroredModels.size(); ++m) {
        delete _mirroredModels[m];
    }

    for (PoseReduction * reduction : _ownedPoseReductions) {
        delete reduction;
    }

    for (Eigen::MatrixXf * matrix : _dampingMatrices) {
        delete matrix;
    }

    delete _pcSource;
    delete _optimizer;

    Model::shutdownRenderer();

}

bool Tracker::addModel(const std::string & filename,
                       const float modelSdfResolution,
                       const float modelSdfPadding,
                       const int obsSdfSize,
                       float obsSdfResolution,
                       float3 obsSdfOffset,
                       PoseReduction * poseReduction,
                       const float collisionCloudDensity,
                       const bool cacheSdfs) {

    HostOnlyModel model;
    if (!readModelXML(filename.c_str(),model)) {
        return false;
    }

    model.computeStructure();

    std::cout << "loading model from " << filename << std::endl;

    const int lastSlash = filename.find_last_of('/');
    const int lastDot = filename.find_last_of('.');
    const int diff = lastDot - lastSlash - 1;
    const int substrStart = lastSlash < filename.size() ? lastSlash + 1: 0;
    const std::string modelName = filename.substr(substrStart, diff > 0 ? diff : filename.size() - substrStart);
    std::cout << "model name: " << modelName << std::endl;

//    // TODO
//    if (model.getNumGeoms() < 2) {
//        model.voxelize2(modelSdfResolution,modelSdfPadding,cacheSdfs ? dart::stringFormat("model%02d",_mirroredModels.size()) : "");
//    } else {
        model.voxelize(modelSdfResolution,modelSdfPadding,cacheSdfs ? dart::stringFormat("/tmp/%s",modelName.c_str()) : "");
//    }

    if (obsSdfResolution <= 0 ) {
        // compute obs sdf size dynamically
        const float obsSdfPadding = 0.02;
        if (model.getPoseDimensionality() == 6) {
            // rigid - just take padding
            dart::Grid3D<float> rootSdf = model.getSdf(0);
            float3 rootSdfSizeMeters = rootSdf.resolution*make_float3(rootSdf.dim.x,rootSdf.dim.y,rootSdf.dim.z);
            float3 rootSdfCenter = rootSdf.offset + 0.5*rootSdfSizeMeters;
            float3 obsSdfSizeMeters = rootSdfSizeMeters -
                                      2*make_float3(modelSdfPadding)
                                      +2*make_float3(obsSdfPadding);
            float maxDimMeters = std::max(std::max(obsSdfSizeMeters.x,obsSdfSizeMeters.y),obsSdfSizeMeters.z);
            obsSdfResolution = maxDimMeters / obsSdfSize;
            obsSdfOffset = rootSdfCenter - make_float3(maxDimMeters/2);
        } else {
            float3 min, max;
            model.getArticulatedBoundingBox(min,max,modelSdfPadding);

            float3 obsSdfSizeMeters = max - min + 2*make_float3(obsSdfPadding);
            float maxDimMeters = std::max(std::max(obsSdfSizeMeters.x,obsSdfSizeMeters.y),obsSdfSizeMeters.z);
            obsSdfResolution = maxDimMeters / obsSdfSize;
            obsSdfOffset = min + 0.5*(obsSdfSizeMeters - make_float3(maxDimMeters)) - make_float3(obsSdfPadding);
        }
    }

    _mirroredModels.push_back(new MirroredModel(model,
                                                make_uint3(obsSdfSize),
                                                obsSdfResolution,
                                                obsSdfOffset));

    _sizeParams.push_back(model.getSizeParams());

    int nDimensions = model.getPoseDimensionality();
    if (poseReduction == 0) {
        std::vector<float> jointMins, jointMaxs;
        std::vector<std::string> jointNames;
        for (int j=0; j<model.getNumJoints(); ++j) {
            jointMins.push_back(model.getJointMin(j));
            jointMaxs.push_back(model.getJointMax(j));
            jointNames.push_back(model.getJointName(j));
        }
        poseReduction = new NullReduction(nDimensions - 6,
                                          jointMins.data(),
                                          jointMaxs.data(),
                                          jointNames.data());
        _ownedPoseReductions.push_back(poseReduction);
    }
    _estimatedPoses.push_back(Pose(poseReduction));

    _filenames.push_back(filename);

    // build collision cloud
    MirroredVector<float4> * collisionCloud = 0;
    _collisionCloudSdfLengths.push_back(std::vector<int>(model.getNumSdfs()));
    _collisionCloudSdfStarts.push_back(std::vector<int>(model.getNumSdfs()));
    int nPointsCumulative = 0;
    for (int f=0; f<model.getNumFrames(); ++f) {
        int sdfNum = model.getFrameSdfNumber(f);
        if (sdfNum >= 0) {
            _collisionCloudSdfStarts[getNumModels()-1][sdfNum] = nPointsCumulative;
            _collisionCloudSdfLengths[getNumModels()-1][sdfNum] = 0;
        }
        for (int g=0; g<model.getFrameNumGeoms(f); ++g) {
            int gNum = model.getFrameGeoms(f)[g];
            const SE3 mT = model.getGeometryTransform(gNum);
            const float3 scale = model.getGeometryScale(gNum);
            std::vector<float3> sampledPoints;
            Mesh * samplerMesh;
            switch (model.getGeometryType(gNum)) {
            case MeshType:
            {
                int mNum = model.getMeshNumber(gNum);
                const Mesh & mesh = model.getMesh(mNum);
                samplerMesh = new Mesh(mesh);
            }
                break;
            case PrimitiveSphereType:
                samplerMesh = generateUnitIcosphereMesh(2);
                break;
            case PrimitiveCylinderType:
                samplerMesh = generateCylinderMesh(30);
                break;
            case PrimitiveCubeType:
                samplerMesh = generateCubeMesh();
                break;
            default:
            {
                std::cerr << "collision clouds for type " << model.getGeometryType(gNum) << " not supported yet" << std::endl;
                continue;
            }
                break;
            }

            scaleMesh(*samplerMesh,scale);
            transformMesh(*samplerMesh,mT);
            sampleMesh(sampledPoints,*samplerMesh,collisionCloudDensity);
            delete samplerMesh;
//            std::cout << "sampled " << sampledPoints.size() << " points" << std::endl;

            int start;
            if (collisionCloud == 0) {
                start = 0;
                collisionCloud = new MirroredVector<float4>(sampledPoints.size());
            } else {
                start = collisionCloud->length();
                collisionCloud->resize(start + sampledPoints.size());
            }
            for (int v=0; v<sampledPoints.size(); ++v) {
                float4 vert = make_float4(sampledPoints[v],sdfNum);
                collisionCloud->hostPtr()[start + v] = vert;
            }

            _collisionCloudSdfLengths[getNumModels()-1][sdfNum] += sampledPoints.size();
            nPointsCumulative += sampledPoints.size();

        }
    }
    collisionCloud->syncHostToDevice();
    _collisionClouds.push_back(collisionCloud);
    MirroredVector<int> * intersectionPotentialMatrix = new MirroredVector<int>(model.getNumSdfs()*model.getNumSdfs());
    memset(intersectionPotentialMatrix->hostPtr(),0,intersectionPotentialMatrix->length()*sizeof(int));
    intersectionPotentialMatrix->syncHostToDevice();
    _intersectionPotentialMatrices.push_back(intersectionPotentialMatrix);

    if (getNumModels() == 1) {
        _T_mcs = new MirroredVector<SE3>(1);
        _T_fms = new MirroredVector<SE3*>(1);
        _sdfFrames = new MirroredVector<int*>(1);
        _sdfs = new MirroredVector<const Grid3D<float>*>(1);
        _nSdfs = new MirroredVector<int>(1);
        _distanceThresholds = new MirroredVector<float>(1);
        _normalThresholds = new MirroredVector<float>(1);
        _planeOffsets = new MirroredVector<float>(1);
        _planeNormals = new MirroredVector<float3>(1);
    } else {
        _T_mcs->resize(getNumModels());
        _T_fms->resize(getNumModels());
        _sdfFrames->resize(getNumModels());
        _sdfs->resize(getNumModels());
        _nSdfs->resize(getNumModels());
        _distanceThresholds->resize(getNumModels());
        _normalThresholds->resize(getNumModels());
        _planeOffsets->resize(getNumModels());
        _planeNormals->resize(getNumModels());
    }

    MirroredModel & mm = *_mirroredModels.back();
    const int m = getNumModels()-1;
    _T_fms->hostPtr()[m] = mm.getDeviceTransformsModelToFrame(); _T_fms->syncHostToDevice();
    _sdfFrames->hostPtr()[m] = mm.getDeviceSdfFrames(); _sdfFrames->syncHostToDevice();
    _sdfs->hostPtr()[m] = mm.getDeviceSdfs(); _sdfs->syncHostToDevice();
    _nSdfs->hostPtr()[m] = mm.getNumSdfs(); _nSdfs->syncHostToDevice();

    const int reducedDims = _estimatedPoses.back().getReducedDimensions();
    _dampingMatrices.push_back(new Eigen::MatrixXf(reducedDims,reducedDims));
    *_dampingMatrices.back() = Eigen::MatrixXf::Zero(reducedDims,reducedDims);

    _estimatedPoses.back().zero();
    _estimatedPoses.back().projectReducedToFull();
    mm.setPose(_estimatedPoses.back());

    if (getNumModels() > 1) {
        _opts.distThreshold.resize(getNumModels());         _opts.distThreshold.back() = _opts.distThreshold.front();
        _opts.regularization.resize(getNumModels());        _opts.regularization.back() = _opts.regularization.front();
        _opts.regularizationScaled.resize(getNumModels());  _opts.regularizationScaled.resize(getNumModels());
        _opts.planeOffset.resize(getNumModels());           _opts.planeOffset.back() = _opts.planeOffset.front();
        _opts.planeNormal.resize(getNumModels());           _opts.planeNormal.back() = _opts.planeNormal.front();
        _opts.lambdaIntersection.resize(getNumModels()*getNumModels());
        for (int i=(getNumModels()-1)*(getNumModels()-1); i<_opts.lambdaIntersection.size(); ++i) { _opts.lambdaIntersection[i] = 0; }

    }

    CheckCudaDieOnError();

    return true;
}

template <typename DepthType, typename ColorType>
bool Tracker::addDepthSource(DepthSource<DepthType,ColorType> * depthSource) {

    if (_depthSource != 0) {
        std::cerr << "using multiple depth sources is not supported yet!" << std::endl;
        return false;
    }

    _depthSource = depthSource;
    _pcSource = new PointCloudSource<DepthType,ColorType>(depthSource,make_float2(0.1,5));

    _optimizer = new Optimizer(depthSource,depthSource->getDepthWidth()/2,depthSource->getDepthHeight()/2);

    return true;

}
template bool Tracker::addDepthSource<float,uchar3>(DepthSource<float,uchar3> * depthSource);
template bool Tracker::addDepthSource<ushort,uchar3>(DepthSource<ushort,uchar3> * depthSource);

void Tracker::updateModel(const int modelNum,
                          const float modelSdfResolution,
                          const float modelSdfPadding,
                          const int obsSdfSize,
                          const float obsSdfResolution,
                          const float3 obsSdfCenter) {

    int modelID = _mirroredModels[modelNum]->getModelID();
    delete _mirroredModels[modelNum];

    HostOnlyModel model;
    readModelXML(_filenames[modelNum].c_str(),model);

    for (std::map<std::string,float>::const_iterator it = _sizeParams[modelNum].begin();
         it != _sizeParams[modelNum].end(); ++it) {
        model.setSizeParam(it->first,it->second);
    }

    model.computeStructure();
    model.voxelize(modelSdfResolution,modelSdfPadding);
    MirroredModel * newModel = new MirroredModel(model,
                                                 make_uint3(obsSdfSize),
                                                 obsSdfResolution,
                                                 obsSdfCenter,
                                                 modelID);
    _mirroredModels[modelNum] = newModel;

    _T_fms->hostPtr()[modelNum] = newModel->getDeviceTransformsModelToFrame(); _T_fms->syncHostToDevice();
    _sdfFrames->hostPtr()[modelNum] = newModel->getDeviceSdfFrames();          _sdfFrames->syncHostToDevice();
    _sdfs->hostPtr()[modelNum] = newModel->getDeviceSdfs();                    _sdfs->syncHostToDevice();

    newModel->setPose(_estimatedPoses[modelNum]);

    CheckCudaDieOnError();
}

void Tracker::stepForward() {

    if (!initialized()) {
        std::cerr << "the tracker is not initialized properly for tracking. make sure a depth source and a model have been added" << std::endl;
        return;
    }

    _pcSource->advance();

}

void Tracker::stepBackward() {

    if (!initialized()) {
        std::cerr << "the tracker is not initialized properly for tracking. make sure a depth source and a model have been added" << std::endl;
        return;
    }

    if (_depthSource->getFrame() > 0) {
        _depthSource->setFrame(_depthSource->getFrame()-1);
    }
}

void Tracker::setFrame(const int frame) {

    if (!initialized()) {
        std::cerr << "the tracker is not initialized properly for tracking. make sure a depth source and a model have been added" << std::endl;
        return;
    }

    _pcSource->setFrame(frame);

}

void Tracker::subtractPlane(const float3 planeNormal,
                            const float planeIntercept,
                            const float distThreshold,
                            const float normThreshold) {

    subtractPlane_(_pcSource->getDeviceVertMap(),
                   _pcSource->getDeviceNormMap(),
                   _pcSource->getDepthWidth(),_pcSource->getDepthHeight(),
                   planeNormal,planeIntercept,
                   distThreshold,normThreshold);

}

void Tracker::optimizePose(const int modelNum) {

    if (!initialized()) {
        std::cerr << "the tracker is not initialized properly for tracking. make sure a depth source and a model have been added" << std::endl;
        return;
    }

    _optimizer->optimizePose(*_mirroredModels[modelNum],
                             _estimatedPoses[modelNum],
                             _pcSource->getDeviceVertMap(),
                             _pcSource->getDeviceNormMap(),
                             _pcSource->getDepthWidth(),
                             _pcSource->getDepthHeight(),
                             _opts,
                             *_collisionClouds[modelNum],
                             *_intersectionPotentialMatrices[modelNum]);

}

void Tracker::optimizePoses() {

    if (!initialized()) {
        std::cerr << "the tracker is not initialized properly for tracking. make sure a depth source and a model have been added" << std::endl;
        return;
    }

    _optimizer->optimizePoses(_mirroredModels,
                              _estimatedPoses,
                              _pcSource->getDeviceVertMap(),
                              _pcSource->getDeviceNormMap(),
                              _pcSource->getDepthWidth(),
                              _pcSource->getDepthHeight(),
                              _opts,
                              *_T_mcs,
                              *_T_fms,
                              *_sdfFrames,
                              *_sdfs,
                              *_nSdfs,
                              *_distanceThresholds,
                              *_normalThresholds,
                              *_planeOffsets,
                              *_planeNormals,
                              _collisionClouds,
                              _intersectionPotentialMatrices,
                              _dampingMatrices,
                              _priors);

}

void Tracker::setIntersectionPotentialMatrix(const int modelNum, const int * mx) {
    delete _intersectionPotentialMatrices[modelNum];
    const int nSdfs = _mirroredModels[modelNum]->getNumSdfs();
    _intersectionPotentialMatrices[modelNum] = new MirroredVector<int>(nSdfs*nSdfs);
    memcpy(_intersectionPotentialMatrices[modelNum]->hostPtr(),mx,nSdfs*nSdfs*sizeof(int));
    _intersectionPotentialMatrices[modelNum]->syncHostToDevice();
}

}
