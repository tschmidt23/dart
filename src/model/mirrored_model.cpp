#include "mirrored_model.h"

#include <cuda_runtime.h>
#include "util/cuda_utils.h"

namespace dart {

MirroredModel::MirroredModel(const HostOnlyModel & copy,
                             const uint3 obsSdfDim,
                             const float obsSdfRes,
                             const float3 obsSdfOffset,
                             const int modelID) : Model(copy) {

    static uint nextModelID = 0;
    if (modelID >= 0) {
        _modelID = modelID;
    } else {
        _modelID = nextModelID;
        ++nextModelID;
    }

    const uint numFrames = copy.getNumFrames();
    const uint numSdfs = copy.getNumSdfs();
    const uint numJoints = copy.getNumJoints();
    const uint numGeoms = copy.getNumGeoms();

    // set up transform data
    _T_mf = new MirroredVector<SE3>(numFrames);
    memcpy(_T_mf->hostPtr(),copy.getTransformsFrameToModel(),numFrames*sizeof(SE3));
    _T_mf->syncHostToDevice();

    _T_fm = new MirroredVector<SE3>(numFrames);
    memcpy(_T_fm->hostPtr(),copy.getTransformsModelToFrame(),numFrames*sizeof(SE3));
    _T_fm->syncHostToDevice();

    if (numJoints > 0) {
        _T_pf = new MirroredVector<SE3>(numJoints);
        for (int j=0; j<numJoints; ++j) {
            _T_pf->hostPtr()[j] = copy.getTransformJointAxisToParent(j);
        }
        _T_pf->syncHostToDevice();
    } else {
        _T_pf = 0;
    }

    // set up frame data
    _frameParents = new MirroredVector<int>(numFrames);
    for (int f=0; f<numFrames; ++f) {
        _frameParents->hostPtr()[f] = copy.getFrameParent(f);
    }
    _frameParents->syncHostToDevice();

    // set up sdf data
    if (numSdfs > 0) {
        _sdfFrames = new MirroredVector<int>(numSdfs);
        for (int s=0; s<numSdfs; ++s) {
            _sdfFrames->hostPtr()[s] = copy.getSdfFrameNumber(s);
        }
        _sdfFrames->syncHostToDevice();

        _sdfs = new MirroredGrid3DVector<float>(numSdfs,(Grid3D<float> *)copy.getSdfs());
        _sdfs->syncHostToDevice();

        _sdfColors = new MirroredVector<uchar3>(numSdfs);
        for (int s=0; s<numSdfs; ++s) {
            _sdfColors->hostPtr()[s] = copy.getSdfColor(s);
        }
        _sdfColors->syncHostToDevice();
    } else {
        _sdfFrames = 0;
        _sdfs = 0;
        _sdfColors = 0;
        std::cerr << "warning: you've created a MirroredModel with no SDFs. Did you forget to call voxelize?" << std::endl;
    }

    // set up joint data
    _jointTypes = new MirroredVector<JointType>(numJoints);
    _jointPositions = new MirroredVector<float3>(numJoints);
    _jointOrientations = new MirroredVector<float3>(numJoints);
    _jointAxes = new MirroredVector<float3>(numJoints);
    for (int j=0; j<numJoints; ++j) {
        _jointTypes->hostPtr()[j] = copy.getJointType(j);
        _jointPositions->hostPtr()[j] = copy.getJointPosition(j);
        _jointOrientations->hostPtr()[j] = copy.getJointOrientation(j);
        _jointAxes->hostPtr()[j] = copy.getJointAxis(j);
    }
    _jointTypes->syncHostToDevice();
    _jointPositions->syncHostToDevice();
    _jointOrientations->syncHostToDevice();
    _jointAxes->syncHostToDevice();

    // set up geometry data
    if (numGeoms > 0) {
        _geomColors = new MirroredVector<uchar3>(numGeoms);
        for (int g=0; g<numGeoms; ++g) {
            _geomColors->hostPtr()[g] = copy.getGeometryColor(g);
        }
        _geomColors->syncHostToDevice();
    } else {
        _geomColors = 0;
    }

    // set up dependency data
    _dependencies = new MirroredVector<int>(numJoints*numFrames);
    memcpy(_dependencies->hostPtr(),copy.getDependencies(),numFrames*numJoints*sizeof(int));
    _dependencies->syncHostToDevice();

    // set up observed SDF data
    Grid3D<float> hObsSdf(obsSdfDim, obsSdfOffset, obsSdfRes);

    _obsSdf = new MirroredGrid3D<float>(hObsSdf);

    _obsSdfOffset = obsSdfOffset;

    memset(_obsSdf->hostData(),0,obsSdfDim.x*obsSdfDim.y*obsSdfDim.z*sizeof(float));

}

MirroredModel::~MirroredModel() {

    // free transform data
    delete _T_mf;
    delete _T_fm;
    delete _T_pf;

    // free frame data
    delete _frameParents;

    // free sdf data
    delete _sdfFrames;
    delete _sdfColors;
    delete _sdfs;

    // free joint data
    delete _jointTypes;
    delete _jointPositions;
    delete _jointOrientations;
    delete _jointAxes;

    // free geometry data
    delete _geomColors;

    // free dependency data
    delete _dependencies;

    // free obs sdf data
    delete _obsSdf;

}

void MirroredModel::setArticulation(const float * pose) {

    int j = 6;
    for (int f=1; f<getNumFrames(); ++f) {

        float p = std::min(std::max(getJointMin(j-6),pose[j]),getJointMax(j-6));

        const int joint = f-1;
        SE3 T_pf = getTransformJointAxisToParent(joint);
        switch(_jointTypes->hostPtr()[joint]) {
            case RotationalJoint:
                T_pf = T_pf*SE3Fromse3(se3(0, 0, 0,
                                           p*_jointAxes->hostPtr()[joint].x, p*_jointAxes->hostPtr()[joint].y, p*_jointAxes->hostPtr()[joint].z));
                ++j;
                break;
            case PrismaticJoint:
                T_pf = T_pf*SE3Fromse3(se3(p*_jointAxes->hostPtr()[joint].x, p*_jointAxes->hostPtr()[joint].y, p*_jointAxes->hostPtr()[joint].z,
                                           0, 0, 0));
                ++j;
                break;
        }
        const int parent = getFrameParent(f);
        _T_mf->hostPtr()[f] = _T_mf->hostPtr()[parent]*T_pf;
        _T_fm->hostPtr()[f] = SE3Invert(_T_mf->hostPtr()[f]);
    }

    // sync to device
    _T_mf->syncHostToDevice();
    _T_fm->syncHostToDevice();
}

void MirroredModel::setPose(const Pose & pose) {

    _T_cm = pose.getTransformModelToCamera();
    _T_mc = pose.getTransformCameraToModel();

    int j = 0;
    for (int f=1; f<getNumFrames(); ++f) {

        float p = std::min(std::max(getJointMin(j),pose.getArticulation()[j]),getJointMax(j));

        const int joint = f-1;
        SE3 T_pf = getTransformJointAxisToParent(joint);
        switch(_jointTypes->hostPtr()[joint]) {
            case RotationalJoint:
                T_pf = T_pf*SE3Fromse3(se3(0, 0, 0,
                                           p*_jointAxes->hostPtr()[joint].x, p*_jointAxes->hostPtr()[joint].y, p*_jointAxes->hostPtr()[joint].z));
                ++j;
                break;
            case PrismaticJoint:
                T_pf = T_pf*SE3Fromse3(se3(p*_jointAxes->hostPtr()[joint].x, p*_jointAxes->hostPtr()[joint].y, p*_jointAxes->hostPtr()[joint].z,
                                           0, 0, 0));
                ++j;
                break;
        }
        const int parent = getFrameParent(f);
        _T_mf->hostPtr()[f] = _T_mf->hostPtr()[parent]*T_pf;
        _T_fm->hostPtr()[f] = SE3Invert(_T_mf->hostPtr()[f]);
    }

    // sync to device
    _T_mf->syncHostToDevice();
    _T_fm->syncHostToDevice();
}

}
