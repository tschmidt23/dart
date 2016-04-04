#ifndef MIRRORED_MODEL_H
#define MIRRORED_MODEL_H

#include "model/model.h"
#include "model/host_only_model.h"
#include "util/mirrored_memory.h"
#include "geometry/grid_3d.h"

namespace dart {

class  MirroredModel : public Model {
public:
    MirroredModel(const HostOnlyModel & copy,
                  const uint3 obsSdfDim,
                  const float obsSdfRes,
                  const float3 obsSdfOffset = make_float3(0,0,0),
                  const int modelID = -1);
    ~MirroredModel();

    void setArticulation(const float * pose);
    void setPose(const Pose & pose);

    const inline SE3 & getTransformFrameToModel(const int frame) const { return _T_mf->hostPtr()[frame]; }
    const inline SE3 & getTransformModelToFrame(const int frame) const { return _T_fm->hostPtr()[frame]; }
    const inline SE3 * getTransformsFrameToModel() const { return _T_mf->hostPtr(); }
    const inline SE3 * getTransformsModelToFrame() const { return _T_fm->hostPtr(); }

    inline uint getNumFrames() const { return _frameParents->length(); }
    inline int getFrameParent(const int frame) const { return _frameParents->hostPtr()[frame]; }

    inline uint getNumSdfs() const  { return _sdfFrames->length(); }
    const inline Grid3D<float> & getSdf(const int sdfNum) const { return _sdfs->hostGrids()[sdfNum]; }
    const inline Grid3D<float> * getSdfs() const { return _sdfs->hostGrids(); }
    inline uint getSdfFrameNumber(const int sdfNum) const { return _sdfFrames->hostPtr()[sdfNum]; }
    inline uchar3 getSdfColor(const int sdfNum) const { return _sdfColors->hostPtr()[sdfNum]; }

    inline uint getNumJoints() const { return getNumFrames()-1; }
    inline int getJointFrame(const int joint) const { return joint+1; }
    inline JointType getJointType(const int joint) const { return _jointTypes->hostPtr()[joint]; }
    inline float3 getJointAxis(const int joint) const { return _jointAxes->hostPtr()[joint]; }
    inline float3 & getJointPosition(const int joint) { return _jointPositions->hostPtr()[joint]; }
    inline float3 & getJointOrientation(const int joint) { return _jointOrientations->hostPtr()[joint]; }
    inline const float3 & getJointPosition(const int joint) const { return _jointPositions->hostPtr()[joint]; }
    inline const float3 & getJointOrientation(const int joint) const { return _jointOrientations->hostPtr()[joint]; }

    inline uchar3 getGeometryColor(const int geomNumber) const { return _geomColors->hostPtr()[geomNumber]; }

    const inline int * getDependencies() const { return _dependencies->hostPtr(); }
    inline int getDependency(const int frame, const int joint) const { return _dependencies->hostPtr()[frame * getNumJoints() + joint]; }

    inline Grid3D<float> * getObsSdf() { return _obsSdf->hostGrid(); }
    inline void syncObsSdfHostToDevice() { _obsSdf->syncHostToDevice(); }
    inline void syncObsSdfDeviceToHost() { _obsSdf->syncDeviceToHost(); }
    inline float3 getObsSdfoffset() const { return _obsSdfOffset; }

    inline uint getModelID() const { return _modelID; }

    inline void setSdfColor(const int sdfNum, const uchar3 color) { _sdfColors->hostPtr()[sdfNum] = color; _sdfColors->syncHostToDevice(); }
    inline void setGeometryColor(const int geomNum, const uchar3 color) { _geomColors->hostPtr()[geomNum] = color; _geomColors->syncHostToDevice(); }

    // ------------- device getters -----------------------

    inline SE3 * getDeviceTransformsFrameToModel() { return _T_mf->devicePtr(); }
    inline SE3 * getDeviceTransformsModelToFrame() { return _T_fm->devicePtr(); }
    inline const SE3 * getDeviceTransformsFrameToModel() const { return _T_mf->devicePtr(); }
    inline const SE3 * getDeviceTransformsModelToFrame() const { return _T_fm->devicePtr(); }

    inline const int * getDeviceFrameParents() const { return _frameParents->devicePtr(); }

    inline int * getDeviceSdfFrames() { return _sdfFrames->devicePtr(); }
    inline const int * getDeviceSdfFrames() const { return _sdfFrames->devicePtr(); }

    inline const Grid3D<float> * getDeviceSdfs() const { return _sdfs->deviceGrids(); }
    inline const uchar3 * getDeviceSdfColors() const { return _sdfColors->devicePtr(); }

    inline const JointType * getDeviceJointTypes() const { return _jointTypes->devicePtr(); }
    inline const float3 * getDeviceJointAxes() const { return _jointAxes->devicePtr(); }

    const inline uchar3 * getDeviceGeometryColors() const { return _geomColors->devicePtr(); }

    const inline int * getDeviceDependencies() const { return _dependencies->devicePtr(); }

    inline Grid3D<float> * getDeviceObsSdf() { return _obsSdf->deviceGrid(); }
    inline const Grid3D<float> * getDeviceObsSdf() const { return _obsSdf->deviceGrid(); }
    inline float * getDeviceObsSdfData() { return _obsSdf->deviceData(); }

    inline const SE3 getTransformJointAxisToParent(const int joint) const { return _T_pf->hostPtr()[joint]; }

private:

    uint _modelID;

    // transforms
    MirroredVector<SE3> * _T_mf;
    MirroredVector<SE3> * _T_fm;
    MirroredVector<SE3> * _T_pf;

    // frame data
    MirroredVector<int> * _frameParents;

    // sdf data
    MirroredVector<int> * _sdfFrames;
    MirroredGrid3DVector<float> * _sdfs;
    MirroredVector<uchar3> * _sdfColors;

    // joint data
    MirroredVector<float3> * _jointPositions;
    MirroredVector<float3> * _jointOrientations;
    MirroredVector<float3> * _jointAxes;
    MirroredVector<JointType> * _jointTypes;

    // geometry data
    MirroredVector<uchar3> * _geomColors;

    // dependency data
    MirroredVector<int> * _dependencies;

    // obs sdf data
    MirroredGrid3D<float> * _obsSdf;
    float3 _obsSdfOffset;

};

}

#endif // MIRRORED_MODEL_H
