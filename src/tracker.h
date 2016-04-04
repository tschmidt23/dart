#ifndef TRACKER_H
#define TRACKER_H

#include <vector>

#include "model/mirrored_model.h"
#include "pose/pose.h"
#include "depth_sources/depth_source.h"
#include "optimization/optimization.h"
#include "optimization/optimizer.h"
#include "optimization/priors.h"
#include "point_cloud_src/point_cloud_src.h"
#include "util/string_format.h"

namespace dart {

const int versionMajor = 0;
const int versionMinor = 0;
const int versionRevision = 1;

const inline std::string getVersionString() { return stringFormat("%d.%d.%d",versionMajor,versionMinor,versionRevision); }

class Tracker {

public:

    Tracker();
    ~Tracker();

    bool addModel(const std::string & filename,
                  const float modelSdfResolution = 0.005,
                  const float modelSdfPadding = 0.10,
                  const int obsSdfSize = 64,
                  float obsSdfResolution = -1,
                  float3 obsSdfCenter = make_float3(0,0,0),
                  PoseReduction * poseReduction = 0,
                  const float collisionCloudDensity = 1e5,
                  const bool cacheSdfs = true);

    template <typename DepthType, typename ColorType>
    bool addDepthSource(DepthSource<DepthType,ColorType> * _depthSource);

    void updateModel(const int modelNum,
                     const float modelSdfResolution = 0.005,
                     const float modelSdfPadding = 0.10,
                     const int obsSdfSize = 64,
                     const float obsSdfResolution = 0.01,
                     const float3 obsSdfCenter = make_float3(0,0,0));

    inline void updatePose(const int modelNum) {
        _estimatedPoses[modelNum].projectReducedToFull();
        _mirroredModels[modelNum]->setPose(_estimatedPoses[modelNum]);
    }

    void stepForward();
    void stepBackward();
    void setFrame(const int frame);

    void optimizePose(const int modelNum = 0);

    /**
     * @brief This function runs the optimizer to infer the poses of all tracked models in the currently observed frame.
     * @param opts A struct setting up various optimzation parameters.
     */
    void optimizePoses();

    void subtractPlane(const float3 planeNormal,
                       const float planeIntercept,
                       const float distThreshold,
                       const float normThreshold);

    // accessors
    inline int getNumModels() const { return _mirroredModels.size(); }

    inline const MirroredModel & getModel(const int modelNum) const { return * _mirroredModels[modelNum]; }
    inline MirroredModel & getModel(const int modelNum) { return * _mirroredModels[modelNum]; }

    inline const Pose & getPose(const int modelNum) const { return _estimatedPoses[modelNum]; }
    inline Pose & getPose(const int modelNum) { return _estimatedPoses[modelNum]; }
    inline std::map<std::string,float> & getSizeParams(const int modelNum) { return _sizeParams[modelNum]; }
    inline const float4 * getHostVertMap() { return _pcSource->getHostVertMap(); }
    inline const float4 * getHostNormMap() { return _pcSource->getHostNormMap(); }

    inline const float * getDeviceDebugErrorObsToMod() { return _optimizer->getDeviceDebugErrorObsToMod(); }
    inline const int * getDeviceDebugDataAssociationObsToMod() { return _optimizer->getDeviceDebugDataAssociationObsToMod(); }
    inline const float * getDeviceDebugErrorModToObs() { return _optimizer->getDeviceDebugErrorModToObs(); }
    inline const int * getDeviceDebugDataAssociationModToObs() { return _optimizer->getDeviceDebugDataAssociationModToObs(); }

    inline const int getPredictionWidth() { return _optimizer->getPredictionWidth(); }
    inline const int getPredictionHeight() { return _optimizer->getPredictionHeight(); }
    inline const float4 * getDevicePredictedVertMap(bool forceComputation = false) {
        if (forceComputation) {
            _optimizer->computePredictedPointCloud(_mirroredModels);
        }
        return _optimizer->getDevicePredictedPoints();
    }

    inline void debugPredictionRay(const int x, const int y, std::vector<MirroredVector<float3> > &  boxIntersects,
                                   std::vector<MirroredVector<float2> > & raySteps) {
        _optimizer->debugPredictionRay(_mirroredModels,x,y,boxIntersects,raySteps);
    }

    inline const void setFilteredNorms(const bool filteredNorms) { _pcSource->setFilteredNorms(filteredNorms); }
    inline const void setFilteredVerts(const bool filteredVerts) { _pcSource->setFilteredVerts(filteredVerts); }
    inline const void setSigmaDepth(const float sigmaDepth) { _pcSource->setSigmaDepth(sigmaDepth); }
    inline const void setSigmaPixels(const float sigmaPixels) { _pcSource->setSigmaPixels(sigmaPixels); }

    inline const float4 * getCollisionCloud(const int modelNum) const { return _collisionClouds[modelNum]->hostPtr(); }
    inline const float4 * getDeviceCollisionCloud(const int modelNum) const { return _collisionClouds[modelNum]->devicePtr(); }
    inline int getCollisionCloudSize(const int modelNum) const { return _collisionClouds[modelNum]->length(); }
    inline int getCollisionCloudSdfStart(const int modelNum, const int sdfNum) const { return _collisionCloudSdfStarts[modelNum][sdfNum]; }
    inline int getCollisionCloudSdfLength(const int modelNum, const int sdfNum) const { return _collisionCloudSdfLengths[modelNum][sdfNum]; }

    void setIntersectionPotentialMatrix(const int modelNum, const int * mx);
    const int * getIntersectionPotentialMatrix(const int modelNum) const { return _intersectionPotentialMatrices[modelNum]->hostPtr(); }

    const Optimizer * getOptimizer() const { return _optimizer; }
    void cropBox(const float3 & min, const float3 & max) { _pcSource->cropBox(min,max); }
    void maskPointCloud(const int * deviceMask) { _pcSource->maskPointCloud(deviceMask); }

    Eigen::MatrixXf & getDampingMatrix(const int modelNum) { return *_dampingMatrices[modelNum]; }

    // TODO
    PointCloudSourceBase & getPointCloudSource() { return *_pcSource; }

    OptimizationOptions & getOptions() { return _opts; }

//protected:
    Optimizer * getOptimizer() { return _optimizer; }

    void addPrior(Prior * prior) { _priors.push_back(prior); }

private:

    inline bool initialized() { return _optimizer != 0 && _estimatedPoses.size() > 0; }

    DepthSourceBase * _depthSource;
    PointCloudSourceBase * _pcSource;
    Optimizer * _optimizer;
    OptimizationOptions _opts;

    std::vector<MirroredModel *> _mirroredModels;
    std::vector<std::map<std::string,float> > _sizeParams;
    std::vector<PoseReduction *> _ownedPoseReductions;
    std::vector<Pose> _estimatedPoses;

    std::vector<std::string> _filenames;

    std::vector<Eigen::MatrixXf *> _dampingMatrices;
    std::vector<Prior *> _priors;

    // collision stuff
    std::vector<MirroredVector<float4> *> _collisionClouds;
    std::vector<std::vector<int> > _collisionCloudSdfStarts;
    std::vector<std::vector<int> > _collisionCloudSdfLengths;
    std::vector<MirroredVector<int> *> _intersectionPotentialMatrices;

    // scratch space
    MirroredVector<SE3> * _T_mcs;
    MirroredVector<SE3 *> * _T_fms;
    MirroredVector<int *> * _sdfFrames;
    MirroredVector<const Grid3D<float> *> * _sdfs;
    MirroredVector<int> * _nSdfs;
    MirroredVector<float> * _distanceThresholds;
    MirroredVector<float> * _normalThresholds;
    MirroredVector<float> * _planeOffsets;
    MirroredVector<float3> * _planeNormals;
};

}

#endif // TRACKER_H
