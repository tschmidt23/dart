#ifndef PRIORS_H
#define PRIORS_H

#include <Eigen/Sparse>
#include "optimization.h"
#include "model/mirrored_model.h"

namespace dart {

class Prior {
public:

    virtual int getNumPriorParams() const { return 0; }

    virtual float * getPriorParams() { return 0; }

    virtual void computeContribution(Eigen::SparseMatrix<float> & JTJ,
                                     Eigen::VectorXf & JTe,
                                     const int * modelOffsets,
                                     const int priorParamOffset,
                                     const std::vector<MirroredModel *> & models,
                                     const std::vector<Pose> & poses,
                                     const OptimizationOptions & opts) = 0;

    virtual void updatePriorParams(const float * update,
                                   const std::vector<MirroredModel *> & models) { };
};

class ContactPrior : public Prior {
public:

    ContactPrior(const int srcModelID, const int dstModelID, const int srcSdfNum, const int dstSdfNum, const float weight, const float3 initialContact, float regularization = 100.f) :
        _srcModelID(srcModelID), _dstModelID(dstModelID), _srcSdfNum(srcSdfNum), _dstSdfNum(dstSdfNum), _contactPoint(initialContact), _weight(weight), _regularization(regularization), _error(0) { }

    inline int getNumPriorParams() const { return 3; }

    float * getPriorParams() { return (float *) &_contactPoint; }

    void computeContribution(Eigen::SparseMatrix<float> & JTJ,
                             Eigen::VectorXf & JTe,
                             const int * modelOffsets,
                             const int priorParamOffset,
                             const std::vector<MirroredModel *> & models,
                             const std::vector<Pose> & poses,
                             const OptimizationOptions & opts);

    void updatePriorParams(const float * update,
                           const std::vector<MirroredModel*> & models);

    void setWeight(float weight) { _weight = weight; }

    float3 getContactPoint() const { return _contactPoint; }
    int getSourceModel() const { return _srcModelID; }
    int getDestinationModel() const { return _dstModelID; }
    int getSourceSdfNum() const { return _srcSdfNum; }
    int getDestinationSdfNum() const { return _dstSdfNum; }
    float getWeight() const { return _weight; }
    float getError() const { return _error; }

    std::vector<float3> & getSrcJ3D() { return srcJ3D; } // TODO: remove
    std::vector<float3> & getDstJ3D() { return dstJ3D; } // TODO: remove

private:
    int _srcModelID, _dstModelID;
    int _srcSdfNum, _dstSdfNum;
    float3 _contactPoint;
    float _weight, _regularization, _error;
    std::vector<float3> srcJ3D, dstJ3D; // TODO: remove
};

class Point3D3DPrior : public Prior {
public:
    Point3D3DPrior(const int srcModelID, const int srcFrame, const float3 point_c, const float3 point_f, const float weight) :
        _srcModelID(srcModelID), _srcFrame(srcFrame), _point_c(point_c), _point_f(point_f), _weight(weight) { }

    void computeContribution(Eigen::SparseMatrix<float> & JTJ,
                             Eigen::VectorXf & JTe,
                             const int * modelOffsets,
                             const int priorParamOffset,
                             const std::vector<MirroredModel *> & models,
                             const std::vector<Pose> & poses,
                             const OptimizationOptions & opts);

    void setWeight(float weight) { _weight = weight; }
    void setTargetCameraPoint(float3 point_c) { _point_c = point_c; }

    float3 getTargetCameraPoint() const { return _point_c; }
    float3 getSourceFramePoint() const { return _point_f; }
    int getSourceModelID() const { return _srcModelID; }
    int getSourceFrame() const { return _srcFrame; }

    float3 getDistGrad_m() const { return distGrad_m; } // TODO: remove
    std::vector<float3> & getSrcJ3D() { return srcJ3D; } // TODO: remove

private:
    int _srcModelID, _srcFrame;
    float _weight;
    float3 _point_c, _point_f;
    float3 distGrad_m; // TODO: remove
    std::vector<float3> srcJ3D; // TODO: remove
};

class Point2D3DPrior : public Prior {
public:
    Point2D3DPrior(const int srcModelID, const int srcFrame, const float2 point_c, const float3 point_f, const float2 principalPoint, const float2 focalLength, const float weight) :
        _srcModelID(srcModelID), _srcFrame(srcFrame), _point_c(point_c), _point_f(point_f), _principalPoint(principalPoint), _focalLength(focalLength), _weight(weight) { }

    void computeContribution(Eigen::SparseMatrix<float> & JTJ,
                             Eigen::VectorXf & JTe,
                             const int * modelOffsets,
                             const int priorParamOffset,
                             const std::vector<MirroredModel *> & models,
                             const std::vector<Pose> & poses,
                             const OptimizationOptions & opts);

    void setWeight(float weight) { _weight = weight; }
    void setTargetCameraPoint(float2 point_c) { _point_c = point_c; }

    float2 getTargetCameraPoint() const { return _point_c; }
    float3 getSourceFramePoint() const { return _point_f; }
    int getSourceModelID() const { return _srcModelID; }
    int getSourceFrame() const { return _srcFrame; }

    float3 getDistGrad_m() const { return pointGrad_m; } // TODO: remove
    std::vector<float3> & getSrcJ3D() { return srcJ3D; } // TODO: remove

private:
    int _srcModelID, _srcFrame;
    float _weight;
    float2 _point_c;
    float3 _point_f;
    float2 _principalPoint, _focalLength;
    float3 pointGrad_m; // TODO: remove
    std::vector<float3> srcJ3D; // TODO: remove
};

} // namespace dart

#endif // PRIORS_H
