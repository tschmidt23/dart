#ifndef POSE_H
#define POSE_H

#include <string>
#include <vector>
#include <string.h>
#include "pose_reduction.h"
#include "util/mirrored_memory.h"
#include "geometry/SE3.h"

namespace dart {

class Pose {
public:
    Pose(PoseReduction * reduction);

    Pose(const Pose & pose);

    ~Pose();

    inline bool isReduced() const { return !_reduction->isNull(); }

    inline int getDimensions() const { return _reduction->getFullDimensions() + 6; }

    inline int getArticulatedDimensions() const { return _reduction->getFullDimensions(); }

    inline int getReducedDimensions() const { return _reduction->getReducedDimensions() + 6; }

    inline int getReducedArticulatedDimensions() const { return _reduction->getReducedDimensions(); }

    inline float * getArticulation() { return _fullArticulation; }

    inline float * getReducedArticulation() { return _reducedArticulation; }

    inline const float * getArticulation() const { return _fullArticulation; }

    inline const float * getReducedArticulation() const { return _reducedArticulation; }

    virtual void projectReducedToFull() { return _reduction->projectReducedToFull(_reducedArticulation,_fullArticulation); }

    inline const float * getFirstDerivatives() const { return _reduction->getFirstDerivatives(); }

    inline const float * getDeviceFirstDerivatives() const { return _reduction->getDeviceFirstDerivatives(); }

    inline float getReducedMin(const int dim) const { return _reduction->getMin(dim); }

    inline float getReducedMax(const int dim) const { return _reduction->getMax(dim); }

    inline const std::string & getReducedName(const int dim) const { return _reduction->getName(dim); }

    Pose & operator=(const Pose & rhs);

    inline void zero() { memset(_reducedArticulation,0,getReducedArticulatedDimensions()*sizeof(float)); }

    inline SE3 & getTransformCameraToModel() { return _T_mc; }

    inline const SE3 & getTransformCameraToModel() const { return _T_mc; }

    inline SE3 & getTransformModelToCamera() { return _T_cm; }

    inline const SE3 & getTransformModelToCamera() const { return _T_cm; }

    inline void setTransformCameraToModel(const SE3 T_mc) { _T_mc = T_mc; _T_cm = SE3Invert(T_mc); }

    inline void setTransformModelToCamera(const SE3 T_cm) { _T_cm = T_cm; _T_mc = SE3Invert(T_cm); }

    inline const PoseReduction * getReduction() const { return _reduction; }

protected:
    float * _fullArticulation;
    float * _reducedArticulation;
    PoseReduction * _reduction;
    SE3 _T_mc;
    SE3 _T_cm;
};


class PosePrior {
public:
    PosePrior(const int length) : _pose(length), _weights(length) { }
    int getLength() const { return _pose.size(); }
    float * getPose() { return _pose.data(); }
    const float * getPose() const { return _pose.data(); }
    float * getWeights() { return _weights.data(); }
    const float * getWeights() const { return _weights.data(); }
private:
    std::vector<float> _pose;
    std::vector<float> _weights;
};

}


#endif // POSE_H
