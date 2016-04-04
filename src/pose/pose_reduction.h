#ifndef POSE_REDUCTION_H
#define POSE_REDUCTION_H

#include <string>
#include <vector>
#include "util/mirrored_memory.h"

namespace dart {

class PoseReduction {
public:
    PoseReduction(int fullDims, int redDims) :
        _fullDims(fullDims), _redDims(redDims),
        _mins(redDims), _maxs(redDims), _names(redDims) { }
    inline int getFullDimensions() { return _fullDims; }
    inline int getReducedDimensions() { return _redDims; }
    virtual void projectReducedToFull(const float * reduced, float * full) const = 0;
    virtual const float * getFirstDerivatives() const = 0;
    virtual const float * getDeviceFirstDerivatives() const = 0;
    virtual bool isNull() const { return false; }

    inline float getMin(const int dim) const { return _mins[dim]; }
    inline float getMax(const int dim) const { return _maxs[dim]; }
    inline const std::string & getName(const int dim) const { return _names[dim]; }
protected:
    int _fullDims;
    int _redDims;
    std::vector<float> _mins;
    std::vector<float> _maxs;
    std::vector<std::string> _names;
};

class LinearPoseReduction : public PoseReduction {
public:
    LinearPoseReduction(const int fullDims, const int redDims);

    LinearPoseReduction(const int fullDims, const int redDims, float * A, float * b, float * mins, float * maxs, std::string * names);

    ~LinearPoseReduction() { delete [] _b; }

    void init(float * A, float * b, float * mins, float * maxs, std::string * names);

    virtual void projectReducedToFull(const float * reduced, float * full) const;

    inline const float * getFirstDerivatives() const { return _A.hostPtr(); }

    inline float getFirstDerivative(const int reducedDim, const int fullDim) { return _A[reducedDim + fullDim*_redDims]; }

    inline const float * getDeviceFirstDerivatives() const { return _A.devicePtr(); }

    inline virtual bool isParamMap() const { return false; }

protected:
    MirroredVector<float> _A;
    float * _b;
};

class ParamMapPoseReduction : public LinearPoseReduction {
public:
    ParamMapPoseReduction(const int fullDims, const int redDims, const int * mapping, float * mins, float * maxs, std::string * names);

    ~ParamMapPoseReduction();

    void projectReducedToFull(const float * reduced, float * full) const;

    inline const int * getMapping() const { return _mapping.hostPtr(); }

    inline const int * getDeviceMapping() const { return _mapping.devicePtr(); }

    inline bool isParamMap() const { return true; }

private:

    MirroredVector<int> _mapping;
};

class NullReduction : public PoseReduction {
public:
    NullReduction(const int dims, const float * mins, const float * maxs, const std::string * names) : PoseReduction(dims,dims) {
        memcpy(_mins.data(),mins,dims*sizeof(float));
        memcpy(_maxs.data(),maxs,dims*sizeof(float));
        for (int i=0; i<dims; ++i) { _names[i] = names[i]; }
    }
    ~NullReduction() { }
    inline bool isNull() const { return true; }

    void projectReducedToFull(const float * reduced, float * full) const {  }
    inline const float * getFirstDerivatives() const { return 0; }
    inline const float * getDeviceFirstDerivatives() const { return 0; }
};

}

#endif // POSE_REDUCTION_H
