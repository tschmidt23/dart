#ifndef POINT_CLOUD_SRC_H
#define POINT_CLOUD_SRC_H

#include <string.h>

#include "depth_sources/depth_source.h"
#include "img_proc/bilateral_filter.h"
#include "img_proc/organized_point_cloud.h"
#include "util/mirrored_memory.h"

namespace dart {

class BackProjector {
public:
    virtual void backProjectDepthMap(const float * depthMap,
                                     float4 * vertMap,
                                     const int width,
                                     const int height) const = 0;
    virtual void backProjectDepthMap(const ushort * depthMap,
                                     float4 * vertMap,
                                     const int width,
                                     const int height) const = 0;
};

class BackProjectorFocalLengthPrincipalPoint : public BackProjector {
public :
    BackProjectorFocalLengthPrincipalPoint(const float2 fl, const float2 pp, const float2 range) :
        _fl(fl), _pp(pp), _range(range) {}
    void backProjectDepthMap(const float * depthMap, float4 * vertMap, const int width, const int height) const {
        depthToVertices(depthMap,vertMap,width,height,_pp,_fl,_range);
    }
    void backProjectDepthMap(const ushort * depthMap, float4 * vertMap, const int width, const int height) const {
        depthToVertices(depthMap,vertMap,width,height,_pp,_fl,_range);
    }
private:
    float2 _pp, _fl, _range;
};

class BackProjectorFocalLengthPrincipalPointScaled : public BackProjector {
public :
    BackProjectorFocalLengthPrincipalPointScaled(const float2 fl, const float2 pp, const float2 range, const float scale) :
        _fl(fl), _pp(pp), _range(range), _scale(scale) {}
    void backProjectDepthMap(const float * depthMap, float4 * vertMap, const int width, const int height) const {
        depthToVertices(depthMap,vertMap,width,height,_pp,_fl,_range,_scale);
    }
    void backProjectDepthMap(const ushort * depthMap, float4 * vertMap, const int width, const int height) const {
        depthToVertices(depthMap,vertMap,width,height,_pp,_fl,_range,_scale);
    }
private:
    float2 _pp, _fl, _range;
    float _scale;
};

class BackProjectorCalibrationParams : public BackProjector {
public:
    BackProjectorCalibrationParams(const float * calibrationParams, const float2 range) :
        _range(range), _calibrationParams(9) {
        memcpy(_calibrationParams.hostPtr(),calibrationParams,9*sizeof(float));
        _calibrationParams.syncHostToDevice();
    }
    void backProjectDepthMap(const float * depthMap, float4 * vertMap, const int width, const int height) const {
        depthToVertices(depthMap,vertMap,width,height,_calibrationParams.devicePtr(),_range);
    }
    void backProjectDepthMap(const ushort * depthMap, float4 * vertMap, const int width, const int height) const {
        depthToVertices(depthMap,vertMap,width,height,_calibrationParams.devicePtr(),_range);
    }
private:
    MirroredVector<float> _calibrationParams;
    float2 _range;
};

class BackProjectorCalibrationParamsScaled : public BackProjector {
public:
    BackProjectorCalibrationParamsScaled(const float * calibrationParams, const float2 range, const float scale) :
        _range(range), _scale(scale), _calibrationParams(9) {
        memcpy(_calibrationParams.hostPtr(),calibrationParams,9*sizeof(float));
        _calibrationParams.syncHostToDevice();
    }
    void backProjectDepthMap(const float * depthMap, float4 * vertMap, const int width, const int height) const {
        depthToVertices(depthMap,vertMap,width,height,_calibrationParams.devicePtr(),_range,_scale);
    }
    void backProjectDepthMap(const ushort * depthMap, float4 * vertMap, const int width, const int height) const {
        depthToVertices(depthMap,vertMap,width,height,_calibrationParams.devicePtr(),_range,_scale);
    }
private:
    MirroredVector<float> _calibrationParams;
    float2 _range;
    float _scale;
};


class PointCloudSourceBase {

public:

    // pass-through functions
    inline bool hasColor() const { return _depthSource->hasColor(); }
    inline ColorLayout getColorLayout() const { return _depthSource->getColorLayout(); }
    inline bool hasTimestamps() const { return _depthSource->hasTimestamps(); }
    inline uint64_t getDepthTime() const { return _depthSource->getDepthTime(); }
    inline uint64_t getColorTime() const { return _depthSource->getColorTime(); }
    virtual uint getDepthWidth() const { return _depthSource->getDepthWidth(); }
    virtual uint getDepthHeight() const { return _depthSource->getDepthHeight(); }
    inline uint getColorWidth() const { return _depthSource->getColorWidth(); }
    inline uint getColorHeight() const { return _depthSource->getColorHeight(); }
    inline uint getFrame() const { return _depthSource->getFrame(); }
    inline void setFrame(const uint frame) { _depthSource->setFrame(frame); }
    virtual float2 getFocalLength() const { return _depthSource->getFocalLength(); }
    inline float2 getPrincipalPoint() const { return _depthSource->getPrincipalPoint(); }
    inline void setFilteredNorms(bool filteredNorms) { _filteredNorms = filteredNorms; }
    inline void setFilteredVerts(bool filteredVerts) { _filteredVerts = filteredVerts; }
    inline void setSigmaDepth(float sigmaDepth) { _sigmaDepth = sigmaDepth; }
    inline void setSigmaPixels(float sigmaPixels) { _sigmaPixels = sigmaPixels; }

    // point cloud functions
    virtual float4 * getDeviceVertMap() { return _vertMap->devicePtr(); }
    virtual float4 * getDeviceNormMap() { return _normMap->devicePtr(); }
    virtual const float4 * getDeviceVertMap() const { return _vertMap->devicePtr(); }
    virtual const float4 * getDeviceNormMap() const { return _normMap->devicePtr(); }

    // TODO: get rid of this, used for injecting arbitrary point clouds
    inline void setVertMap(const float4 * vertMap) {
        memcpy(_vertMap->hostPtr(),vertMap,_vertMap->length()*sizeof(float4));
        _vertMap->syncHostToDevice();
        _vertsOnHost = true;
    }

    inline const float4 * getHostVertMap() {
        if (!_vertsOnHost) {
            _vertMap->syncDeviceToHost();
            _vertsOnHost = true;
        }
        return _vertMap->hostPtr();
    }

    inline const float4 * getHostNormMap() {
        if (!_normsOnHost) {
            _normMap->syncDeviceToHost();
            _normsOnHost = true;
        }
        return _normMap->hostPtr();
    }

    inline void cropBox(const float3 & min, const float3 & max) {
        dart::cropBox(_vertMap->devicePtr(),getDepthWidth(),getDepthHeight(),min,max);
        _vertsOnHost = false;
    }

    inline void maskPointCloud(const int * deviceMask) {
        dart::maskPointCloud(_vertMap->devicePtr(),getDepthWidth(),getDepthHeight(),deviceMask);
        _vertsOnHost = false;
    }

    virtual void advance() = 0;

    inline void setFocalLength(const float2 focalLength) {
        _depthSource->setFocalLength(focalLength);
        delete _backProjector;
        float s = _depthSource->getScaleToMeters();
        if (s == 1.0f) { _backProjector = new BackProjectorFocalLengthPrincipalPoint(_depthSource->getFocalLength(),
                                                                                     _depthSource->getPrincipalPoint(),
                                                                                     make_float2(0,3));
        } else { _backProjector = new BackProjectorFocalLengthPrincipalPointScaled(_depthSource->getFocalLength(),
                                                                                   _depthSource->getPrincipalPoint(),
                                                                                   make_float2(0,3), s); // TODO
        }
    }

    inline void setPrincipalPoint(const float2 principalPoint) {
        _depthSource->setPrincipalPoint(principalPoint);
        delete _backProjector;
        float s = _depthSource->getScaleToMeters();
        if (s == 1.0f) { _backProjector = new BackProjectorFocalLengthPrincipalPoint(_depthSource->getFocalLength(),
                                                                                     _depthSource->getPrincipalPoint(),
                                                                                     make_float2(0,3));
        } else { _backProjector = new BackProjectorFocalLengthPrincipalPointScaled(_depthSource->getFocalLength(),
                                                                                   _depthSource->getPrincipalPoint(),
                                                                                   make_float2(0,3), s); // TODO
        }
    }

protected:

    // -=-=-=-=- methods -=-=-=-=-
    virtual void projectDepthMap() = 0;

    // -=-=-=-=- members -=-=-=-=-
    DepthSourceBase * _depthSource;
    BackProjector * _backProjector;

    MirroredVector<float4> * _vertMap;
    MirroredVector<float4> * _normMap;

    bool _vertsOnHost, _normsOnHost;
    bool _filteredVerts, _filteredNorms;
    float _sigmaDepth, _sigmaPixels;

    float * _dFilteredDepth;

};

template <typename DepthType, typename ColorType>
class PointCloudSource : public PointCloudSourceBase {

public:
    PointCloudSource(DepthSource<DepthType,ColorType> * depthSource,
                     float2 range) {
        _depthSource = depthSource;
        float s = depthSource->getScaleToMeters();
        if (depthSource->hasRadialDistortionParams()) {
            float calibrationParams[9];
            calibrationParams[0] = depthSource->getFocalLength().x;
            calibrationParams[1] = depthSource->getFocalLength().y;
            calibrationParams[2] = depthSource->getPrincipalPoint().x;
            calibrationParams[3] = depthSource->getPrincipalPoint().y;
            memcpy(&calibrationParams[4],depthSource->getRadialDistortionParams(),5*sizeof(float));
            if (s != 1.0f) {
                _backProjector = new BackProjectorCalibrationParamsScaled(calibrationParams,range,s);
            } else {
                _backProjector = new BackProjectorCalibrationParams(calibrationParams,range);
            }
        } else {
            if (s != 1.0f) {
                _backProjector = new BackProjectorFocalLengthPrincipalPointScaled(depthSource->getFocalLength(),
                                                                                  depthSource->getPrincipalPoint(),
                                                                                  range,s);
            } else {
                _backProjector = new BackProjectorFocalLengthPrincipalPoint(depthSource->getFocalLength(),
                                                                            depthSource->getPrincipalPoint(),
                                                                            range);
            }
        }
        float w = depthSource->getDepthWidth();
        float h = depthSource->getDepthHeight();
        cudaMalloc(&_dFilteredDepth,w*h*sizeof(float));
        _vertMap = new MirroredVector<float4>(w*h);
        _normMap = new MirroredVector<float4>(w*h);
        _vertsOnHost = _normsOnHost = false;
        _filteredVerts = false;
        _filteredNorms = false;
        _sigmaDepth = 0.10/_depthSource->getScaleToMeters();
        _sigmaPixels = 3.0;

        projectDepthMap();
    }
    ~PointCloudSource() {
        delete _backProjector;
        delete _vertMap;
        delete _normMap;
        cudaFree(_dFilteredDepth);
    }

    const DepthType * getDeviceDepth();

    inline void advance() {
        _vertsOnHost =  _normsOnHost = false;
        _depthSource->advance();
        projectDepthMap();
    }

    inline void projectDepthMap() {
        DepthSource<DepthType,ColorType> * depthSource = static_cast<DepthSource<DepthType,ColorType> *>(_depthSource);
        if (_filteredNorms) {
            bilateralFilter(depthSource->getDeviceDepth(),_dFilteredDepth,
                            depthSource->getDepthWidth(),depthSource->getDepthHeight(),
                            _sigmaPixels,_sigmaDepth);
            _backProjector->backProjectDepthMap(_dFilteredDepth,
                                                _vertMap->devicePtr(),
                                                _depthSource->getDepthWidth(),
                                                _depthSource->getDepthHeight());
            verticesToNormals(_vertMap->devicePtr(),_normMap->devicePtr(),depthSource->getDepthWidth(),depthSource->getDepthHeight());
            if (!_filteredVerts) {
                _backProjector->backProjectDepthMap(depthSource->getDeviceDepth(),
                                                    _vertMap->devicePtr(),
                                                    depthSource->getDepthWidth(),
                                                    depthSource->getDepthHeight());
            }
        } else {
            _backProjector->backProjectDepthMap(depthSource->getDeviceDepth(),
                                                _vertMap->devicePtr(),
                                                depthSource->getDepthWidth(),
                                                depthSource->getDepthHeight());
            verticesToNormals(_vertMap->devicePtr(),_normMap->devicePtr(),depthSource->getDepthWidth(),depthSource->getDepthHeight());
        }
    }

};


}


#endif // POINT_CLOUD_SRC_H
