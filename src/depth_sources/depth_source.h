#ifndef DEPTH_SOURCE_H
#define DEPTH_SOURCE_H

#include <stdint.h>
#include <sys/types.h>
#include <vector_types.h>

namespace dart {

enum ColorLayout {
    LAYOUT_RGB, LAYOUT_BGR
};

class DepthSourceBase {

public:
    DepthSourceBase() :
        _hasColor(false),
        _hasTimestamps(false),
        _isLive(false),
        _depthWidth(0),
        _depthHeight(0),
        _colorWidth(0),
        _colorHeight(0),
        _frame(0) { }

    virtual ColorLayout getColorLayout() const { return LAYOUT_RGB; }

    inline bool hasColor() const { return _hasColor; }

    inline bool hasTimestamps() const { return _hasTimestamps; }

    inline bool isLive() const { return _isLive; }

    inline uint getDepthWidth() const { return _depthWidth; }

    inline uint getDepthHeight() const { return _depthHeight; }

    inline uint getColorWidth() const { return _colorWidth; }

    inline uint getColorHeight() const { return _colorHeight; }

    inline float2 getFocalLength() const { return _focalLength; }
    inline void setFocalLength(const float2 focalLength) { _focalLength = focalLength; }

    inline float2 getPrincipalPoint() const { return _principalPoint; }
    inline void setPrincipalPoint(const float2 principalPoint) { _principalPoint = principalPoint; }

    virtual uint64_t getDepthTime() const { return 0; }

    virtual uint64_t getColorTime() const { return 0; }

    virtual uint getFrame() const { return _frame; }

    virtual void setFrame(const uint frame) = 0;

    virtual void advance() = 0;

    virtual bool hasRadialDistortionParams() const = 0;

    virtual const float * getRadialDistortionParams() const { return 0; }

    virtual float getScaleToMeters() const { return 1.0f; }

protected:
    bool _hasColor;
    bool _hasTimestamps;
    bool _isLive;
    uint _depthWidth;
    uint _depthHeight;
    uint _colorWidth;
    uint _colorHeight;
    uint _frame;
    float2 _focalLength;
    float2 _principalPoint;

};

template <typename DepthType, typename ColorType>
class DepthSource : public DepthSourceBase {
public:

    virtual const DepthType * getDepth() const = 0;
    virtual const DepthType * getDeviceDepth() const = 0;

    virtual const ColorType * getColor() const { return 0; }
};

}

#endif // DEPTH_SOURCE_H
