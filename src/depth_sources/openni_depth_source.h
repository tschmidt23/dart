#ifndef OPENNI_DEPTH_SOURCE_H
#define OPENNI_DEPTH_SOURCE_H

#include "depth_source.h"
#include <OpenNI.h>
#include <vector_types.h>

namespace dart {

class OpenNIDepthSource : public DepthSource<ushort,uchar3> {
public:
    OpenNIDepthSource();

    ~OpenNIDepthSource();

    bool initialize(const char * deviceURI = openni::ANY_DEVICE,
                    const bool getColor = true,
                    const uint depthWidth = 640,
                    const uint depthHeight = 480,
                    const uint depthFPS = 30,
                    const uint colorWidth = 640,
                    const uint colorHeight = 480,
                    const uint colorFPS = 30,
                    const bool mirror = false,
                    const bool frameSync = true,
                    const bool registerDepth = true);

    const ushort * getDepth() const;

    const ushort * getDeviceDepth() const;

    const uchar3 * getColor() const;

    ColorLayout getColorLayout() const { return LAYOUT_RGB; }

    uint64_t getDepthTime() const;

    uint64_t getColorTime() const;

    void setFrame(const uint frame);

    void advance();

    bool hasRadialDistortionParams() const { return false; }

    inline float getScaleToMeters() const { return 1/(1000.0f); }

private:
    openni::Device _device;
    openni::VideoStream _depthStream;
    openni::VideoStream _colorStream;
    openni::VideoFrameRef _depthFrame;
    openni::VideoFrameRef _colorFrame;
    int _frameIndexOffset;
    ushort * _deviceDepth;
};

}

#endif // OPENNI_DEPTH_SOURCE_H
