#ifndef DEPTHSENSE_DEPTH_SOURCE_H
#define DEPTHSENSE_DEPTH_SOURCE_H

#include "depth_source.h"

#include "DepthSense.hxx"

#include <pthread.h>

namespace dart {

class DepthSenseDepthSource : public DepthSource<ushort,uchar3> {
public:
    DepthSenseDepthSource();
    ~DepthSenseDepthSource();

    bool initialize(const bool getColor=true,
                    const bool enableDenoising=true,
                    const uint confidenceThreshold=250);

    const ushort * getDepth() const { return _depthData; }

    const ushort * getDeviceDepth() const { return _deviceDepthData; }

    const uchar3 * getColor() const { return _colorData; }

    ColorLayout getColorLayout() const { return LAYOUT_BGR; }

    uint64_t getDepthTime() const { return _depthTime; }

    uint64_t getColorTime() const { return _colorTime; }

    void setFrame(const uint frame) { }

    void advance();

    bool hasRadialDistortionParams() const { return false; }

    float getScaleToMeters() const { return 1.e-3; }

private:
    void updateDepth(DepthSense::DepthNode node, DepthSense::DepthNode::NewSampleReceivedData data);
    void updateColor(DepthSense::ColorNode node, DepthSense::ColorNode::NewSampleReceivedData data);

    DepthSense::Context _context;
    DepthSense::Device _device;
    DepthSense::DepthNode _depthNode;
    DepthSense::ColorNode _colorNode;
    uint64_t _colorTime;
    uint64_t _depthTime;
    pthread_mutex_t _depthMutex;
    pthread_mutex_t _colorMutex;
    ushort * _depthData;
    ushort * _deviceDepthData;
    uchar3 * _colorData;

    ushort * _nextDepthData;
    uint64_t _nextDepthTime;
    uchar3 * _nextColorData;
    uint64_t _nextColorTime;
};

}

#endif // DEPTHSENSE_DEPTH_SOURCE_H
