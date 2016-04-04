#include "depthsense_depth_source.h"

#include <iostream>
#include <string.h>
#include <vector>
#include <cuda_runtime.h>
#include <vector_types.h>

namespace dart {

void * startContext(void * wrapper) {
    DepthSense::Context * context = (DepthSense::Context *)wrapper;
    context->run();
    pthread_exit(NULL);
}

DepthSenseDepthSource::DepthSenseDepthSource() : DepthSource<ushort,uchar3>() {

    _depthData = 0;
    _colorData = 0;
    _depthTime = _colorTime = 0;

    _nextDepthData = 0;
    _nextColorData = 0;
    _nextDepthTime = _nextColorTime = 0;

}

DepthSenseDepthSource::~DepthSenseDepthSource() {

    if (_context.isSet()) {
        _context.stopNodes();
        _context.releaseControl(_device);
        _context.unregisterNode(_depthNode);
        if (_colorNode.isSet()) {
            _context.unregisterNode(_colorNode);
        }
    }

    delete [] _depthData;
    cudaFree(_deviceDepthData);
    delete [] _colorData;

    delete [] _nextDepthData;
    delete [] _nextColorData;

}

bool DepthSenseDepthSource::initialize(const bool getColor,
                                       const bool enableDenoising,
                                       const uint confidenceThreshold) {

    _context = DepthSense::Context::createStandalone();

    std::vector<DepthSense::Device> devices = _context.getDevices();

    if (devices.size() == 0) {
        std::cerr << "could not find any DepthSense devices" << std::endl;
        return false;
    }

    _device = devices[0];

    std::vector<DepthSense::Node> nodes = _device.getNodes();

    for (int i=0; i<nodes.size(); ++i) {
        if (nodes[i].is<DepthSense::DepthNode>()) {
            _depthNode = (DepthSense::DepthNode)nodes[i];
        }
    }

    if (!_depthNode.isSet()) {
        std::cerr << "could not find depth node" << std::endl;
        return false;
    }

    if (getColor) {
        for (int i=0; i<nodes.size(); ++i) {
            if (nodes[i].is<DepthSense::ColorNode>()) {
                _colorNode = (DepthSense::ColorNode)nodes[i];
            }
        }

        if (!_colorNode.isSet()) {
            std::cerr << "could not find color node" << std::endl;
            return false;
        }

        _hasColor = true;
    }

    _context.requestControl(_device);

    _context.registerNode(_depthNode);

    _depthNode.setEnableDepthMap(true);
    _depthNode.setEnableDenoising(enableDenoising);
    _depthNode.setConfidenceThreshold(confidenceThreshold);

    if (getColor) {
        _context.registerNode(_colorNode);
        _colorNode.setEnableColorMap(true);
        DepthSense::ColorNode::Configuration config = _colorNode.getConfiguration();
//        config.frameFormat = DepthSense::FRAME_FORMAT_QVGA;
        config.compression = DepthSense::COMPRESSION_TYPE_MJPEG;
//        config.framerate = 30;
        _colorNode.setConfiguration(config);
    }

    DepthSense::StereoCameraParameters parameters = _device.getStereoCameraParameters();

    _depthWidth = parameters.depthIntrinsics.width;
    _depthHeight = parameters.depthIntrinsics.height;

    _focalLength = make_float2(228.0,228.0);
    _principalPoint = make_float2(_depthWidth/2,_depthHeight/2);

    _depthData = new ushort[_depthWidth*_depthHeight];
    memset(_depthData,0,_depthWidth*_depthHeight*sizeof(ushort));
    cudaMalloc(&_deviceDepthData,_depthWidth*_depthHeight*sizeof(ushort));
    _nextDepthData = new ushort[_depthWidth*_depthHeight];

    if (getColor) {
        _colorWidth = parameters.colorIntrinsics.width;
        _colorHeight = parameters.colorIntrinsics.height;

        _colorData = new uchar3[_colorWidth*_colorHeight];
        memset(_colorData,0,_colorWidth*_colorHeight*sizeof(uchar3));
        _nextColorData = new uchar3[_colorWidth*_colorHeight];
    }

    _hasTimestamps = true;
    _isLive = true;

    _depthTime = 0;
    _colorTime = 0;

    pthread_mutex_init(&_depthMutex,NULL);
    pthread_mutex_init(&_colorMutex,NULL);
    pthread_t contextThread;
    pthread_create(&contextThread, NULL, startContext, (void*)&_context);

    _depthNode.newSampleReceivedEvent().connect<DepthSenseDepthSource>(this, &DepthSenseDepthSource::updateDepth);
    if (getColor) {
        _colorNode.newSampleReceivedEvent().connect<DepthSenseDepthSource>(this, &DepthSenseDepthSource::updateColor);
    }

    _context.startNodes();

    advance();

    return true;
}

void DepthSenseDepthSource::advance() {

    // swap depth buffers
    pthread_mutex_lock(&_depthMutex);
    if (_nextDepthTime > _depthTime) {
        ushort * tmp = _depthData;
        _depthData = _nextDepthData;
        _nextDepthData = tmp;
        _depthTime = _nextDepthTime;
        _frame++;
    }
    pthread_mutex_unlock(&_depthMutex);

    // swap color buffers
    pthread_mutex_lock(&_colorMutex);
    if (_nextColorTime > _colorTime) {
        uchar3 * tmp = _colorData;
        _colorData = _nextColorData;
        _nextColorData = tmp;
        _colorTime = _nextColorTime;
    }
    pthread_mutex_unlock(&_colorMutex);

    cudaMemcpy(_deviceDepthData,_depthData,_depthWidth*_depthHeight*sizeof(ushort),cudaMemcpyHostToDevice);

}

void DepthSenseDepthSource::updateDepth(DepthSense::DepthNode node, DepthSense::DepthNode::NewSampleReceivedData data) {

    pthread_mutex_lock(&_depthMutex);
    _nextDepthTime = data.timeOfCapture;
    memcpy(_nextDepthData,data.depthMap,_depthWidth*_depthHeight*sizeof(ushort));
    pthread_mutex_unlock(&_depthMutex);

}

void DepthSenseDepthSource::updateColor(DepthSense::ColorNode node, DepthSense::ColorNode::NewSampleReceivedData data) {

    pthread_mutex_lock(&_colorMutex);
    _nextColorTime = data.timeOfCapture;
    memcpy(_nextColorData,data.colorMap,_colorWidth*_colorHeight*sizeof(uchar3));
    pthread_mutex_unlock(&_colorMutex);

}

}
