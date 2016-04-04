#include "openni_depth_source.h"

#include <iostream>
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>

namespace dart {

OpenNIDepthSource::OpenNIDepthSource() : DepthSource<ushort,uchar3>() {

    openni::OpenNI::initialize();

    _deviceDepth = 0;

}

OpenNIDepthSource::~OpenNIDepthSource() {

    if (_colorStream.isValid()) {
        _colorStream.stop();
        _colorStream.destroy();
    }

    if (_depthStream.isValid()) {
        _depthStream.stop();
        _depthStream.destroy();
    }

    if (_device.isValid()) {
        _device.close();
    }

    if (_deviceDepth) {
        cudaFree(_deviceDepth);
    }

    openni::OpenNI::shutdown();

}

bool OpenNIDepthSource::initialize(const char * deviceURI,
                                   const bool getColor,
                                   const uint depthWidth,
                                   const uint depthHeight,
                                   const uint depthFPS,
                                   const uint colorWidth,
                                   const uint colorHeight,
                                   const uint colorFPS,
                                   const bool mirror,
                                   const bool frameSync,
                                   const bool registerDepth) {

    openni::Status rc = openni::STATUS_OK;

    // open device
    rc = _device.open(deviceURI);
    if (rc != openni::STATUS_OK) {
        std::cerr << "Could not open device: " << openni::OpenNI::getExtendedError() << std::endl;
        return false;
    }

    _isLive = !_device.isFile();

    // create depth stream
    rc = _depthStream.create(_device, openni::SENSOR_DEPTH);
    if (rc != openni::STATUS_OK) {
        std::cerr << "Could not create depth stream: " << openni::OpenNI::getExtendedError() << std::endl;
        return false;
    }

    if (_isLive) {

        // initialize depth stream settings
        int targetMode = -1;

        const openni::Array<openni::VideoMode>& modes = _device.getSensorInfo(openni::SENSOR_DEPTH)->getSupportedVideoModes();
        for (int i=0; i<modes.getSize(); ++i) {
            if (    modes[i].getResolutionX() == depthWidth &&
                    modes[i].getResolutionY() == depthHeight &&
                    modes[i].getPixelFormat() == openni::PIXEL_FORMAT_DEPTH_1_MM &&
                    modes[i].getFps() == depthFPS) {
                targetMode = i;
                break;
            }
        }

        if (targetMode == -1) {
            std::cerr << "Could not find depth video mode " << depthWidth << "x" << depthHeight << "@" << depthFPS << std::endl;
            return false;
        }

        rc = _depthStream.setVideoMode(modes[targetMode]);
        if (rc != openni::STATUS_OK) {
            std::cerr << "Could not set depth video mode: " << openni::OpenNI::getExtendedError() << std::endl;
            return false;
        }

        rc = _depthStream.setMirroringEnabled(mirror);
        if (rc != openni::STATUS_OK) {
            std::cerr << "Could not set depth mirroring mode: " << openni::OpenNI::getExtendedError() << std::endl;
            return false;
        }

        _depthWidth = depthWidth;
        _depthHeight = depthHeight;

    } else {

        openni::VideoMode depthMode = _depthStream.getVideoMode();

        _depthWidth = depthMode.getResolutionX();
        _depthHeight = depthMode.getResolutionY();

    }

    _focalLength = make_float2(525.*_depthWidth/640,525.*_depthWidth/640);
    _principalPoint = make_float2(_depthWidth/2,_depthHeight/2);

    cudaMalloc(&_deviceDepth,_depthWidth*_depthHeight*sizeof(ushort));

    if (getColor) {

        // create color stream
        rc = _colorStream.create(_device, openni::SENSOR_COLOR);
        if (rc != openni::STATUS_OK) {
            std::cerr << "Could not create color stream: " << openni::OpenNI::getExtendedError() << std::endl;
            return false;
        }

        if (_isLive) {

            // initialize color stream settings
            int targetMode = -1;

            const openni::Array<openni::VideoMode>& modes = _device.getSensorInfo(openni::SENSOR_COLOR)->getSupportedVideoModes();
            for (int i=0; i<modes.getSize(); ++i) {
                if (    modes[i].getResolutionX() == colorWidth &&
                        modes[i].getResolutionY() == colorHeight &&
                        modes[i].getPixelFormat() == openni::PIXEL_FORMAT_RGB888 &&
                        modes[i].getFps() == colorFPS) {
                    targetMode = i;
                    break;
                }
            }

            if (targetMode == -1) {
                std::cerr << "Could not find color video mode " << colorWidth << "x" << colorHeight << "@" << colorFPS << std::endl;
                return false;
            }

            rc = _colorStream.setVideoMode(modes[targetMode]);
            if (rc != openni::STATUS_OK) {
                std::cerr << "Could not set color video mode: " << openni::OpenNI::getExtendedError() << std::endl;
                return false;
            }

            rc = _colorStream.setMirroringEnabled(mirror);
            if (rc != openni::STATUS_OK) {
                std::cerr << "Could not set color mirroring mode: " << openni::OpenNI::getExtendedError() << std::endl;
                return false;
            }

            rc = _device.setDepthColorSyncEnabled(frameSync);
            if (rc != openni::STATUS_OK) {
                std::cerr << "Could not set frame sync: " << openni::OpenNI::getExtendedError() << std::endl;
                return false;
            }

            rc = _device.setImageRegistrationMode(registerDepth ? openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR : openni::IMAGE_REGISTRATION_OFF);
            if (rc != openni::STATUS_OK) {
                std::cerr << "Could not set registration mode: " << openni::OpenNI::getExtendedError() << std::endl;
                return false;
            }

            _colorWidth = colorWidth;
            _colorHeight = colorHeight;
            _hasColor = true;

        }
        else {

            openni::VideoMode colorMode = _colorStream.getVideoMode();

            _colorWidth = colorMode.getResolutionX();
            _colorHeight = colorMode.getResolutionY();
            _hasColor = true;

        }

    }

    rc = _depthStream.start();
    if (rc != openni::STATUS_OK || !_depthStream.isValid()) {
        std::cerr << "Could not start depth stream: " << openni::OpenNI::getExtendedError() << std::endl;
        return false;
    }

    if (getColor) {
        rc = _colorStream.start();
        if (rc != openni::STATUS_OK || !_colorStream.isValid()) {
            std::cerr << "Could not start depth stream: " << openni::OpenNI::getExtendedError() << std::endl;
            return false;
        }
    }

    if (!_isLive) {
        openni::PlaybackControl *pc = _device.getPlaybackControl();
        pc->setSpeed(-1);
    }

    _hasTimestamps = true;
    _frame = 0;

    _depthStream.readFrame(&_depthFrame);
    _frameIndexOffset = _depthFrame.getFrameIndex();
    if (_hasColor) {
        _colorStream.readFrame(&_colorFrame);
    }

}

const ushort * OpenNIDepthSource::getDepth() const {
    return (ushort *)_depthFrame.getData();
}

const ushort * OpenNIDepthSource::getDeviceDepth() const {
    return (ushort *)_deviceDepth;
}

const uchar3 * OpenNIDepthSource::getColor() const {
    return (uchar3 *)_colorFrame.getData();
}

uint64_t OpenNIDepthSource::getDepthTime() const {
    return _depthFrame.getTimestamp();
}

uint64_t OpenNIDepthSource::getColorTime() const {
    return _colorFrame.getTimestamp();
}

void OpenNIDepthSource::setFrame(const uint frame) {

    if (_isLive)
        return;

    openni::PlaybackControl * pc = _device.getPlaybackControl();

    pc->seek(_depthStream,frame + _frameIndexOffset);

    advance();

}

void OpenNIDepthSource::advance() {

    _depthStream.readFrame(&_depthFrame);
    _frame = _depthFrame.getFrameIndex() - _frameIndexOffset;
    if (_hasColor) {
        _colorStream.readFrame(&_colorFrame);
    }

    cudaMemcpy(_deviceDepth,_depthFrame.getData(),_depthWidth*_depthHeight*sizeof(ushort),cudaMemcpyHostToDevice);

}

}
