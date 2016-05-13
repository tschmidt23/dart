#ifndef PVN_DEPTH_SOURCE_H
#define PVN_DEPTH_SOURCE_H

#include "depth_source.h"

#include <dirent.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <png.h>
#include <jpeglib.h>
#include <sys/stat.h>
#include <pangolin/video/video_record_repeat.h>
#include <pangolin/image/image_common.h>
#include "util/string_format.h"

#include "util/image_io.h"
#include "util/mirrored_memory.h"
#include "vector_types.h"

namespace dart{

template <typename DepthType, typename ColorType>
class PangoDepthSource : public DepthSource<DepthType, ColorType> {
public:
    PangoDepthSource();
    
    ~PangoDepthSource();
    
    bool initialize(const std::string & pangoFilename,
                    const float2 focalLength,
                    const float2 principalPoint = make_float2(0,0),
                    const uint depthWidth = 0,
                    const uint depthHeight = 0,
                    const float scaleToMeters = 1.f);
    
#ifdef CUDA_BUILD
    const DepthType * getDepth() const { return _depthData->hostPtr(); }
    const DepthType * getDeviceDepth() const { return _depthData->devicePtr(); }
#else
    const DepthType * getDepth() const { return _depthData; }
    const DepthType * getDeviceDepth() const { return 0; }
#endif // CUDA_BUILD
    
    //const ColorType * getColor() const { return _colorData; }
    
    //ColorLayout getColorLayout() const { return LAYOUT_RGB; }
    
    uint64_t getDepthTime() const { return _depthTimes[this->_frame]; }
    
    //uint64_t getColorTime() const { return _colorTimes[this->_frame]; }
    
    void setFrame(const uint frame);
    
    void advance();
    
    bool hasRadialDistortionParams() const { return false; }
    
    float getScaleToMeters() const { return _scaleToMeters; }
    
private:
    
    void readDepth();
    //void readColor();
    
#ifdef CUDA_BUILD
    MirroredVector<DepthType> * _depthData;
#else
    DepthType * _depthData;
#endif // CUDA_BUILD
    uint _firstDepthFrame;
    uint _lastDepthFrame;
    std::string _pangoFilename;
    float _scaleToMeters;
    std::vector<ulong> _depthTimes;
    
    pangolin::VideoRecordRepeat _video;
};

// Implementation
template <typename DepthType, typename ColorType>
PangoDepthSource<DepthType,ColorType>::PangoDepthSource() :
    DepthSource<DepthType,ColorType>(),
    _firstDepthFrame(0),
    _depthData(0) {}

template <typename DepthType, typename ColorType>
PangoDepthSource<DepthType,ColorType>::~PangoDepthSource() {
#ifdef CUDA_BUILD
    delete _depthData;
#else
    delete [] _depthData;
#endif // CUDA_BUILD
}

template <typename DepthType, typename ColorType>
bool PangoDepthSource<DepthType,ColorType>::initialize(
        const std::string & pangoFilename,
        const float2 focalLength,
        const float2 principalPoint,
        const uint depthWidth,
        const uint depthHeight,
        const float scaleToMeters){
    
    this->_frame = 0;
    _pangoFilename = pangoFilename;
    this->_focalLength = focalLength;
    _scaleToMeters = scaleToMeters;
    
    struct stat buffer;
    bool exists = (stat(pangoFilename.c_str(), &buffer) == 0);
    if(!exists){
        std::cerr << "pangolin file not found" << std::endl;
        return false;
    }
    
    std::string uri = std::string("file://") + _pangoFilename;
    _video.Open(uri);
    
    // set depth dimensions
    if(depthWidth > 0 && depthHeight > 0) {
        this->_depthWidth = depthWidth;
        this->_depthHeight = depthHeight;
    }
    else{
        this->_depthWidth = _video.Width();
        this->_depthHeight = _video.Height();
    }
    
    // set principal point
    if(principalPoint.x == 0) {
        this->_principalPoint = make_float2(
                this->_depthWidth/2, this->_depthHeight/2);
    } else {
        this->_principalPoint = principalPoint;
    }
    
    // allocate data
#ifdef CUDA_BUILD
    _depthData = new MirroredVector<DepthType>(
            this->_depthWidth*this->_depthHeight);
#else
    _depthData = new DepthType[this->_depthWidth*this->_depthHeight];
#endif // CUDA_BUILD
    
    return true;
}

template <typename DepthType, typename ColorType>
void PangoDepthSource<DepthType,ColorType>::setFrame(const uint frame) {
    //this->_frame = frame;
    pangolin::VideoPlaybackInterface * playback =
            _video.Cast<pangolin::VideoPlaybackInterface>();
    playback->Seek(frame);
    
    readDepth();
#ifdef CUDA_BUILD
    _depthData->syncHostToDevice();
#endif // CUDA_BUILD
    
    this->_frame = playback->GetCurrentFrameId();
}

template <typename DepthType, typename ColorType>
void PangoDepthSource<DepthType,ColorType>::advance() {
    pangolin::VideoPlaybackInterface * playback =
            _video.Cast<pangolin::VideoPlaybackInterface>();
    if(this->_frame == playback->GetTotalFrames()){
    //if(this->_frame == 90){
        playback->Seek(0);
    }
    
    readDepth();
#ifdef CUDA_BUILD
    _depthData->syncHostToDevice();
#endif // CUDA_BUILD
    
    this->_frame = playback->GetCurrentFrameId();
}

template <typename DepthType, typename ColorType>
void PangoDepthSource<DepthType, ColorType>::readDepth() {
    std::vector<pangolin::Image<unsigned char> > tmp(1);
    _video.Grab((unsigned char*) _depthData->hostPtr(), tmp);
}

}

#endif
