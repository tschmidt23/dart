#ifndef IMAGE_DEPTH_SOURCE_H
#define IMAGE_DEPTH_SOURCE_H

#include "depth_source.h"

#include <dirent.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <png.h>
#include <jpeglib.h>
#include "util/string_format.h"

#include "util/image_io.h"
#include "util/mirrored_memory.h"
#include "vector_types.h"

namespace dart {

enum ImageType {
    IMAGE_PNG,
    IMAGE_JPG
};

template <typename DepthType, typename ColorType>
class ImageDepthSource : public DepthSource<DepthType,ColorType> {
public:
    ImageDepthSource();

    ~ImageDepthSource();

    /**
    * An initializer which uses printf formatting to locate depth and color frames.
    * @return true if the initialization succeeds.
    */
    bool initialize(const std::string depthFilenameFormat,
                    const float2 focalLength,
                    const float2 principalPoint = make_float2(0,0),
                    const uint depthWidth = 0,
                    const uint depthHeight = 0,
                    const uint firstDepthFrame = 0,
                    const bool hasColor = false,
                    const std::string colorFilenameFormat = "",
                    const uint colorWidth = 0,
                    const uint colorHeight = 0,
                    const uint firstColorFrame = 0 );

    /**
    * An initializer which locates depth and color frames by iterating over images in a directory.
    * @return true if the initialization succeeds.
    */
    bool initialize(const std::string & depthDirectory,
                    const ImageType depthImageType,
                    const float2 focalLength,
                    const float2 principalPoint = make_float2(0,0),
                    const uint depthWidth = 0,
                    const uint depthHeight = 0,
                    const float scaleToMeters = 1.0f,
                    const std::vector<ulong> * depthTimes = 0,
                    const bool hasColor = false,
                    const std::string colorDirectory = "",
                    const ImageType colorImageType = IMAGE_JPG,
                    const uint colorWidth = 0,
                    const uint colorHeight = 0,
                    const std::vector<long> * colorTimes = 0);

#ifdef CUDA_BUILD
    const DepthType * getDepth() const { return _depthData->hostPtr(); }
    const DepthType * getDeviceDepth() const { return _depthData->devicePtr(); }
#else
    const DepthType * getDepth() const { return _depthData; }
    const DepthType * getDeviceDepth() const { return 0; }
#endif // CUDA_BUILD

    const ColorType * getColor() const { return _colorData; }

    ColorLayout getColorLayout() const { return LAYOUT_RGB; }

    uint64_t getDepthTime() const { return _depthTimes[this->_frame]; }

    uint64_t getColorTime() const { return _colorTimes[this->_frame]; }

    void setFrame(const uint frame);

    void advance();

    bool hasRadialDistortionParams() const { return false; }


    float getScaleToMeters() const { return _scaleToMeters; }

    void setPngSwap(bool pngSwap) { _pngSwap = pngSwap; }

    int getNumDepthFrames() const { return _depthFilenames.size(); }

private:

    void readDepth();
    void readColor();

    bool readPNG(const char * filename,
                 bool isDepth=true);
    bool getDimPNG(const char * filename,
                   uint & width,
                   uint & height);
    bool readJPG(const char * filename,
                 bool isDepth=false);
    bool getDimJPG(const char * filename,
                   uint & width,
                   uint & height);

#ifdef CUDA_BUILD
    MirroredVector<DepthType> * _depthData;
#else
    DepthType * _depthData;
#endif // CUDA_BUILD
    ColorType * _colorData;
    ImageType _depthImageType;
    ImageType _colorImageType;
    std::string _depthFormat;
    std::string _colorFormat;
    uint _firstDepthFrame;
    uint _lastDepthFrame;
    uint _firstColorFrame;
    uint _lastColorFrame;
    std::string _depthDirectory;
    std::string _colorDirectory;
    std::vector<std::string> _depthFilenames;
    std::vector<std::string> _colorFilenames;
    std::vector<ulong> _depthTimes;
    std::vector<ulong> _colorTimes;
    std::vector<uint> _correspondingColorFrames;
    float _scaleToMeters;
    bool _pngSwap;
};

// Implementation
template <typename DepthType, typename ColorType>
ImageDepthSource<DepthType,ColorType>::ImageDepthSource() :
    DepthSource<DepthType,ColorType>(), _firstDepthFrame(0), _firstColorFrame(0), _depthData(0), _pngSwap(true) {}

template <typename DepthType, typename ColorType>
ImageDepthSource<DepthType,ColorType>::~ImageDepthSource() {
#ifdef CUDA_BUILD
    delete _depthData;
#else
    delete [] _depthData;
#endif // CUDA_BUILD
}

template <typename DepthType, typename ColorType>
bool ImageDepthSource<DepthType,ColorType>::initialize( const std::string & depthDirectory,
                                                        const ImageType depthImageType,
                                                        const float2 focalLength,
                                                        const float2 principalPoint,
                                                        const uint depthWidth,
                                                        const uint depthHeight,
                                                        const float scaleToMeters,
                                                        const std::vector<ulong> * depthTimes,
                                                        const bool hasColor,
                                                        const std::string colorDirectory,
                                                        const ImageType colorImageType,
                                                        const uint colorWidth,
                                                        const uint colorHeight,
                                                        const std::vector<long> * colorTimes) {

    this->_frame = 0;
    _depthImageType = depthImageType;
    _depthDirectory = depthDirectory;
    this->_focalLength = focalLength;
    _scaleToMeters = scaleToMeters;

    std::string expectedExtension;
    switch (depthImageType) {
    case IMAGE_PNG:
        expectedExtension = "png";
        break;
    case IMAGE_JPG:
        expectedExtension = "jpg";
        break;
    }

    DIR * dir;
    struct dirent * ent;
    if ((dir = opendir(_depthDirectory.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = std::string(ent->d_name);
            std::string extension = filename.substr(filename.find_last_of(".")+1);
            if (extension == expectedExtension) {
                _depthFilenames.push_back(filename);
            }
        }
        closedir(dir);
    }
    else {
        std::cerr << "could not open directory " << _depthDirectory << std::endl;
        return false;
    }

    if (_depthFilenames.size() == 0) {
        std::cerr << "no files with extension " << expectedExtension << " in directory " << _depthDirectory << std::endl;
        return false;
    }

    std::cout << "found " << _depthFilenames.size() << " depth frames" << std::endl;

    std::sort(_depthFilenames.begin(),_depthFilenames.end());

    // set depth dimensions
    if (depthWidth > 0 && depthHeight > 0) {
        this->_depthWidth = depthWidth;
        this->_depthHeight = depthHeight;
    }
    else {
        std::string firstFile = _depthDirectory + "/" + _depthFilenames[0];
        switch(_depthImageType) {
        case IMAGE_PNG:
            getDimPNG(firstFile.c_str(),this->_depthWidth,this->_depthHeight);
            break;
        case IMAGE_JPG:
            getDimJPG(firstFile.c_str(),this->_depthWidth,this->_depthHeight);
            break;
        }
    }

    if (principalPoint.x == 0) {
        this->_principalPoint = make_float2(this->_depthWidth/2,this->_depthHeight/2);
    } else {
        this->_principalPoint = principalPoint;
    }

    // allocate data
#ifdef CUDA_BUILD
    _depthData = new MirroredVector<DepthType>(this->_depthWidth*this->_depthHeight);
#else
    _depthData = new DepthType[this->_depthWidth*this->_depthHeight];
#endif // CUDA_BUILD

    // save timestamps
    if (depthTimes) {
        this->_hasTimestamps = true;
        _depthTimes = *depthTimes;
    }
    else {
        this->_hasTimestamps = false;
    }

    // read first frame
    readDepth();
#ifdef CUDA_BUILD
    _depthData->syncHostToDevice();
#endif // CUDA_BUILD

    if (hasColor) {
        this->_hasColor = hasColor;
        _colorImageType = colorImageType;
        _colorDirectory = colorDirectory;

        std::string expectedExtension;
        switch (colorImageType) {
        case IMAGE_PNG:
            expectedExtension = "png";
            break;
        case IMAGE_JPG:
            expectedExtension = "jpg";
            break;
        }

        DIR * dir;
        struct direntent;
        if ((dir = opendir(_colorDirectory.c_str())) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                std::string filename = std::string(ent->d_name);
                std::string extension = filename.substr(filename.find_last_of(".")+1);
                if (extension == expectedExtension) {
                    _colorFilenames.push_back(filename);
                }
            }
            closedir(dir);
        }
        else {
            std::cerr << "could not open directory " << _colorDirectory << std::endl;
            return false;
        }

        if (_depthFilenames.size() == 0) {
            std::cerr << "no files with extension " << expectedExtension << " in directory " << _colorDirectory << std::endl;
            return false;
        }

        std::cout << "found " << _colorFilenames.size() << " color frames" << std::endl;

        std::sort(_colorFilenames.begin(),_colorFilenames.end());

        // set color dimensions
        if (colorWidth > 0 && colorHeight > 0) {
            this->_colorWidth = colorWidth;
            this->_colorHeight = colorHeight;
        }
        else {
            std::string firstFile = _colorDirectory + "/" + _colorFilenames[0];
            switch(_colorImageType) {
            case IMAGE_PNG:
                getDimPNG(firstFile.c_str(),this->_colorWidth,this->_colorHeight);
                break;
            case IMAGE_JPG:
                getDimJPG(firstFile.c_str(),this->_colorWidth,this->_colorHeight);
                break;
            }
        }

        // allocate data
        _colorData = new ColorType[this->_colorWidth*this->_colorHeight];

        // save timestamps
        if (colorTimes && depthTimes) {
//            _colorTimes = *colorTimes;
            std::vector<long>::const_iterator colorIt = colorTimes->begin();
            for (int i=0; i<_depthTimes.size(); ++i) {
                long targetTime = _depthTimes[i];
                colorIt = std::lower_bound(colorIt,colorTimes->end(),targetTime);
                if (colorIt == colorTimes->end()) {
                    --colorIt;
                    _correspondingColorFrames.push_back(colorIt - colorTimes->begin());
                }
                else {
                    _correspondingColorFrames.push_back(colorIt - colorTimes->begin());
                }
            }
        }
        else {
            for (int i=0; i<_depthFilenames.size(); ++i) {
                _correspondingColorFrames.push_back(i);
            }
        }

    }

    return true;

}

template <typename DepthType, typename ColorType>
bool ImageDepthSource<DepthType,ColorType>::initialize(const std::string depthFilenameFormat,
                                                       const float2 focalLength,
                                                       const float2 principalPoint,
                                                       const uint depthWidth,
                                                       const uint depthHeight,
                                                       const uint firstDepthFrame,
                                                       const bool hasColor,
                                                       const std::string colorFilenameFormat,
                                                       const uint colorWidth,
                                                       const uint colorHeight,
                                                       const uint firstColorFrame) {

    // set depth file type
    std::string depthExtension = depthFilenameFormat.substr(depthFilenameFormat.find_last_of(".")+1);
    if (depthExtension == "png") {
        _depthImageType = IMAGE_PNG;
    }
    else {
        std::cerr << "image extension " << depthExtension << " is not supported" << std::endl;
        return false;
    }

    this->_focalLength = focalLength;

    // set first and last depth frame
    _depthFormat = depthFilenameFormat;
    _firstDepthFrame = firstDepthFrame;
    _lastDepthFrame = firstDepthFrame;
    bool exists;
    do {
        std::string filename = stringFormat(depthFilenameFormat,_lastDepthFrame);
        struct stat buffer;
        exists = (stat(filename.c_str(),&buffer) == 0);
        _lastDepthFrame += exists ? 1 : 0;
    } while (exists);

    if (_lastDepthFrame == _firstDepthFrame) {
        std::cerr << "no images with depth filename format found" << std::endl;
        return false;
    }

    std::string firstFile = stringFormat(depthFilenameFormat,firstDepthFrame);

    // set depth dimensions
    if (depthWidth > 0 && depthHeight > 0) {
        this->_depthWidth = depthWidth;
        this->_depthHeight = depthHeight;
    }
    else {
        switch(_depthImageType) {
        case IMAGE_PNG:
            getDimPNG(firstFile.c_str(),this->_depthWidth,this->_depthHeight);
            break;
        }
    }

    // set principal point
    if (principalPoint.x == 0) {
        this->_principalPoint = make_float2(this->_depthWidth/2,this->_depthHeight/2);
    } else {
        this->_principalPoint = principalPoint;
    }

    // allocate data
#ifdef CUDA_BUILD
    _depthData = new MirroredVector<DepthType>(this->_depthWidth*this->_depthHeight);
#else
    _depthData = new DepthType[this->_depthWidth*this->_depthHeight];
#endif // CUDA_BUILD

    // read first image
    switch(_depthImageType) {
    case IMAGE_PNG:
        readPNG(firstFile.c_str());
        break;
    }

    if (hasColor) {

        // set color file type
        std::string colorExtension = colorFilenameFormat.substr(colorFilenameFormat.find_last_of(".")+1);
        if (colorExtension == "png") {
            _colorImageType = IMAGE_PNG;
        }
        else {
            std::cerr << "image extension " << colorExtension << " is not supported" << std::endl;
            return false;
        }

        // set first and last color frame
        _colorFormat = colorFilenameFormat;
        _firstColorFrame = firstColorFrame;
        _lastColorFrame = firstColorFrame;
        bool exists;
        do {
            std::string filename = stringFormat(colorFilenameFormat,_lastColorFrame);
            struct stat buffer;
            std::cout << "checking " << filename << std::endl;
            exists = (stat(filename.c_str(),&buffer) == 0);
            _lastColorFrame += exists ? 1 : 0;
        } while (exists);

        if (_lastColorFrame == _firstColorFrame) {
            std::cerr << "no images with color filename format found" << std::endl;
            return false;
        }

        std::string firstFile = stringFormat(colorFilenameFormat,firstColorFrame);

        // set color dimensions
        if (colorWidth > 0 && colorHeight > 0) {
            this->_colorWidth = colorWidth;
            this->_colorHeight = colorHeight;
        }
        else {
            switch(_colorImageType) {
            case IMAGE_JPG:
                getDimJPG(firstFile.c_str(),this->_colorWidth,this->_colorHeight);
                break;
            }
        }

        // allocate data
        _colorData = new ColorType[this->_colorWidth*this->_colorHeight];

        this->_hasColor = true;

        // read first farme
        switch(_colorImageType) {
        case IMAGE_JPG:
            readJPG(firstFile.c_str(),false);
            break;
        case IMAGE_PNG:
            readPNG(firstFile.c_str(),false);
            break;
        }

    }

    return true;
}

template <typename DepthType, typename ColorType>
void ImageDepthSource<DepthType,ColorType>::setFrame(const uint frame) {

    this->_frame = frame;

    readDepth();
    if (this->_hasColor) {
        readColor();
    }

}

template <typename DepthType, typename ColorType>
void ImageDepthSource<DepthType,ColorType>::advance() {

    // update frame
    this->_frame++;
    if (this->_frame == (_depthFilenames.size())) {
        this->_frame = 0;
    }

    readDepth();
#ifdef CUDA_BUILD
    _depthData->syncHostToDevice();
#endif // CUDA_BUILD

    if (this->_hasColor) {
        readColor();
    }
}

template <typename DepthType, typename ColorType>
void ImageDepthSource<DepthType, ColorType>::readDepth() {

    // read depth
    const std::string depthFilename = _depthDirectory + "/" + _depthFilenames[this->_frame];
    switch (_depthImageType) {
    case IMAGE_PNG:
        readPNG(depthFilename.c_str());
        break;
    }

//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::normal_distribution<> nd(0.0,1.0);

//    int factor = 4;
//    float noiseMap[this->_depthWidth*this->_depthHeight/(factor*factor)];
//    for (int i=0; i<this->_depthWidth*this->_depthHeight/(factor*factor); ++i) {
//        noiseMap[i] = 2*nd(gen);
//    }

//    for (int y=0; y<this->_depthHeight; ++y) {
//        for (int x=0; x<this->_depthWidth; ++x) {
//            float xx = (x+0.5)/factor;
//            float yy = (y+0.5)/factor;
//            int px = floor(xx);
//            int py = floor(yy);
//            float tx = xx - px;
//            float ty = yy - py;
//            float noise = (1-tx)*(1-ty)*noiseMap[px + py*this->_depthWidth/factor] +
//                    (1-tx)*ty*noiseMap[px + (py+1)*this->_depthWidth/factor] +
//                    tx*(1-ty)*noiseMap[(px+1) + py*this->_depthWidth/factor] +
//                    tx*ty*noiseMap[(px+1) + (py+1)*this->_depthWidth/factor];
//            _depthData->hostPtr()[x + y*this->_depthWidth] += noise;
//        }
//    }

}

template <typename DepthType, typename ColorType>
void ImageDepthSource<DepthType, ColorType>::readColor() {

//    std::cout << "looking for corresponding color frame to " << this->_frame <<": " <<  _correspondingColorFrames[this->_frame];

    // read color
    int correspondingColorFrame = _correspondingColorFrames[this->_frame];
    const std::string colorFilename = _colorDirectory + "/" + _colorFilenames[correspondingColorFrame];
    switch (_colorImageType) {
    case IMAGE_PNG:
        readPNG(colorFilename.c_str(),false);
        break;
    case IMAGE_JPG:
        readJPG(colorFilename.c_str(),false);
    }

}

template <typename DepthType, typename ColorType>
bool ImageDepthSource<DepthType, ColorType>::readPNG(const char * filename,
                                                     const bool isDepth) {

    FILE * file = fopen(filename,"r");

    unsigned char sig[8];
    int nRead = fread(sig, 1, 8, file);
    if (nRead != 8 || !png_check_sig(sig,8)) {
        std::cerr << filename << " is not a valid png file" << std::endl;
        fclose(file);
        return false;
    }

    jmp_buf buff;
    png_structp pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, &buff, pngErrorHandler, NULL);
    if (!pngPtr) {
        std::cerr << "could not create png pointer" << std::endl;
        fclose(file);
        return false;
    }

    png_infop infoPtr = png_create_info_struct(pngPtr);
    if (!infoPtr) {
        png_destroy_read_struct(&pngPtr,NULL,NULL);
        std::cerr << "could not create info pointer" << std::endl;
        fclose(file);
        return false;
    }

    if (setjmp(buff)) {
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(file);
        return false;
    }

    png_init_io(pngPtr, file);
    png_set_sig_bytes(pngPtr, 8);
    png_read_info(pngPtr, infoPtr);
    if (_pngSwap) {
        png_set_swap(pngPtr);
    }

    png_uint_32 width, height;
    int bitDepth, colorType;
    png_get_IHDR(pngPtr, infoPtr, &width, &height, &bitDepth, &colorType, NULL, NULL, NULL);

    int channels = (int)png_get_channels(pngPtr, infoPtr);

//    std::cout << width << "x" << height << ", bit depth: " << bitDepth << ", color type: " << colorType << std::endl;

    // check type and size
    if (isDepth) {
        if (width != this->_depthWidth || height != this->_depthHeight) {
            std::cerr << "expected " << this->_depthWidth << "x" << this->_depthHeight << " depth but got " << width << "x" << height << std::endl;
            fclose(file);
            return false;
        }
        if (bitDepth*channels != 8*sizeof(DepthType)) {
            std::cerr << "expected " << 8*sizeof(DepthType) << " bits but got " << bitDepth*channels << std::endl;
            fclose(file);
            return false;
        }
    }
    else {
        if (width != this->_colorWidth || height != this->_colorHeight) {
            std::cerr << "expected " << this->_colorWidth << "x" << this->_colorHeight << " color but got " << width << "x" << height << std::endl;
            fclose(file);
            return false;
        }
        if (bitDepth*channels != 8*sizeof(ColorType)) {
            std::cerr << "expected " << 8*sizeof(ColorType) << " bits but got " << bitDepth*channels << std::endl;
            fclose(file);
            return false;
        }
    }

    png_uint_32 i;
    png_bytep rowPointers[height];

    png_read_update_info(pngPtr,infoPtr);
    if (isDepth) {
        for (i=0; i<height; ++i) {
#ifdef CUDA_BUILD
            rowPointers[i] = ((png_bytep)_depthData->hostPtr()) + i*png_get_rowbytes(pngPtr,infoPtr);
#else
            rowPointers[i] = ((png_bytep)_depthData) + i*png_get_rowbytes(pngPtr,infoPtr);
#endif // CUDA_BUILD
        }
    }
    else {
        for (i=0; i<height; ++i) {
            rowPointers[i] = ((png_bytep)_colorData) + i*png_get_rowbytes(pngPtr,infoPtr);
        }
    }

    png_read_image(pngPtr,rowPointers);

    png_read_end(pngPtr, NULL);

    png_destroy_read_struct(&pngPtr, &infoPtr, NULL);

    fclose(file);
    return true;

}

template <typename DepthType, typename ColorType>
bool ImageDepthSource<DepthType, ColorType>::getDimPNG(const char * filename,
                                                       uint & width,
                                                       uint & height) {

    FILE * file = fopen(filename,"r");

    unsigned char sig[8];
    int nRead = fread(sig, 1, 8, file);
    if (nRead != 8 || !png_check_sig(sig,8)) {
        std::cerr << filename << " is not a valid png file" << std::endl;
        fclose(file);
        return false;
    }

    jmp_buf buff;
    png_structp pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, &buff, pngErrorHandler, NULL);
    if (!pngPtr) {
        std::cerr << "could not create png pointer" << std::endl;
        fclose(file);
        return false;
    }

    png_infop infoPtr = png_create_info_struct(pngPtr);
    if (!infoPtr) {
        png_destroy_read_struct(&pngPtr,NULL,NULL);
        std::cerr << "could not create info pointer" << std::endl;
        fclose(file);
        return false;
    }

    if (setjmp(buff)) {
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(file);
        return false;
    }

    png_init_io(pngPtr, file);
    png_set_sig_bytes(pngPtr, 8);
    png_read_info(pngPtr, infoPtr);

    png_uint_32 widthLong, heightLong;
    int bitDepth, colorType;
    png_get_IHDR(pngPtr, infoPtr, &widthLong, &heightLong, &bitDepth, &colorType, NULL, NULL, NULL);

    width = (uint)widthLong;
    height = (uint)heightLong;

    png_destroy_read_struct(&pngPtr, &infoPtr, NULL);

    fclose(file);

    return true;

}

template <typename DepthType, typename ColorType>
bool ImageDepthSource<DepthType, ColorType>::readJPG(const char * filename,
                                                     bool isDepth) {

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

//    std::cout << "reading " << filename << std::endl;

    FILE * file = fopen(filename,"r");

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);

    jpeg_read_header(&cinfo,true);
    jpeg_start_decompress(&cinfo);

    int width = cinfo.output_width;
    int height = cinfo.output_height;

    int outputComponents = cinfo.output_components;
    unsigned char* rowPointers[height];

    if (isDepth) {
        if (width != this->_depthWidth || height != this->_depthHeight) {
            std::cerr << "expected " << this->_depthWidth << "x" << this->_depthHeight << " depth but got " << width << "x" << height << std::endl;
            fclose(file);
            jpeg_finish_decompress(&cinfo);
            jpeg_destroy_decompress(&cinfo);
            return false;
        }
        if (outputComponents != sizeof(DepthType)) {
            std::cerr << "expected " << sizeof(DepthType) << " bytes but got " << outputComponents << std::endl;
            fclose(file);
            jpeg_finish_decompress(&cinfo);
            jpeg_destroy_decompress(&cinfo);
            return false;
        }

        for (int v=0; v<height; ++v) {
#ifdef CUDA_BUILD
            rowPointers[v] = (unsigned char*)&this->_depthData->hostPtr()[v*width];
#else
            rowPointers[v] = (unsigned char*)&this->_depthData[v*width];
#endif // CUDA_BUILD
        }

    }
    else {
        if (width != this->_colorWidth || height != this->_colorHeight) {
            std::cerr << "expected " << this->_colorWidth << "x" << this->_colorHeight << " color but got " << width << "x" << height << std::endl;
            fclose(file);
            jpeg_finish_decompress(&cinfo);
            jpeg_destroy_decompress(&cinfo);
            return false;
        }
        if (outputComponents != sizeof(ColorType)) {
            std::cerr << "expected " << sizeof(ColorType) << " bytes but got " << outputComponents << std::endl;
            fclose(file);
            jpeg_finish_decompress(&cinfo);
            jpeg_destroy_decompress(&cinfo);
            return false;

        }

        for (int v=0; v<height; ++v) {
            rowPointers[v] = (unsigned char*)&this->_colorData[v*width];
        }

    }

    while (cinfo.output_scanline < height) {
        jpeg_read_scanlines(&cinfo,rowPointers + cinfo.output_scanline,height-cinfo.output_scanline);
    }

    fclose(file);
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    return true;

}

template <typename DepthType, typename ColorType>
bool ImageDepthSource<DepthType, ColorType>::getDimJPG(const char * filename,
                                                       uint & width,
                                                       uint & height) {

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE * file = fopen(filename,"r");

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);

    jpeg_read_header(&cinfo,true);
    jpeg_start_decompress(&cinfo);

    width = cinfo.output_width;
    height = cinfo.output_height;

    uchar3 * buffer = new uchar3[width];

    while (cinfo.output_scanline < height) {
        jpeg_read_scanlines(&cinfo,(JSAMPARRAY)&buffer, 1);
    }

    fclose(file);
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    return true;

}

}

#endif // IMAGE_DEPTH_SOURCE_H
