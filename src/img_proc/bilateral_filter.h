#ifndef BILATERAL_FILTER_H
#define BILATERAL_FILTER_H

namespace dart {

template <typename T>
void bilateralFilter(const T * depthIn, float * depthOut, const int width, const int height, const float sigmaDomain, const float sigmaRange);

}

#endif // BILATERAL_FILTER_H
