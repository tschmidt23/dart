#ifndef IMG_MATH_H
#define IMG_MATH_H

namespace dart {

void imageSquare(float * out, const float * in, const int width, const int height);

void imageSqrt(float * out, const float * in, const int width, const int height);

template <typename T>
void imageFlipX(T * out, const T * in, const int width, const int height);

template <typename T>
void imageFlipY(T * out, const T * in, const int width, const int height);

template <typename T>
void unitNormalize(const T * in, T * out, const int width, const int height, const T zeroVal, const T oneVal);


}

#endif // IMG_MATH_H
