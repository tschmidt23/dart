#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <sys/types.h>
#include "vector_types.h"
#include <png.h>

namespace dart {

void pngErrorHandler(png_structp pngPtr, png_const_charp msg);

void writePNG(const char * filename, const uchar3 * imgData, const int width, const int height);

void writePNG(const char * filename, const ushort * imgData, const int width, const int height);

unsigned char * readPNG(const char * filename, int & width, int & height, int & channels);

}

#endif // IMAGE_IO_H
