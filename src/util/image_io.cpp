#include "image_io.h"

#include <iostream>

namespace dart {

void pngErrorHandler(png_structp pngPtr, png_const_charp msg) {
    jmp_buf * buff;

    std::cerr << "libpng error: " << msg << std::endl;

    buff = (jmp_buf *)png_get_error_ptr(pngPtr);
    if (buff == NULL) {
        std::cerr << "png severe error:  jmpbuf not recoverable; terminating" << std::endl;
        exit(99);
    }

    longjmp(*buff, 1);
}

void writePNG(const char * filename, const uchar3 * imgData, const int width, const int height) {

    FILE * fp = fopen(filename,"wb");

    png_byte * rowPointers[height];

    jmp_buf buff;
    png_structp pngPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, &buff, pngErrorHandler, NULL);
    png_infop infoPtr = png_create_info_struct(pngPtr);

    if (setjmp(buff)) {
        goto png_failure;
    }

    png_set_IHDR( pngPtr,
                  infoPtr,
                  width,
                  height,
                  sizeof(unsigned char)*8,
                  PNG_COLOR_TYPE_RGB,
                  PNG_INTERLACE_NONE,
                  PNG_COMPRESSION_TYPE_DEFAULT,
                  PNG_FILTER_TYPE_DEFAULT);


    for (int y = 0; y<height; ++y) {
        rowPointers[y] = (png_byte*)&imgData[y*width];
    }

    png_init_io(pngPtr, fp);
    png_set_rows(pngPtr,infoPtr,rowPointers);
    png_write_png(pngPtr,infoPtr,PNG_TRANSFORM_SWAP_ENDIAN,NULL);

png_failure:
    png_destroy_write_struct(&pngPtr,&infoPtr);
    fclose(fp);
}

void writePNG(const char * filename, const ushort * imgData, const int width, const int height) {

    FILE * fp = fopen(filename,"wb");

    png_byte * rowPointers[height];

    jmp_buf buff;
    png_structp pngPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, &buff, pngErrorHandler, NULL);
    png_infop infoPtr = png_create_info_struct(pngPtr);

    if (setjmp(buff)) {
        goto png_failure;
    }

    png_set_swap( pngPtr );
    png_set_IHDR( pngPtr,
                  infoPtr,
                  width,
                  height,
                  sizeof(ushort)*8,
                  PNG_COLOR_TYPE_GRAY,
                  PNG_INTERLACE_NONE,
                  PNG_COMPRESSION_TYPE_DEFAULT,
                  PNG_FILTER_TYPE_DEFAULT);

    for (int y = 0; y<height; ++y) {
        rowPointers[y] = (png_byte*)&imgData[y*width];
    }

    png_init_io(pngPtr, fp);
    png_set_rows(pngPtr,infoPtr,rowPointers);
    png_write_png(pngPtr,infoPtr,PNG_TRANSFORM_SWAP_ENDIAN,NULL);

png_failure:
    png_destroy_write_struct(&pngPtr,&infoPtr);
    fclose(fp);
}

unsigned char * readPNG(const char * filename, int & width, int & height, int & channels) {
    
    FILE * file = fopen(filename, "r");
    
    unsigned char sig[8];
    size_t nRead = fread(sig, 1, 8, file);

    if (!png_check_sig(sig,8) || nRead != 8) {
        std::cerr << filename << " is not a valid png file" << std::endl;
        fclose(file);
        return 0;
    }

    jmp_buf buff;
    png_structp pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, &buff, pngErrorHandler, NULL);
    if (!pngPtr) {
        std::cerr << "could not create png pointer" << std::endl;
        fclose(file);
        return 0;
    }

    png_infop infoPtr = png_create_info_struct(pngPtr);
    if (!infoPtr) {
        png_destroy_read_struct(&pngPtr,NULL,NULL);
        std::cerr << "could not create info pointer" << std::endl;
        fclose(file);
        return 0;
    }

    if (setjmp(buff)) {
        png_destroy_read_struct(&pngPtr, &infoPtr, NULL);
        fclose(file);
        return 0;
    }

    png_init_io(pngPtr, file);
    png_set_sig_bytes(pngPtr, 8);
    png_read_info(pngPtr, infoPtr);
    png_set_swap(pngPtr);

    int colorType, bitDepth;
    png_uint_32 pngWidth, pngHeight;
    png_get_IHDR(pngPtr, infoPtr, &pngWidth, &pngHeight, &bitDepth, &colorType, NULL, NULL, NULL);

    channels = (int)png_get_channels(pngPtr, infoPtr);
    width = pngWidth;
    height = pngHeight;

    png_uint_32 i;
    png_bytep rowPointers[height];

    unsigned char * data = new unsigned char[width*height*channels*bitDepth/8];

    png_read_update_info(pngPtr,infoPtr);
    for (i=0; i<height; ++i) {
        rowPointers[i] = ((png_bytep)data) + i*png_get_rowbytes(pngPtr,infoPtr);
    }

    png_read_image(pngPtr,rowPointers);

    png_read_end(pngPtr, NULL);

    png_destroy_read_struct(&pngPtr, &infoPtr, NULL);

    fclose(file);
    return data;

}

}
