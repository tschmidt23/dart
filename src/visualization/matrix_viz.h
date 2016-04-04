#ifndef MATRIX_VIZ_H
#define MATRIX_VIZ_H

#include <vector_types.h>

namespace dart {

    void visualizeMatrix(const float * mxData,
                         const int mxCols,
                         const int mxRows,
                         uchar3 * img,
                         const int width,
                         const int height,
                         const uchar3 zeroColor,
                         const float minVal,
                         const float maxVal);

}

#endif // MATRIX_VIZ_H
