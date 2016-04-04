#include "matrix_viz.h"

#include <stdio.h>

namespace dart {

// -=-=-=-=-=-=-=-=-=- kernels -=-=-=-=-=-=-=-=-=-
__global__ void gpu_visualizeMatrix(
     const float * mxData,
     const int mxCols,
     const int mxRows,
     uchar3 * img,
     const int width,
     const int height,
     const uchar3 zeroColor,
     const float minVal,
     const float maxVal) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int mxCol = (float)mxCols*x/width;
    int mxRow = (float)mxRows*y/height;

    mxCol = min(max(0,mxCol),mxCols-1);
    mxRow = min(max(0,mxRow),mxRows-1);

//    printf("%d,%d\n",mxRow,mxCol);
    const float val = mxData[mxRow*mxCols + mxCol];
    //const int val = 0.0f;

    if (val == 0.0f) {
        img[x + y*width] = zeroColor;
        return;
    }

    float a = min(max(0.0f,(val - minVal)/(val - maxVal)),1.0f);
    img[x + y*width] = make_uchar3(255*a,255*a,255*a);

}

// -=-=-=-=-=-=-=-=-=- interface -=-=-=-=-=-=-=-=-=-
void visualizeMatrix(const float * mxData,
     const int mxCols,
     const int mxRows,
     uchar3 * img,
     const int width,
     const int height,
     const uchar3 zeroColor,
     const float minVal,
     const float maxVal) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_visualizeMatrix<<<grid,block>>>(mxData,mxCols,mxRows,img,width,height,zeroColor,minVal,maxVal);

}

}
