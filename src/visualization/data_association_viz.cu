#include "data_association_viz.h"

#include <stdio.h>
\
namespace dart {

// -=-=-=-=-=-=-=-=-=- kernels -=-=-=-=-=-=-=-=-=-
__global__ void gpu_colorDataAssociation(uchar3 * coloredAssociation,
                                         const int * dataAssociation,
                                         const uchar3 * colors,
                                         const int width,
                                         const int height,
                                         const uchar3 unassociatedColor) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    const int association = dataAssociation[index];
    if (association < 0) {
        coloredAssociation[index] = unassociatedColor;
    }
    else {
        coloredAssociation[index] = colors[association];
    }
//    coloredAssociation[index] = make_uchar3(0,1,0);

}

__global__ void gpu_colorDataAssociationMultiModel(uchar3 * coloredAssociation,
                                                   const int * dataAssociation,
                                                   const uchar3 * * colors,
                                                   const int width,
                                                   const int height,
                                                   const uchar3 unassociatedColor) {

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y*width;
    if (dataAssociation[index] < 0) {
        coloredAssociation[index] = unassociatedColor;
    }
    else {
        const int association = 0xffff & dataAssociation[index];
        const int model = dataAssociation[index] >> 16;
        coloredAssociation[index] = colors[model][association];
    }

}

// -=-=-=-=-=-=-=-=-=- interface -=-=-=-=-=-=-=-=-=-
void colorDataAssociation(uchar3 * coloredAssociation,
                          const int * dataAssociation,
                          const uchar3 * colors,
                          const int width,
                          const int height,
                          const uchar3 unassociatedColor) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_colorDataAssociation<<<grid,block>>>(coloredAssociation,dataAssociation,colors,width,height,unassociatedColor);
}

void colorDataAssociationMultiModel(uchar3 * coloredAssociation,
                                    const int * dataAssociation,
                                    const uchar3 * * colors,
                                    const int width,
                                    const int height,
                                    const uchar3 unassociatedColor) {

    dim3 block(16,8,1);
    dim3 grid( ceil( width / (float)block.x), ceil(height / (float)block.y ));

    gpu_colorDataAssociationMultiModel<<<grid,block>>>(coloredAssociation,dataAssociation,colors,width,height,unassociatedColor);

}

}
