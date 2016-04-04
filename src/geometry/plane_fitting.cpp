#include "plane_fitting.h"

#include <Eigen/Dense>
#include "util/mirrored_memory.h"

namespace dart {

void fitPlane(float3 & planeNormal,
              float & planeIntercept,
              const float4 * dObsVertMap,
              const float4 * dObsNormMap,
              const int width,
              const int height,
              const float distanceThreshold,
              const float normalThreshold,
              const int maxIters,
              const float regularization,
              int * dbgAssociated) {

    dart::MirroredVector<float> result(4 + 16 + 1);


    for (int iter=0; iter<maxIters; ++iter) {

        cudaMemset(result.devicePtr(),0,(1+4+16)*sizeof(float));

        fitPlaneIter(planeNormal,
                     planeIntercept,
                     dObsVertMap,
                     dObsNormMap,
                     width,
                     height,
                     distanceThreshold,
                     normalThreshold,
                     regularization,
                     result.devicePtr(),
                     dbgAssociated);

        result.syncDeviceToHost();

        Eigen::MatrixXf JTJ = regularization*Eigen::MatrixXf::Identity(4,4);
        Eigen::VectorXf eJ = Eigen::VectorXf::Zero(4,1);

        for (int i=0; i<4; ++i) {
            eJ(i) = result.hostPtr()[i];
            for(int j=0; j<4; ++j) {
                JTJ(i,j) += result.hostPtr()[4 + j + i*4];
            }
        }

        //std::cout << "JTJ: \n" << JTJ << std::endl;
        //std::cout << "eJ: \n" << eJ << std::endl;

        Eigen::VectorXf update = -JTJ.ldlt().solve(eJ);

        //std::cout << "error: " << result.hostPtr()[4+16] << std::endl;
        //std::cout << "update: " << update << std::endl;

        planeNormal.x += update(0);
        planeNormal.y += update(1);
        planeNormal.z += update(2);
        planeIntercept += update(3);

        planeNormal = normalize(planeNormal);

        //std::cout << "new normal " << planeNormal.x << ", " << planeNormal.y << ", " << planeNormal.z << std::endl;
        //std::cout << "new intercept " << planeIntercept << std::endl;
    }

}

}
