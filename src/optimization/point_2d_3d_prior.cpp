#include "priors.h"
#include "geometry/sdf.h"

namespace dart {

void Point2D3DPrior::computeContribution(Eigen::SparseMatrix<float> & JTJ,
                                         Eigen::VectorXf & JTe,
                                         const int * modelOffsets,
                                         const int priorParamOffset,
                                         const std::vector<MirroredModel *> & models,
                                         const std::vector<Pose> & poses,
                                         const OptimizationOptions & opts) {

    if (_weight == 0) { return; }

    const MirroredModel & srcModel = *models[_srcModelID];

    const Pose & srcPose = poses[_srcModelID];

    const int srcOffset = modelOffsets[_srcModelID];

    const int srcDims = srcPose.getReducedDimensions();

    const float3 srcPoint_c = SE3Transform(srcModel.getTransformFrameToCamera(_srcFrame),_point_f);

    const float2 projectedSrcPoint = make_float2( srcPoint_c.x*_focalLength.x/srcPoint_c.z + _principalPoint.x,
                                                  srcPoint_c.y*_focalLength.y/srcPoint_c.z + _principalPoint.y);

    std::cout << _principalPoint.x << ", " << _principalPoint.y << std::endl;
    std::cout << _focalLength.x << ", " << _focalLength.y << std::endl;

    std::cout << projectedSrcPoint.x << ", " << projectedSrcPoint.y << std::endl;

    // compute error
    const float2 pixelDiff = projectedSrcPoint - _point_c;
    const float pixelDist = length(pixelDiff);

    std::cout << pixelDist << std::endl;
    std::cout << std::endl;

    if (pixelDist == 0) { return; }

    const float2 pixelGrad = pixelDiff / pixelDist;

//    const float3 pointGrad_c = make_float3(dot(pixelGrad,make_float2( _focalLength.x / srcPoint_c.z,                            0 )),
//                                           dot(pixelGrad,make_float2( 0                            ,_focalLength.y / srcPoint_c.z )),
//                                           dot(pixelGrad,make_float2(-(srcPoint_c.x)*_focalLength.x / (srcPoint_c.z*srcPoint_c.z),
//                                                                     -(srcPoint_c.y)*_focalLength.y / (srcPoint_c.z*srcPoint_c.z))));
//    const float3 distGrad_m = SE3Rotate(srcModel.getTransformCameraToModel(),distGrad_c);
//    pointGrad_m = SE3Rotate(srcModel.getTransformCameraToModel(),pointGrad_c);


    // src pose gradient
//    std::vector<float3> srcJ3D;
    float4 point_m = srcModel.getTransformFrameToModel(_srcFrame)*make_float4(_point_f,1);
    srcModel.getModelJacobianOfModelPoint(point_m,_srcFrame,srcJ3D);

    Eigen::MatrixXf subJ = Eigen::MatrixXf::Zero(2,srcDims);
    for (int i=0; i<srcDims; ++i) {
        const float3 j3D_c = SE3Rotate(srcModel.getTransformModelToCamera(),srcJ3D[i]);
        subJ(0,i) = dot(make_float3(_focalLength.x / srcPoint_c.z, 0, -(srcPoint_c.x)*_focalLength.x / (srcPoint_c.z*srcPoint_c.z)), j3D_c);
        subJ(1,i) = dot(make_float3(0, _focalLength.y / srcPoint_c.z, -(srcPoint_c.y)*_focalLength.y / (srcPoint_c.z*srcPoint_c.z)), j3D_c);
    }
    Eigen::Vector2f subE = Eigen::VectorXf::Zero(2);
    subE(0) = pixelDiff.x;
    subE(1) = pixelDiff.y;

    Eigen::VectorXf subJTe = _weight*subJ.transpose()*subE;
    Eigen::MatrixXf subJTJ = _weight*subJ.transpose()*subJ;

    // stick in JTe
    JTe.segment(srcOffset,srcPose.getReducedDimensions()) += subJTe;

    // stick in JTJ
    for (int i=0; i<srcDims; ++i) {
        for (int j=i; j<srcDims; ++j) {
            float val = subJTJ(i,j);
            if (val == 0) { continue; }
            JTJ.coeffRef(srcOffset + i,srcOffset + j) += val;
        }
    }

}

} // namespace dart
