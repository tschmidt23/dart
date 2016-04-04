#include "priors.h"
#include "geometry/sdf.h"

namespace dart {

void Point3D3DPrior::computeContribution(Eigen::SparseMatrix<float> & JTJ,
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

    // compute error
    const float3 diff_c = srcPoint_c - _point_c;
    const float3 diff_m = SE3Rotate(srcModel.getTransformCameraToModel(),diff_c);
    const float dist = length(diff_c);

    if (dist == 0) { return; }
//    const float3 distGrad_c = diff_c / dist;
//    const float3 distGrad_m = SE3Rotate(srcModel.getTransformCameraToModel(),distGrad_c);
//    distGrad_m = SE3Rotate(srcModel.getTransformCameraToModel(),distGrad_c);

    // compute jacobian

    // src pose gradient
//    std::vector<float3> srcJ3D;
    float4 point_m = srcModel.getTransformFrameToModel(_srcFrame)*make_float4(_point_f,1);
    srcModel.getModelJacobianOfModelPoint(point_m,_srcFrame,srcJ3D);

    Eigen::MatrixXf subJ = Eigen::MatrixXf::Zero(3,srcDims); // TODO: this is kind of a waste
    for (int i=0; i<srcDims; ++i) {
        subJ(0,i) = srcJ3D[i].x;
        subJ(1,i) = srcJ3D[i].y;
        subJ(2,i) = srcJ3D[i].z;
    }
    Eigen::VectorXf subE = Eigen::MatrixXf::Zero(3,1);
    subE(0) = diff_m.x;
    subE(1) = diff_m.y;
    subE(2) = diff_m.z;

    Eigen::MatrixXf subJTJ = _weight*subJ.transpose()*subJ;
    Eigen::VectorXf subJTe = _weight*subJ.transpose()*subE;

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
