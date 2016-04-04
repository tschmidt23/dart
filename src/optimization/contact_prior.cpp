#include "priors.h"
#include "geometry/sdf.h"

namespace dart {

void ContactPrior::computeContribution(Eigen::SparseMatrix<float> & JTJ,
                                       Eigen::VectorXf & JTe,
                                       const int * modelOffsets,
                                       const int priorParamOffset,
                                       const std::vector<MirroredModel *> & models,
                                       const std::vector<Pose> & poses,
                                       const OptimizationOptions & opts) {

    if (_weight == 0) { return; }

    const MirroredModel & srcModel = *models[_srcModelID];
    const MirroredModel & dstModel = *models[_dstModelID];

    const Pose & srcPose = poses[_srcModelID];
    const Pose & dstPose = poses[_dstModelID];

    const int srcOffset = modelOffsets[_srcModelID];
    const int dstOffset = modelOffsets[_dstModelID];

    const int srcDims = srcPose.getReducedDimensions();
    const int dstDims = dstPose.getReducedDimensions();

    const int srcFrame = srcModel.getSdfFrameNumber(_srcSdfNum);
    const int dstFrame = dstModel.getSdfFrameNumber(_dstSdfNum);

    const float4 contact_sf = make_float4(_contactPoint,1);
    const float4 contact_sm = srcModel.getTransformFrameToModel(srcFrame)*contact_sf;
    const float4 contact_c = srcModel.getTransformModelToCamera()*contact_sm;
    const float4 contact_dm = dstModel.getTransformCameraToModel()*contact_c;
    const float4 contact_df = dstModel.getTransformModelToFrame(dstFrame)*contact_dm;

    const Grid3D<float> & dstSdf = dstModel.getSdf(_dstSdfNum);
    const float3 contact_g = dstSdf.getGridCoords(make_float3(contact_df));

    if (!dstSdf.isInBoundsGradientInterp(contact_g)) {
        return;
    }

    // compute contact error
    const float errContact = dstSdf.getValueInterpolated(contact_g)*dstSdf.resolution;

    if (errContact > opts.contactThreshold) {
        return;
    }

    // compute contact jacobian
    const float3 sdfGrad_df = dstSdf.getGradientInterpolated(contact_g);
    const float3 sdfGrad_dm = SE3Rotate(dstModel.getTransformFrameToModel(dstFrame),sdfGrad_df);
    const float3 sdfGrad_c = SE3Rotate(dstModel.getTransformModelToCamera(),sdfGrad_dm);
    const float3 sdfGrad_sm = SE3Rotate(srcModel.getTransformCameraToModel(),sdfGrad_c);
    const float3 sdfGrad_sf = SE3Rotate(srcModel.getTransformModelToFrame(srcFrame),sdfGrad_sm);

    const int subSysSize = srcDims + dstDims + 3;

    // contact point gradient
    Eigen::VectorXf subJ = Eigen::VectorXf::Zero(subSysSize);
    subJ(srcDims + dstDims + 0) = sdfGrad_sf.x;
    subJ(srcDims + dstDims + 1) = sdfGrad_sf.y;
    subJ(srcDims + dstDims + 2) = sdfGrad_sf.z;

    // dst pose gradient
//    std::vector<float3> dstJ3D;
    dstModel.getModelJacobianOfModelPoint(contact_dm,dstFrame,dstJ3D);

    for (int i=0; i<dstDims; ++i) {
        subJ(srcDims + i) = -dot(sdfGrad_dm,dstJ3D[i]);
    }

    // src pose gradient
//    std::vector<float3> srcJ3D;
    srcModel.getModelJacobianOfModelPoint(contact_sm,srcFrame,srcJ3D);

    for (int i=0; i<srcDims; ++i) {
        subJ(i) = dot(sdfGrad_sm,srcJ3D[i]);
    }

    Eigen::VectorXf subJTe = _weight*errContact*subJ;
    Eigen::MatrixXf subJTJ = _weight*subJ*subJ.transpose();

    // regularization
    for (int i=0; i<3; ++i) {
        subJTJ(srcDims+dstDims+i,srcDims+dstDims+i) += _regularization; // TODO: _weight*_regularization?
    }

    // stick in JTe
    JTe.segment(srcOffset,srcPose.getReducedDimensions()) += subJTe.head(srcDims);
    JTe.segment(dstOffset,dstPose.getReducedDimensions()) += subJTe.segment(srcDims,dstDims);
    JTe.segment(priorParamOffset,3) += subJTe.tail(3);

    // stick digonal blocks in JTJ
    for (int i=0; i<srcDims; ++i) {
        for (int j=i; j<srcDims; ++j) {
            float val = subJTJ(i,j); // TODO: write a #define for these three lines or something
            if (val == 0) { continue; }
            JTJ.coeffRef(srcOffset + i,srcOffset + j) += val;
        }
    }
    for (int i=0; i<dstDims; ++i) {
        for (int j=i; j<dstDims; ++j) {
            float val = subJTJ(srcDims+i,srcDims+j);
            if (val == 0) { continue; }
            JTJ.coeffRef(dstOffset + i,dstOffset + j) += val;
        }
    }
    for (int i=0; i<3; ++i) {
        for (int j=i; j<3; ++j) {
            float val = subJTJ(srcDims+dstDims+i,srcDims+dstDims+j);
            if ( val == 0) { continue; }
            JTJ.coeffRef(priorParamOffset + i,priorParamOffset + j) += val;
        }
    }

    // stick off-diagonal blocks in JTJ
    for (int i=0; i<srcDims; ++i) {
        for (int j=0; j<3; ++j) {
            float val = subJTJ(i,srcDims+dstDims+j);
            if (val == 0) { continue; }
            JTJ.coeffRef(srcOffset + i,priorParamOffset + j) += val;
        }
    }
    for (int i=0; i<dstDims; ++i) {
        for (int j=0; j<3; ++j) {
            float val = subJTJ(srcDims+i,srcDims+dstDims+j);
            if (val == 0) { continue; }
            JTJ.coeffRef(dstOffset + i,priorParamOffset + j) += val;
        }
    }
    // use only upper triangular src/dst block
    if (srcOffset < dstOffset) {
        for (int i=0; i<srcDims; ++i) {
            for (int j=0; j<dstDims; ++j) {
                float val = subJTJ(i,srcDims+j);
                if (val == 0) { continue; }
                JTJ.coeffRef(srcOffset + i,dstOffset + j) += val;
            }
        }
    } else {
        for (int i=0; i<dstDims; ++i) {
            for (int j=0; j<srcDims; ++j) {
                float val = subJTJ(srcDims+i,j);
                if (val == 0) { continue; }
                JTJ.coeffRef(dstOffset + i,srcOffset + j) += val;
            }
        }
    }

    _error = errContact;
}

void ContactPrior::updatePriorParams(const float * update, const std::vector<MirroredModel *> & models) {

    _contactPoint.x += update[0];
    _contactPoint.y += update[1];
    _contactPoint.z += update[2];

    const Grid3D<float> sdf = models[_srcModelID]->getSdf(_srcSdfNum);
    float3 est_g = sdf.getGridCoords(_contactPoint);
    projectToSdfSurface(sdf,est_g,1e-9);
    _contactPoint = sdf.getWorldCoords(est_g);

}


}
