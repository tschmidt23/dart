#include "gtest/gtest.h"

#include <random>

#include "mesh/assimp_mesh_reader.h"
#include "optimization/kernels/obsToMod.h"
#include "model/host_only_model.h"
#include "model/mirrored_model.h"
#include "util/ostream_operators.h"
#include "util/string_format.h"

namespace {

TEST(TestObsToModKernels,TestErrorAndDataAssociationObsToMod) {

    dart::Model::initializeRenderer(new dart::AssimpMeshReader());

    dart::HostOnlyModel hostModel;
    const float sphereRadius = 0.1;
    const float sdfResolution = 0.005;
    const std::string sphereRadiusStr = dart::stringFormat("%f",sphereRadius);
    hostModel.addGeometry(0,dart::PrimitiveSphereType,sphereRadiusStr,sphereRadiusStr,sphereRadiusStr,"0","0","0","0","0","0",255,255,255);

    hostModel.computeStructure();
    hostModel.voxelize2(sdfResolution,0.1);

    dart::MirroredModel model(hostModel,make_uint3(64,64,64),0.01);

    ASSERT_EQ(hostModel.getPoseDimensionality(),6);

    // memory setup
    const int obsWidth = 10;
    const int obsHeight = 10;
    dart::MirroredVector<float4> obsVertMap(obsWidth*obsHeight);
    dart::MirroredVector<float4> obsNormMap(obsWidth*obsHeight);
    dart::OptimizationOptions opts;
    dart::MirroredVector<dart::DataAssociatedPoint> associatedPoints(obsWidth*obsHeight);
    dart::MirroredVector<int> lastElement(1);
    dart::MirroredVector<int> debugDataAssociation(obsWidth*obsHeight);
    dart::MirroredVector<float> debugError(obsWidth*obsHeight);

    std::default_random_engine generator;
    std::bernoulli_distribution validDistribution(0.8);
    std::normal_distribution<float> normalDistribution(0,0.1);

    // fill observation map
    for (int i=0; i<obsWidth*obsHeight; ++i) {
        if (validDistribution(generator)) {
            obsVertMap[i] = make_float4(normalDistribution(generator),normalDistribution(generator),normalDistribution(generator),1);
        } else {
            obsVertMap[i] = make_float4(0,0,0,0);
        }
    }

    // sync host to device
    obsVertMap.syncHostToDevice();

    dart::PoseReduction * reduction = new dart::NullReduction(model.getPoseDimensionality()-6,0,0,0);
    dart::Pose pose(reduction);
    pose.setTransformModelToCamera(dart::SE3());
    model.setPose(pose);
    opts.distThreshold[0] = 0.05;

    std::cout << model.getTransformCameraToModel() << std::endl;
    std::cout << model.getTransformFrameToCamera(0) << std::endl;


    // run data association
    dart::errorAndDataAssociation(obsVertMap.devicePtr(),obsNormMap.devicePtr(),
                                  obsWidth,obsHeight,model,opts,
                                  associatedPoints.devicePtr(),lastElement.devicePtr(),
                                  lastElement.hostPtr(),
                                  debugDataAssociation.devicePtr(),debugError.devicePtr(),
                                  0); // TODO

    // sync device to host
    debugDataAssociation.syncDeviceToHost();
    debugError.syncDeviceToHost();

    const float3 sphereCenter_w = make_float3(0,0,0);
    const float3 sphereCenter_g = model.getSdf(0).getGridCoords(sphereCenter_w);
    std::cout << "sphereCenter_g = " << sphereCenter_g.x << ", " << sphereCenter_g.y << ", " << sphereCenter_g.z << std::endl;
    std::cout << model.getSdf(0).getValueInterpolated(sphereCenter_g) << std::endl;
    std::cout << model.getGeometryTransform(0) << std::endl;

    // check
    for (int i=0; i<obsWidth*obsHeight; ++i) {
        float4 & obsPoint = obsVertMap[i];
        if (obsPoint.w == 0) {
            EXPECT_EQ(-1,debugDataAssociation[i]);
            EXPECT_TRUE(std::isnan(debugError[i]) );
        } else {
            const float dist = length(make_float3(obsPoint)) - sphereRadius;
            if (fabs(dist) > opts.distThreshold[0]) {
                EXPECT_EQ(-1,debugDataAssociation[i]);
                EXPECT_TRUE(std::isnan(debugError[i]));
            } else {
                EXPECT_EQ( 0,debugDataAssociation[i]);
                EXPECT_NEAR( dist, debugError[i], sdfResolution / 8 );
            }
        }
    }


    dart::Model::shutdownRenderer();

    delete reduction;

}


} // namespace

