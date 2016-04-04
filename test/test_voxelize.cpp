#include "gtest/gtest.h"

#include <random>

#include "model/host_only_model.h"
#include "model/mirrored_model.h"
#include "util/string_format.h"

namespace {

TEST(TestVoxelize,TestSphereVoxelize) {

    dart::Model::initializeRenderer();

    dart::HostOnlyModel hostModel;
    const float sphereRadius = 0.1;
    const float sdfResolution = 0.005;
    const std::string sphereRadiusStr = dart::stringFormat("%f",sphereRadius);
    hostModel.addGeometry(0,dart::PrimitiveSphereType,sphereRadiusStr,sphereRadiusStr,sphereRadiusStr,"0","0","0","0","0","0",255,255,255);

    hostModel.computeStructure();
    hostModel.voxelize2(sdfResolution,0.1);

    dart::MirroredModel model(hostModel,make_uint3(64,64,64),0.01);

    ASSERT_EQ(6,hostModel.getPoseDimensionality());
    ASSERT_EQ(6,model.getPoseDimensionality());

    ASSERT_EQ(1,model.getNumSdfs());
    const dart::Grid3D<float> & sdf = model.getSdf(0);

    ASSERT_EQ(sdfResolution,sdf.resolution);

    for (int z=0; z<sdf.dim.z-1; ++z) {
        for (int y=0; y<sdf.dim.y-1; ++y) {
            for (int x=0; x<sdf.dim.x-1; ++x) {
                const float3 voxelCenter = make_float3(x + 0.5, y + 0.5, z + 0.5);
                const float voxelVal = sdf.data[x + sdf.dim.x*(y + sdf.dim.y*z)];
                EXPECT_EQ(voxelVal,sdf.getValueInterpolated(voxelCenter));
                const float3 voxelCenter_w = sdf.getWorldCoords(voxelCenter);
                EXPECT_NEAR(voxelVal*sdfResolution,length(voxelCenter_w) - sphereRadius,1e-6);
            }
        }
    }

    dart::Model::shutdownRenderer();

}

} // namespace

