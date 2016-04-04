#include "gtest/gtest.h"

#include "pose/pose.h"
#include "util/string_format.h"

namespace {

TEST(TestPoseReduction,TestNullPoseReduction) {

    const int nDims = 10;
    const int nArticulatedDims = nDims-6;

    std::vector<float> mins(nArticulatedDims);
    std::vector<float> maxs(nArticulatedDims);
    std::vector<std::string> names(nArticulatedDims);
    for (int i=0; i<nArticulatedDims; ++i) {
        names[i] = dart::stringFormat("%d",i);
        mins[i] = -i;
        maxs[i] = i;
    }

    dart::NullReduction reduction(nArticulatedDims, mins.data(), maxs.data(), names.data());

    ASSERT_TRUE(reduction.isNull());

    dart::Pose pose(&reduction);

    ASSERT_FALSE(pose.isReduced());

    ASSERT_EQ(pose.getDimensions(),nDims);
    ASSERT_EQ(pose.getArticulatedDimensions(),nArticulatedDims);
    ASSERT_EQ(pose.getReducedDimensions(),nDims);
    ASSERT_EQ(pose.getReducedArticulatedDimensions(),nArticulatedDims);

    for (int i=0; i<nArticulatedDims; ++i) {
        ASSERT_EQ(pose.getReducedMin(i),-(float)i);
        ASSERT_EQ(pose.getReducedMax(i),(float)i);
    }

    pose.getReducedArticulation()[0] = 5.f;
    ASSERT_EQ(pose.getArticulation()[0],5.f);

    pose.getArticulation()[nArticulatedDims-1] = 4.2f;
    ASSERT_EQ(pose.getReducedArticulation()[nArticulatedDims-1],4.2f);

}

} // namespace
