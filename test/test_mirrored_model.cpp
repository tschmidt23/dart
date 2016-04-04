#include "gtest/gtest.h"

#include "mesh/assimp_mesh_reader.h"
#include "model/mirrored_model.h"
#include "util/dart_io.h"

namespace {

TEST(TestMirroredModel,TestMirredModelConstructorDestructor) {

    std::vector<std::string> testModels;
    testModels.push_back("../models/leftHand/leftHand.xml");

    cudaError_t err = cudaGetLastError();

    dart::Model::initializeRenderer(new dart::AssimpMeshReader());


    for (int m=0; m<testModels.size(); ++m) {

        err = cudaGetLastError();
        ASSERT_EQ(err,cudaSuccess);

        dart::HostOnlyModel hostModel;
        dart::readModelXML(testModels[m].c_str(),hostModel);
        hostModel.computeStructure();

        dart::MirroredModel mirroredModel(hostModel,make_uint3(64,64,64),0.01);

    }

    err = cudaGetLastError();
    ASSERT_EQ(err,cudaSuccess);

    dart::Model::shutdownRenderer();

}

} // namespace
