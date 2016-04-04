#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "model/host_only_model.h"
#include "util/dart_io.h"
#include "mesh/assimp_mesh_reader.h"

namespace {

TEST(TestModelJacobian,TestModelArticulationJacobian) {

    const float dPose = 1e-3;
    const float magTolerance = 1e-4;
    const int nPoses = 20;

    std::vector<std::string> testModels;
    testModels.push_back("../models/leftHand/leftHand.xml");

    for (int m=0; m<testModels.size(); ++m) {

        dart::Model::initializeRenderer(new dart::AssimpMeshReader());
        dart::HostOnlyModel model;
        dart::readModelXML(testModels[m].c_str(),model);
        model.computeStructure();

        const int dims = model.getPoseDimensionality();

        std::vector<float4> testPoints_f;
        testPoints_f.push_back(make_float4(0,0,0,1));
        testPoints_f.push_back(make_float4(0.1,0.1,0.1,1));

        for (int n=0; n<nPoses; ++n) {

            float pose[dims];
            for (int i=6; i<dims; ++i) {
                const int joint = i-6;
                const float jointMin = model.getJointMin(joint) + dPose;
                const float jointMax = model.getJointMax(joint) - dPose;
                pose[i] = jointMin + (jointMax - jointMin) * rand() / (float(RAND_MAX)); // TODO: proper randomness
            }

            std::vector<float3> J3Danalytic;
            float tmpPose[dims];

            for (int frame=0; frame < model.getNumFrames(); ++frame) {
                for (int p=0; p<testPoints_f.size(); ++p) {
                    float4 framePoint = testPoints_f[p];
                    model.setArticulation(pose);
                    float4 modelPoint = model.getTransformFrameToModel(frame)*framePoint;
                    model.getModelJacobianOfModelPoint(modelPoint,frame,J3Danalytic);

                    ASSERT_EQ(J3Danalytic.size(),dims);

                    for (int joint=0; joint<model.getNumJoints(); ++joint) {
                        int i = joint+6;
                        memcpy(tmpPose,pose,dims*sizeof(float));

                        tmpPose[i] = pose[i] - dPose;
                        model.setArticulation(tmpPose);
                        float4 neg = model.getTransformFrameToModel(frame)*framePoint;

                        tmpPose[i] = pose[i] + dPose;
                        model.setArticulation(tmpPose);
                        float4 pos = model.getTransformFrameToModel(frame)*framePoint;

                        float3 J3Dnumeric = make_float3((1/(2*dPose))*(pos-neg));

                        float magNumeric = length(J3Dnumeric);
                        float magAnalytic = length(J3Danalytic[i]);

                        EXPECT_NEAR(magAnalytic,magNumeric,magTolerance);
                    }
                }
            }
        }
    }

    dart::Model::shutdownRenderer();

}

TEST(TestModelJacobian,TestModel6DoFJacobian) {

    const float dPose = 1e-3;
    const float magTolerance = 1e-4;
    const int nPoses = 20;

    std::vector<std::string> testModels;
    testModels.push_back("../models/leftHand.xml");

    for (int m=0; m<testModels.size(); ++m) {

        dart::Model::initializeRenderer(new dart::AssimpMeshReader());
        dart::HostOnlyModel model;
        dart::readModelXML(testModels[m].c_str(),model);

        const int dims = model.getPoseDimensionality();

        std::vector<float4> testPoints_f;
        testPoints_f.push_back(make_float4(0,0,0,1));
        testPoints_f.push_back(make_float4(0.1,0.1,0.1,1));

        std::vector<float> mins(dims);
        std::vector<float> maxs(dims);
        std::vector<std::string> names(dims);
        dart::Pose pose(new dart::NullReduction(dims,mins.data(),maxs.data(),names.data()));

        for (int n=0; n<nPoses; ++n) {

//            float pose[dims];
            for (int joint=0; joint<dims-6; ++joint) {
                const float jointMin = model.getJointMin(joint) + dPose;
                const float jointMax = model.getJointMax(joint) - dPose;
                pose.getArticulation()[joint] = jointMin + (jointMax - jointMin) * ( rand() / ((float)(RAND_MAX)) ); // TODO: proper randomness
            }

            dart::se3 t_mc;
            for (int i=0; i<3; ++i) {
                t_mc.p[i] = -0.5 + rand() / (float)(RAND_MAX);
                t_mc.p[i+3] = -M_PI + 2*M_PI*( rand() / ((float)(RAND_MAX)) );
            }
            dart::SE3 T_mc = dart::SE3Fromse3(t_mc);
            pose.setTransformCameraToModel(T_mc);

            std::vector<float3> J3Danalytic;

            for (int frame=0; frame < model.getNumFrames(); ++frame) {
                for (int p=0; p<testPoints_f.size(); ++p) {
                    float4 framePoint = testPoints_f[p];
                    model.setPose(pose);
                    float4 modelPoint = model.getTransformFrameToModel(frame)*framePoint;
                    model.getModelJacobianOfModelPoint(modelPoint,frame,J3Danalytic);

                    ASSERT_EQ(J3Danalytic.size(),dims);

                    dart::se3 dt_mc;
                    for (int i=0; i<6; ++i) {

                        memset(dt_mc.p,0,6*sizeof(float));
                        dt_mc.p[i] = -dPose;
                        pose.setTransformCameraToModel(dart::SE3Fromse3(dt_mc)*T_mc);
                        model.setPose(pose);
                        float4 neg = model.getTransformFrameToCamera(frame)*framePoint;

                        dt_mc.p[i] = dPose;
                        pose.setTransformCameraToModel(dart::SE3Fromse3(dt_mc)*T_mc);
                        model.setPose(pose);
                        float4 pos = model.getTransformFrameToCamera(frame)*framePoint;

                        float3 J3Dnumeric_c = make_float3((1/(2*dPose))*(pos-neg));
                        float3 J3Dnumeric_m = dart::SE3Rotate(T_mc,J3Dnumeric_c);

                        float magNumeric = length(J3Dnumeric_m);
                        float magAnalytic = length(J3Danalytic[i]);

                        EXPECT_NEAR(magAnalytic,magNumeric,magTolerance);

                    }
                }
            }
        }
    }

    dart::Model::shutdownRenderer();

}


}
