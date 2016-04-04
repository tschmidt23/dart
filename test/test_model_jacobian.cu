#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "optimization/kernels/kernel_common.h"
#include "model/host_only_model.h"
#include "model/mirrored_model.h"
#include "util/dart_io.h"
#include "util/dart_types.h"
#include "util/mirrored_memory.h"
#include "mesh/assimp_mesh_reader.h"


namespace {

__global__ void getErrorJacobianOfSingleModelPoint(float * J, const float4 * point_m, const int frame, const float3 * errorGrad3D_m,
                                             const int dims, const int * dependencies, const dart::JointType * jointTypes,
                                             const float3 * jointAxes, const dart::SE3 * T_fms, const dart::SE3 * T_mfs) {

    dart::getErrorJacobianOfModelPoint(J,*point_m,frame,*errorGrad3D_m,dims,dependencies,jointTypes,jointAxes,T_fms,T_mfs);

}


TEST(TestModelJacobianGPU,TestModelArticulationJacobianGPU) {

    const float dPose = 1e-3;
    const float magTolerance = 1e-4;
    const int nPoses = 20;

    std::vector<std::string> testModels;
    testModels.push_back("../models/leftHand/leftHand.xml");

    dart::Model::initializeRenderer(new dart::AssimpMeshReader());

    dim3 block(1,1,1);
    dim3 grid(1,1,1);

    for (int m=0; m<testModels.size(); ++m) {

        cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err,cudaSuccess);

        dart::HostOnlyModel hostModel;
        dart::readModelXML(testModels[m].c_str(),hostModel);
        hostModel.computeStructure();
        hostModel.voxelize(0.1,0.0);

        dart::MirroredModel model(hostModel,make_uint3(64,64,64),1);

        const int dims = model.getPoseDimensionality();

        std::vector<float4> testPoints_f;
        testPoints_f.push_back(make_float4(0,0,0,1));
        testPoints_f.push_back(make_float4(0.1,0.1,0.1,1));

        dart::MirroredVector<float> Jx(dims);
        dart::MirroredVector<float> Jy(dims);
        dart::MirroredVector<float> Jz(dims);

        dart::MirroredVector<float3> X(3);
        X[0] = make_float3(1,0,0);
        X[1] = make_float3(0,1,0);
        X[2] = make_float3(0,0,1);

        X.syncHostToDevice();

        for (int n=0; n<nPoses; ++n) {

            float pose[dims];
            for (int i=6; i<dims; ++i) {
                const int joint = i-6;
                const float jointMin = model.getJointMin(joint) + dPose;
                const float jointMax = model.getJointMax(joint) - dPose;
                pose[i] = jointMin + (jointMax - jointMin) * rand() / (float(RAND_MAX)); // TODO: proper randomness
            }

            float tmpPose[dims];

            for (int frame=0; frame < model.getNumFrames(); ++frame) {
                for (int p=0; p<testPoints_f.size(); ++p) {
                    float4 framePoint = testPoints_f[p];
                    model.setArticulation(pose);
                    dart::MirroredVector<float4> modelPoint(1);
                    modelPoint[0] = model.getTransformFrameToModel(frame)*framePoint;
                    modelPoint.syncHostToDevice();

                    // TODO: parallelize!!!
                    getErrorJacobianOfSingleModelPoint<<<grid,block>>>(Jx.devicePtr(),modelPoint.devicePtr(),frame,X.devicePtr(),dims,
                                                                       model.getDeviceDependencies(),model.getDeviceJointTypes(),
                                                                       model.getDeviceJointAxes(),model.getDeviceTransformsModelToFrame(),
                                                                       model.getDeviceTransformsFrameToModel());
                    Jx.syncDeviceToHost();

                    getErrorJacobianOfSingleModelPoint<<<grid,block>>>(Jy.devicePtr(),modelPoint.devicePtr(),frame,X.devicePtr()+1,dims,
                                                                       model.getDeviceDependencies(),model.getDeviceJointTypes(),
                                                                       model.getDeviceJointAxes(),model.getDeviceTransformsModelToFrame(),
                                                                       model.getDeviceTransformsFrameToModel());
                    Jy.syncDeviceToHost();

                    getErrorJacobianOfSingleModelPoint<<<grid,block>>>(Jz.devicePtr(),modelPoint.devicePtr(),frame,X.devicePtr()+2,dims,
                                                                       model.getDeviceDependencies(),model.getDeviceJointTypes(),
                                                                       model.getDeviceJointAxes(),model.getDeviceTransformsModelToFrame(),
                                                                       model.getDeviceTransformsFrameToModel());
                    Jz.syncDeviceToHost();

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
                        float3 J3Danalytic = make_float3(Jx[i],Jy[i],Jz[i]);

                        EXPECT_NEAR(J3Dnumeric.x,J3Danalytic.x,magTolerance);
                        EXPECT_NEAR(J3Dnumeric.y,J3Danalytic.y,magTolerance);
                        EXPECT_NEAR(J3Dnumeric.z,J3Danalytic.z,magTolerance);

//                        float magNumeric = length(J3Dnumeric);
//                        float magAnalytic = length(J3Danalytic);

//                        EXPECT_NEAR(magAnalytic,magNumeric,magTolerance);
                    }
                }

                err = cudaGetLastError();
                ASSERT_EQ(err,cudaSuccess);

            }
        }
    }

    dart::Model::shutdownRenderer();

}

TEST(TestModelJacobianGPU,TestModel6DoFJacobianGPU) {

    const float dPose = 1e-3;
    const float magTolerance = 1e-4;
    const int nPoses = 20;

    std::vector<std::string> testModels;
    testModels.push_back("../models/leftHand/leftHand.xml");

    dart::Model::initializeRenderer(new dart::AssimpMeshReader());

    dim3 block(1,1,1);
    dim3 grid(1,1,1);

    for (int m=0; m<testModels.size(); ++m) {

        dart::HostOnlyModel hostModel;
        dart::readModelXML(testModels[m].c_str(),hostModel);
        hostModel.computeStructure();
        hostModel.voxelize(0.1,0.0);

        dart::MirroredModel model(hostModel,make_uint3(64,64,64),1);

        const int dims = model.getPoseDimensionality();

        std::vector<float4> testPoints_f;
        testPoints_f.push_back(make_float4(0,0,0,1));
        testPoints_f.push_back(make_float4(0.1,0.1,0.1,1));

        dart::MirroredVector<float> Jx(dims);
        dart::MirroredVector<float> Jy(dims);
        dart::MirroredVector<float> Jz(dims);

        dart::MirroredVector<float3> X(3);
        X[0] = make_float3(1,0,0);
        X[1] = make_float3(0,1,0);
        X[2] = make_float3(0,0,1);

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

            for (int frame=0; frame < model.getNumFrames(); ++frame) {
                for (int p=0; p<testPoints_f.size(); ++p) {
                    float4 framePoint = testPoints_f[p];
                    model.setPose(pose);
                    dart::MirroredVector<float4> modelPoint(1);
                    modelPoint[0] = model.getTransformFrameToModel(frame)*framePoint;
                    modelPoint.syncHostToDevice();

                    // TODO: parallelize!!!
                    getErrorJacobianOfSingleModelPoint<<<grid,block>>>(Jx.devicePtr(),modelPoint.devicePtr(),frame,X.devicePtr(),dims,
                                                                       model.getDeviceDependencies(),model.getDeviceJointTypes(),
                                                                       model.getDeviceJointAxes(),model.getDeviceTransformsModelToFrame(),
                                                                       model.getDeviceTransformsFrameToModel());
                    Jx.syncDeviceToHost();

                    getErrorJacobianOfSingleModelPoint<<<grid,block>>>(Jy.devicePtr(),modelPoint.devicePtr(),frame,X.devicePtr()+1,dims,
                                                                       model.getDeviceDependencies(),model.getDeviceJointTypes(),
                                                                       model.getDeviceJointAxes(),model.getDeviceTransformsModelToFrame(),
                                                                       model.getDeviceTransformsFrameToModel());
                    Jy.syncDeviceToHost();

                    getErrorJacobianOfSingleModelPoint<<<grid,block>>>(Jz.devicePtr(),modelPoint.devicePtr(),frame,X.devicePtr()+2,dims,
                                                                       model.getDeviceDependencies(),model.getDeviceJointTypes(),
                                                                       model.getDeviceJointAxes(),model.getDeviceTransformsModelToFrame(),
                                                                       model.getDeviceTransformsFrameToModel());
                    Jz.syncDeviceToHost();

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

                        float3 J3Danalytic = make_float3(Jx[i],Jy[i],Jz[i]);

                        EXPECT_NEAR(J3Dnumeric_m.x,J3Danalytic.x,magTolerance);
                        EXPECT_NEAR(J3Dnumeric_m.y,J3Danalytic.y,magTolerance);
                        EXPECT_NEAR(J3Dnumeric_m.z,J3Danalytic.z,magTolerance);

//                        float magNumeric = length(J3Dnumeric_m);
//                        float magAnalytic = length(J3Danalytic[i]);

//                        EXPECT_NEAR(magAnalytic,magNumeric,magTolerance);

                    }
                }
            }
        }
    }

    dart::Model::shutdownRenderer();

}


} // namespace
