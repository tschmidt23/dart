#include "kinematics.h"

#include <cmath>
#include <ctime>

#include "kernel_common.h"
#include "geometry/grid_3d.h"
#include "geometry/SE3.h"
#include "optimization/optimization.h"
#include "util/mirrored_memory.h"

namespace dart {

static const float truncVal = 1000.0;
static const bool BENCHMARK = false;


// -=-=-=-=-=-=-=-=-=- kernels -=-=-=-=-=-=-=-=-=-
// probably need some well defined place to store the result
// that we can access later
// also a flag or something to show when we're done that the main
// thread can block on
// use cudaMemcpy(_hVector,_dVector,_length*sizeof(T),cudaMemcpyDeviceToHost); for getting things back
__global__ void gpu_computeForwardKinematics(
        const int maxidx,
        const int numtoproc,
        const int poselen,
        // array that corresponds to a pose/joint configuration
        const float * poses,
        //const int numDofs,
        // result store
        SE3 *T_mfss,
        //const int numJoints,
        const float2 * jointLims,
        const SE3 *T_pfs,
        const int *frameParents,
        const int numframes,
        const JointType *jointTypes,
        const float3 *jointAxes) {
    int start_idx = (threadIdx.x + blockIdx.x * blockDim.x) * numtoproc;
    for (int idx = start_idx;idx < start_idx + numtoproc; ++idx){
    if (idx >= maxidx) return;
    const float * pose = poses + poselen * idx;
    SE3 * T_mfs = T_mfss + numframes * idx;
    SE3 val;
    // setup
    T_mfs[0] = val;

    // need to figure out how to turn a configuration into a full set of joint angles (pose reduction to full pose)
    // for now assume that jointConfigurations has already been transformed into a full pose
    int j = 6;
    for (int f=1; f<numframes; ++f) {
        float p_ = pose[j];
        if(j >= poselen) p_ = 0.0;
        float p = fminf(fmaxf(jointLims[j-6].x,p_),jointLims[j-6].y);

        const int joint = f-1;
        SE3 T_pf = T_pfs[joint];
        switch(jointTypes[joint]) {
            case RotationalJoint:
                T_pf = T_pf*SE3Fromse3(se3(0, 0, 0,
                                p*jointAxes[joint].x, p*jointAxes[joint].y, p*jointAxes[joint].z));
                ++j;
                break;
            case PrismaticJoint:
                T_pf = T_pf*SE3Fromse3(se3(p*jointAxes[joint].x, p*jointAxes[joint].y, p*jointAxes[joint].z,
                                0, 0, 0));
                ++j;
                break;
        }
        const int parent = frameParents[f];
        T_mfs[f] = T_mfs[parent]*T_pf;
    }}
}

__global__ void gpu_computeKinematics(
        const int maxidx,
        const int numtoproc,
        const int poselen,
        // array that corresponds to a pose/joint configuration
        const float * poses,
        //const int numDofs,
        // result stores
        SE3 *T_mfss,
        SE3 *T_fmss,
        //const int numJoints,
        const float2 * jointLims,
        const SE3 *T_pfs,
        const int *frameParents,
        const int numframes,
        const JointType *jointTypes,
        const float3 *jointAxes) {
    int start_idx = (threadIdx.x + blockIdx.x * blockDim.x) * numtoproc;
    for (int idx = start_idx;idx < start_idx + numtoproc; ++idx){
    if (idx >= maxidx) return;
    const float * pose = poses + poselen * idx;
    SE3 * T_mfs = T_mfss + numframes * idx;
    SE3 * T_fms = T_fmss + numframes * idx;
    SE3 val;
    // setup
    T_mfs[0] = val;
    int j = 6;
    for (int f=1; f<numframes; ++f) {
        float p_ = pose[j];
        if(j >= poselen) p_ = 0.0;
        float p = fminf(fmaxf(jointLims[j-6].x,p_),jointLims[j-6].y);

        const int joint = f-1;
        SE3 T_pf = T_pfs[joint];
        switch(jointTypes[joint]) {
            case RotationalJoint:
                T_pf = T_pf*SE3Fromse3(se3(0, 0, 0,
                                p*jointAxes[joint].x, p*jointAxes[joint].y, p*jointAxes[joint].z));
                ++j;
                break;
            case PrismaticJoint:
                T_pf = T_pf*SE3Fromse3(se3(p*jointAxes[joint].x, p*jointAxes[joint].y, p*jointAxes[joint].z,
                                0, 0, 0));
                ++j;
                break;
        }
        const int parent = frameParents[f];
        T_mfs[f] = T_mfs[parent]*T_pf;
        T_fms[f] = SE3Invert(T_mfs[f]);
    }}
}

void computeForwardKinematics(
        float *host_pose,
        int poselen,
        MirroredModel &robot,
        MirroredVector<float2> *limits) {
    // limit pose by joint limits and extend the articulation
    // make sure not to leak around this
    //MirroredVector<float2> limits = new MirroredVector<float2>(robot._jointLimits);
    // move these to a spot on the gpu
    // need to tune these/accept larger parameters so that this makes sense
    //dim3 block(8,1,1);
    //dim3 grid(ceil(total / block.x), ceil(total / block.y));
    //std::cout << host_pose[0] << std::endl;
    float *device_pose;
    cudaError_t mal_err = cudaMalloc((void**) &device_pose, sizeof(float) * poselen);
    //float *posea = (float*) malloc(sizeof(float) * poselen);
    cudaError_t mc_err1 = cudaMemcpy(device_pose, host_pose, sizeof(float) * poselen, cudaMemcpyHostToDevice);
    //cudaError_t mc_err2 = cudaMemcpy(posea, device_pose, sizeof(float) * poselen, cudaMemcpyDeviceToHost);
    if (mal_err != cudaSuccess) {
        std::cout << "Cuda malloc error:" << cudaGetErrorString(mal_err) << std::endl;
        return;
    }
    if (mc_err1 != cudaSuccess) {
        std::cout << "Cuda copy error1:" << cudaGetErrorString(mc_err1) << std::endl;
        return;
    }
    /*if (mc_err2 != cudaSuccess) {
        std::cout << "Cuda copy error2:" << cudaGetErrorString(mc_err2) << std::endl;
        return;
    }
    /*std::cout << "Calling fk code with " << robot.getNumFrames() << " frames, " << poselen << " dimensions" << std::endl;
    for (int i = 0; i < robot.getNumFrames() - 1; ++i) {
        std::cout << "POSEX: " << posea[i] << " " << limits->hostPtr()[i].x
                  << " " << limits->hostPtr()[i].y <<  std::endl;
    }
    free(posea);*/
    robot.syncKinematicsHostToDevice();

    std::cout << 1 <<" " << poselen << " " << (uint64_t) limits->devicePtr() << std::endl;
    gpu_computeForwardKinematics<<<1,1>>>(
                1,1,
                poselen,
                device_pose,
                robot.getDeviceTransformsFrameToModel(),
                limits->devicePtr(),
                robot.getDeviceTransformsParentJointToFrame(),
                robot.getDeviceFrameParents(),
                robot.getNumFrames(),
                robot.getDeviceJointTypes(),
                robot.getDeviceJointAxes()
                );
    cudaFree(device_pose);
    robot.syncKinematics();
}

SE3 **computeForwardKinematicsBatch(
        float *host_poses,
        int nposes,
        int ndims,
        MirroredModel &robot,
        MirroredVector<float2> *limits) {

    clock_t s,e;
    if (BENCHMARK) s = clock();
    // limit pose by joint limits and extend the articulation
    // make sure not to leak around this
    //MirroredVector<float2> limits = new MirroredVector<float2>(robot._jointLimits);
    // move these to a spot on the gpu
    // need to tune these/accept larger parameters so that this makes sense
    //dim3 block(8,1,1);
    //dim3 grid(ceil(total / block.x), ceil(total / block.y));
    // round up to a power of two for better calculation
    int calc_num = std::pow(2, std::ceil(std::log(nposes)/std::log(2)));
    int nblk = 16;
    int perthr = 128;
    int threadct = max(calc_num / perthr / nblk, 1);
    int nframes = robot.getNumFrames();
    float *device_poses;
    cudaError_t mal_err = cudaMalloc((void**) &device_poses, sizeof(float) * ndims * nposes);
    if (mal_err != cudaSuccess) {
        std::cout << "Cuda malloc error:" << cudaGetErrorString(mal_err) << std::endl;
        return NULL;
    }
    SE3 *device_results;
    mal_err = cudaMalloc((void**) &device_results, sizeof(SE3) * nframes * nposes);
    if (mal_err != cudaSuccess) {
        std::cout << "Cuda malloc error:" << cudaGetErrorString(mal_err) << std::endl;
        return NULL;
    }
    cudaMemset(device_results,0, sizeof(SE3) * nframes * nposes);

    if (BENCHMARK) {
      e = clock();
      std::cout << "CUDA Mallocs: " << e << " " << s << " " << (e-s) << std::endl;
      s = clock();
    }
    cudaError_t mc_err1 = cudaMemcpy(device_poses, host_poses, sizeof(float) * ndims * nposes, cudaMemcpyHostToDevice);
    //for (int i = 0; i < host_poses.size(); ++i) {
        //cudaError_t mc_err1 = cudaMemcpy(device_poses + ndims * i, host_poses[i], sizeof(float) * ndims, cudaMemcpyHostToDevice);
        if (mc_err1 != cudaSuccess) {
            std::cout << "Cuda copy error1:" << cudaGetErrorString(mc_err1) << std::endl;
            return NULL;
        }
    //}

    if(BENCHMARK) {
      e = clock();
      std::cout << "Device setup: " << e << " " << s << " " << (e-s) << std::endl;
    }
    /*float *posea = (float*) malloc(sizeof(float) * ndims);
    cudaError_t mc_err2 = cudaMemcpy(posea, device_poses, sizeof(float) * ndims, cudaMemcpyDeviceToHost);
    if (mc_err2 != cudaSuccess) {
        std::cout << "Cuda copy error2:" << cudaGetErrorString(mc_err2) << std::endl;
        return NULL;
    }
    //std::cout << "Calling fk code with " << robot.getNumFrames() << " frames, " << ndims << " dimensions" << std::endl;
    for (int i = 0; i < ndims; ++i) {
        std::cout << "POSEX: " << posea[i] << " " << limits->hostPtr()[i].x
                  << " " << limits->hostPtr()[i].y << " " <<host_poses[i] <<  std::endl;
    }
    free(posea);//*/
    if(BENCHMARK) s = clock();
    SE3 **results = (SE3 **) malloc(sizeof(SE3*) * nposes + sizeof(SE3) * nframes * nposes);
    SE3 *datapos = (SE3 *) &results[nposes];
    // initalize raw array
    for (int i = 0; i < nposes; ++i) {
        results[i] = datapos + i * nframes;
    }
    if(BENCHMARK) {
      e = clock();
      std::cout << "End Setup: " << e << " " << s << " " << (e-s) << std::endl;
    }
    /*cudaMemcpy(datapos, device_results, sizeof(SE3) * nframes * nposes, cudaMemcpyDeviceToHost);
    std::cout << "Result " << datapos[2].r0.w << " " << datapos[2].r0.x << " " <<datapos[2].r0.y << " " <<
                 datapos[2].r0.z << " " <<std::endl;
    std::cout << nposes <<" " << ndims << " " << (uint64_t) limits->devicePtr() << std::endl;*/
    if(BENCHMARK) s = clock();
    std::cout << "Launching " << calc_num<< " "<< threadct << " " << nblk << " " << perthr << std::endl;
    gpu_computeForwardKinematics<<<threadct,nblk>>>(
                                            nposes,
                                                         perthr,
                                            ndims,
                                            device_poses,
                                            device_results,
                                            limits->devicePtr(),
                                            robot.getDeviceTransformsParentJointToFrame(),
                                            robot.getDeviceFrameParents(),
                                            robot.getNumFrames(),
                                            robot.getDeviceJointTypes(),
                                            robot.getDeviceJointAxes()
                                            );
    if (BENCHMARK) {
      e = clock();
      std::cout << "Actual computation: " << e << " " << s << " " << (e-s) << std::endl;
    }
                /*nposes,
                ndims,
                device_poses,
                robot.getDeviceTransformsFrameToModel(),//device_results,
                limits->devicePtr(),
                robot.getDeviceTransformsParentJointToFrame(),
                robot.getDeviceFrameParents(),
                robot.getNumFrames(),
                robot.getDeviceJointTypes(),
                robot.getDeviceJointAxes()
                );*/
    //cudaFree(device_poses);

    /*cudaMemcpy(datapos, device_results, sizeof(SE3) * nframes * nposes, cudaMemcpyDeviceToHost);
    std::cout << "Result " << datapos[2].r0.w << " " << datapos[2].r0.x << " " <<datapos[2].r0.y << " " <<
                 datapos[2].r0.z << " " <<std::endl;

    std::cout << (uint64_t) results << " " << (uint64_t) datapos << " " << (uint64_t) results[0] << " " << datapos[0].r0.w <<std::endl;
    std::cout << results[0][0].r0.w <<" " << datapos[0].r0.w << " " <<std::endl;*/
    if (BENCHMARK) s = clock();
    cudaMemcpy(datapos, device_results, sizeof(SE3) * nframes * nposes, cudaMemcpyDeviceToHost);
    if (BENCHMARK) {
      e = clock();
      std::cout << "End Memcpy: " << e << " " << s << " " << (e-s) << std::endl;
    }
    cudaFree(device_results);
    return results;
}

void computeKinematicsBatchGPU(
        float *host_poses,
        int nposes,
        int ndims,
        MirroredModel &robot,
        MirroredVector<float2> *limits,
        SE3 *&t_mfs,
        SE3 *&t_fms) {

    // limit pose by joint limits and extend the articulation
    // make sure not to leak around this
    //MirroredVector<float2> limits = new MirroredVector<float2>(robot._jointLimits);
    // move these to a spot on the gpu
    // need to tune these/accept larger parameters so that this makes sense
    //dim3 block(8,1,1);
    //dim3 grid(ceil(total / block.x), ceil(total / block.y));
    // round up to a power of two for better calculation
    int calc_num = std::pow(2, std::ceil(std::log(nposes)/std::log(2)));
    int nblk = 16;
    int perthr = 128;
    int threadct = max(calc_num / perthr / nblk, 1);
    int nframes = robot.getNumFrames();
    float *device_poses;
    cudaError_t mal_err = cudaMalloc((void**) &device_poses, sizeof(float) * ndims * nposes);
    if (mal_err != cudaSuccess) {
        std::cout << "Cuda malloc error:" << cudaGetErrorString(mal_err) << std::endl;
        return;
    }
    SE3 *device_mfs;
    mal_err = cudaMalloc((void**) &device_mfs, sizeof(SE3) * nframes * nposes);
    if (mal_err != cudaSuccess) {
        std::cout << "Cuda malloc error:" << cudaGetErrorString(mal_err) << std::endl;
        return;
    }
    t_mfs = device_mfs;
    cudaMemset(device_mfs,0, sizeof(SE3) * nframes * nposes);
    SE3 *device_fms;
    mal_err = cudaMalloc((void**) &device_fms, sizeof(SE3) * nframes * nposes);
    if (mal_err != cudaSuccess) {
        std::cout << "Cuda malloc error:" << cudaGetErrorString(mal_err) << std::endl;
        return;
    }
    t_fms = device_fms;
    cudaMemset(device_fms,0, sizeof(SE3) * nframes * nposes);

    cudaError_t mc_err1 = cudaMemcpy(device_poses, host_poses, sizeof(float) * ndims * nposes, cudaMemcpyHostToDevice);
    //for (int i = 0; i < host_poses.size(); ++i) {
        //cudaError_t mc_err1 = cudaMemcpy(device_poses + ndims * i, host_poses[i], sizeof(float) * ndims, cudaMemcpyHostToDevice);
        if (mc_err1 != cudaSuccess) {
            std::cout << "Cuda copy error1:" << cudaGetErrorString(mc_err1) << std::endl;
            return;
        }
    //}

    /*float *posea = (float*) malloc(sizeof(float) * ndims);
    cudaError_t mc_err2 = cudaMemcpy(posea, device_poses, sizeof(float) * ndims, cudaMemcpyDeviceToHost);
    if (mc_err2 != cudaSuccess) {
        std::cout << "Cuda copy error2:" << cudaGetErrorString(mc_err2) << std::endl;
        return;
    }
    //std::cout << "Calling fk code with " << robot.getNumFrames() << " frames, " << ndims << " dimensions" << std::endl;
    for (int i = 0; i < ndims; ++i) {
        std::cout << "POSEX: " << posea[i] << " " << limits->hostPtr()[i].x
                  << " " << limits->hostPtr()[i].y << " " <<host_poses[i] <<  std::endl;
    }
    free(posea);// */

    /*cudaMemcpy(datapos, device_results, sizeof(SE3) * nframes * nposes, cudaMemcpyDeviceToHost);
    std::cout << "Result " << datapos[2].r0.w << " " << datapos[2].r0.x << " " <<datapos[2].r0.y << " " <<
                 datapos[2].r0.z << " " <<std::endl;
    std::cout << nposes <<" " << ndims << " " << (uint64_t) limits->devicePtr() << std::endl;*/
    std::cout << "Launching " << calc_num<< " "<< threadct << " " << nblk << " " << perthr << std::endl;
    gpu_computeKinematics<<<threadct,nblk>>>(
                                            nposes,
                                             perthr,
                                            ndims,
                                            device_poses,
                                            device_mfs,
                                            device_fms,
                                            limits->devicePtr(),
                                            robot.getDeviceTransformsParentJointToFrame(),
                                            robot.getDeviceFrameParents(),
                                            robot.getNumFrames(),
                                            robot.getDeviceJointTypes(),
                                            robot.getDeviceJointAxes()
                                            );

                /*nposes,
                ndims,
                device_poses,
                robot.getDeviceTransformsFrameToModel(),//device_results,
                limits->devicePtr(),
                robot.getDeviceTransformsParentJointToFrame(),
                robot.getDeviceFrameParents(),
                robot.getNumFrames(),
                robot.getDeviceJointTypes(),
                robot.getDeviceJointAxes()
                );*/
    //cudaFree(device_poses);

    /*cudaMemcpy(datapos, device_results, sizeof(SE3) * nframes * nposes, cudaMemcpyDeviceToHost);
    std::cout << "Result " << datapos[2].r0.w << " " << datapos[2].r0.x << " " <<datapos[2].r0.y << " " <<
                 datapos[2].r0.z << " " <<std::endl;

    std::cout << (uint64_t) results << " " << (uint64_t) datapos << " " << (uint64_t) results[0] << " " << datapos[0].r0.w <<std::endl;
    std::cout << results[0][0].r0.w <<" " << datapos[0].r0.w << " " <<std::endl;*/
}

void computeKinematicsBatch(
        float *host_poses,
        int nposes,
        int ndims,
        MirroredModel &robot,
        MirroredVector<float2> *limits,
        SE3 **&t_mfs,
        SE3 **&t_fms) {
    int nframes = robot.getNumFrames();
    t_mfs = (SE3 **) malloc(sizeof(SE3*) * nposes + sizeof(SE3) * nframes * nposes);
    t_fms = (SE3 **) malloc(sizeof(SE3*) * nposes + sizeof(SE3) * nframes * nposes);
    SE3 *mfs_pos = (SE3 *) &t_mfs[nposes];
    // initalize raw array
    for (int i = 0; i < nposes; ++i) {
        t_mfs[i] = mfs_pos + i * nframes;
    }
    SE3 *fms_pos = (SE3 *) &t_fms[nposes];
    // initalize raw array
    for (int i = 0; i < nposes; ++i) {
        t_fms[i] = fms_pos + i * nframes;
    }
    SE3 *device_mfs;
    SE3 *device_fms;
    computeKinematicsBatchGPU(host_poses, nposes, ndims, robot, limits, device_mfs, device_fms);
    cudaMemcpy(mfs_pos, device_mfs, sizeof(SE3) * nframes * nposes, cudaMemcpyDeviceToHost);
    cudaMemcpy(fms_pos, device_fms, sizeof(SE3) * nframes * nposes, cudaMemcpyDeviceToHost);
    cudaFree(device_mfs);
    cudaFree(device_fms);
}

}
