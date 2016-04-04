#include "optimizer.h"

#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include "kernels/intersection.h"
#include "kernels/modToObs.h"
#include "kernels/obsToMod.h"
#include "geometry/distance_transforms.h"
#include "util/ostream_operators.h"
#include "visualization/matrix_viz.h"
#include "util/cuda_utils.h"

namespace dart {

Optimizer::Optimizer(const DepthSourceBase * depthSource, const int predictionWidth, const int predictionHeight) {

    const int pWidth = predictionWidth == -1 ? depthSource->getDepthWidth() : predictionWidth;
    const int pHeight = predictionHeight == -1 ? depthSource->getDepthHeight() : predictionHeight;

    const float2 focalLength = depthSource->getFocalLength()*pWidth/(float)depthSource->getDepthWidth();

    init(depthSource->getDepthWidth(),depthSource->getDepthHeight(),focalLength,pWidth,pHeight);

}

Optimizer::Optimizer(const int depthWidth, const int depthHeight, const float2 focalLength,
                     const int predictionWidth, const int predictionHeight) {

    int pWidth = predictionWidth == -1 ? depthWidth : predictionWidth;
    int pHeight = predictionHeight == -1 ? depthHeight : predictionHeight;

    init(depthWidth,depthHeight,focalLength,pWidth,pHeight);
}

void Optimizer::init(const int depthWidth, const int depthHeight, const float2 focalLength,
                     const int predictionWidth, const int predictionHeight) {

    std::cout << "predicting depth maps at " << predictionWidth << " x " << predictionHeight << std::endl;
    std::cout << "prediction focal length: " << focalLength.x << ", " << focalLength.y << std::endl;

    _predictionRenderer = new PredictionRenderer(predictionWidth,predictionHeight,focalLength);

    _maxModels = 3;
    _maxDims = 100;
    _maxIntersectionSites = 131072; // TODO: ???

    // scratch buffering
    cudaMalloc(&_dDebugObsToModNorm,predictionWidth*predictionHeight*sizeof(float4));
    cudaMalloc(&_dDebugModToObsNorm,predictionWidth*predictionHeight*sizeof(float4));
    cudaMalloc(&_dError,sizeof(float));

    cudaMalloc(&_dDebugDataAssocObsToMod,depthWidth*depthHeight*sizeof(int));
    cudaMemset(_dDebugDataAssocObsToMod,0,depthWidth*depthHeight*sizeof(int));

    cudaMalloc(&_dDebugDataAssocModToObs,predictionWidth*predictionHeight*sizeof(int));
    cudaMemset(_dDebugDataAssocModToObs,0,predictionWidth*predictionHeight*sizeof(int));

    cudaMalloc(&_dDebugObsToModError,depthWidth*depthHeight*sizeof(float));
    cudaMemset(_dDebugObsToModError,0,depthWidth*depthHeight*sizeof(float));

    cudaMalloc(&_dDebugModToObsError,predictionWidth*predictionHeight*sizeof(float));
    cudaMemset(_dDebugModToObsError,0,predictionWidth*predictionHeight*sizeof(float));

    cudaMalloc(&_dDebugIntersectionError,_maxIntersectionSites*sizeof(float));
    {
        std::vector<float> hostDebugIntersectionError(_maxIntersectionSites,42); // TODO
        cudaMemcpy(_dDebugIntersectionError,hostDebugIntersectionError.data(),_maxIntersectionSites*sizeof(float),cudaMemcpyHostToDevice);
    }

    // TODO
    cudaMalloc(&_dDebugObsToModJs,depthWidth*depthHeight*26*sizeof(float3));

    _lastElements = new MirroredVector<int>(_maxModels);

    _dPts = new MirroredVector<DataAssociatedPoint*>(_maxModels);

    for (int i=0; i<_maxModels; ++i) {
        cudaMalloc(&_dPts->hostPtr()[i],depthWidth*depthHeight*sizeof(DataAssociatedPoint));
    }

    _dPts->syncHostToDevice();


    _JTJ.resize(_maxModels);
    for (int i=0; i<_maxModels; ++i) {
        _JTJ[i] = new Eigen::MatrixXf();
    }

    _iterationSummaries.resize(_maxModels);
    for (int i=0; i<_maxModels; ++i) { _iterationSummaries[i].resize(10); }

   _result = new MirroredVector<float>(_maxDims + JTJSize(_maxDims) + 1);

   _JTJimg = new MirroredVector<uchar3>(320*240);

    // make streams
    cudaStreamCreate(&_depthPredictStream);
    cudaStreamCreate(&_posInfoStream);

    _sigmas = 0;

}

Optimizer::~Optimizer() {
    delete _predictionRenderer;
    delete _lastElements;
    delete _result;
    for (int i=0; i<_maxModels; ++i) {
        cudaFree(_dPts->hostPtr()[i]);
    }
    delete _dPts;
    delete _JTJimg;

    cudaFree(_dDebugDataAssocObsToMod);
    cudaFree(_dDebugObsToModError);
    cudaFree(_dDebugModToObsError);

    cudaFree(_dDebugObsToModNorm);
    cudaFree(_dDebugModToObsNorm);
    cudaFree(_dError);
    cudaFree(_dDebugDataAssocModToObs);

    cudaFree(_dDebugIntersectionError);

    cudaStreamDestroy(_depthPredictStream);
    cudaStreamDestroy(_posInfoStream);

}

void Optimizer::unpack(Eigen::VectorXf & eJ,
                       Eigen::MatrixXf & JTJ,
                       float & e,
                       float * sys,
                       const float multiplier,
                       const int dimensions) {

    float * sys_eJ = sys;
    float * sys_JTJ = &sys[dimensions];
    float * sys_e = &sys[dimensions + JTJSize(dimensions)];

    for (int j=0; j<dimensions; ++j) {
        eJ(j) += multiplier*sys_eJ[j];
        for (int i=0; i<=j; ++i) {
            JTJ(i,j) += multiplier*sys_JTJ[((j*(j+1))>>1) + i];
        }
    }

    e = multiplier*(*sys_e);

}

void Optimizer::generateObsSdf(MirroredModel & model,
                               const Observation & observation,
                               const OptimizationOptions & opts) {

    generateObsSdfSplatZeroAndDistanceTransform(model,observation,opts);
//    generateObsSdfDirectTruncated(model,observation,opts);

}

void Optimizer::generateObsSdfSplatZeroAndDistanceTransform(MirroredModel & model,
                                                            const Observation & observation,
                                                            const OptimizationOptions & opts) {
    //    const float3 defaultOffset = model.getObsSdfoffset();
    //    const float4 sdfCenter =  model.getTransformModelToCamera()*make_float4(defaultOffset.x,defaultOffset.y,defaultOffset.z,1);

    Grid3D<float> & hObsSdf = *model.getObsSdf();
    const uint3 & dim = hObsSdf.dim;
    //    const float & resolution = hObsSdf.resolution;
    //    const float4 o = make_float4(sdfCenter.x - resolution*dim.x*0.5,
    //                                 sdfCenter.y - resolution*dim.y*0.5,
    //                                 sdfCenter.z - resolution*dim.z*0.5,
    //                                 1);

    //    hObsSdf.offset = make_float3(o);

    model.syncObsSdfHostToDevice();

    {
        splatObsSdfZeros(observation.dVertMap,
                         observation.width,
                         observation.height,
                         model.getTransformModelToCamera(),
                         model.getDeviceObsSdf(),
                         hObsSdf.dim,
                         opts.focalLength);

        // TODO:
        static float * dTmp = 0;
        static float * dZ = 0;
        static int * dV = 0;
        static int maxDim = 0;
        if ((dim.x+1)*(dim.y+1)*(dim.z+1) > maxDim) {
            maxDim = (dim.x+1)*(dim.y+1)*(dim.z+1);
            cudaFree(dTmp);
            cudaFree(dZ);
            cudaFree(dV);
            cudaMalloc(&dTmp,dim.x*dim.y*dim.z*sizeof(float));
            cudaMalloc(&dZ,(dim.x+1)*(dim.y+1)*(dim.z+1)*sizeof(float));
            cudaMalloc(&dV,dim.x*dim.y*dim.z*sizeof(int));
        }

        distanceTransform3D<float,true>(model.getDeviceObsSdfData(),
                                        dTmp,
                                        dim.x,dim.y,dim.z,
                                        dZ,dV);

        cudaMemcpy(model.getDeviceObsSdfData(),dTmp,dim.x*dim.y*dim.z*sizeof(float),cudaMemcpyDeviceToDevice);

    }

}

void Optimizer::generateObsSdfDirectTruncated(MirroredModel & model,
                                              const Observation & observation,
                                              const OptimizationOptions & opts) {

    Grid3D<float> & hObsSdf = *model.getObsSdf();
    const uint3 & dim = hObsSdf.dim;

    computeTruncatedObsSdf(observation.dVertMap,observation.width,observation.height,model.getTransformCameraToModel(),model.getDeviceObsSdf(),dim,4);

}

void Optimizer::computeObsToModContribution(Eigen::VectorXf & eJ, Eigen::MatrixXf & JTJ, float & error,
                                            const MirroredModel & model, const Pose & pose,
                                            const OptimizationOptions & opts, const Observation & observation) {

    const int dims = pose.getReducedDimensions();
    const int modelNum = model.getModelID();

    if (pose.isReduced()) {
        const LinearPoseReduction * reduction = static_cast<const LinearPoseReduction *>(pose.getReduction());
        if (reduction->isParamMap()) {
            const ParamMapPoseReduction * paramMapReduction = static_cast<const ParamMapPoseReduction *>(reduction);
            normEqnsObsToModParamMap(pose.getDimensions(),
                                     dims, paramMapReduction->getDeviceMapping(),
                                     observation.dVertMap, observation.width, observation.height,
                                     model, opts, _dPts->hostPtr()[modelNum], _lastElements->hostPtr()[modelNum],
                                     _result->devicePtr());
        } else {
            normEqnsObsToModReduced(pose.getDimensions(),
                                    dims,
                                    pose.getDeviceFirstDerivatives(),
                                    observation.dVertMap,
                                    observation.width, observation.height,
                                    model,
                                    opts,
                                    _dPts->hostPtr()[modelNum],
                                    _lastElements->hostPtr()[modelNum], // TODO
                                    _result->devicePtr());
        }
    } else {
        normEqnsObsToMod(dims,
                         observation.dVertMap,
                         observation.width, observation.height,
                         model,
                         opts,
                         _dPts->hostPtr()[modelNum],
                         _lastElements->hostPtr()[modelNum], // TODO
                         _result->devicePtr(),
                         0); // TODO
    }

    cudaMemcpy(_result->hostPtr(),_result->devicePtr(),(dims + JTJSize(dims) + 1)*sizeof(float),cudaMemcpyDeviceToHost);
    unpack(eJ,JTJ,error,_result->hostPtr(),opts.lambdaObsToMod,dims);

}

void Optimizer::computeModToObsContribution(Eigen::VectorXf & eJ, Eigen::MatrixXf & JTJ, float & error,
                                            const MirroredModel & model, const Pose & pose,
                                            const SE3 & T_obsSdf_c,
                                            const OptimizationOptions & opts) {

    const int dims = pose.getReducedDimensions();

    if (pose.isReduced()) {
        const LinearPoseReduction * reduction = static_cast<const LinearPoseReduction *>(pose.getReduction());
        if (reduction->isParamMap()) {
            const ParamMapPoseReduction * paramMapReduction = static_cast<const ParamMapPoseReduction *>(reduction);
            normEqnsModToObsParamMap(pose.getDimensions(),
                                     dims, paramMapReduction->getDeviceMapping(),
                                     _predictionRenderer->getDevicePrediction(),
                                     _predictionRenderer->getWidth(),
                                     _predictionRenderer->getHeight(),
                                     model,
                                     _result->devicePtr(),
                                     _lastElements->devicePtr() + model.getModelID(),
                                     opts.debugModToObsDA ? _dDebugDataAssocModToObs : 0,
                                     opts.debugModToObsErr ? _dDebugModToObsError : 0,
                                     opts.debugModToObsNorm ? _dDebugModToObsNorm : 0);
        } else {
            normEqnsModToObsReduced(pose.getDimensions(),
                                    dims,
                                    pose.getDeviceFirstDerivatives(),
                                    _predictionRenderer->getDevicePrediction(),
                                    _predictionRenderer->getWidth(),
                                    _predictionRenderer->getHeight(),
                                    model,
                                    _result->devicePtr(),
                                    _lastElements->devicePtr() + model.getModelID(),
                                    opts.debugModToObsDA ? _dDebugDataAssocModToObs : 0,
                                    opts.debugModToObsErr ? _dDebugModToObsError : 0,
                                    opts.debugModToObsNorm ? _dDebugModToObsNorm : 0);
        }
    } else {
        normEqnsModToObs(dims,
                         _predictionRenderer->getDevicePrediction(),
                         _predictionRenderer->getWidth(),
                         _predictionRenderer->getHeight(),
                         model,
                         T_obsSdf_c,
                         _result->devicePtr(),
                         _lastElements->devicePtr() + model.getModelID(),
                         opts.debugModToObsDA ? _dDebugDataAssocModToObs : 0,
                         opts.debugModToObsErr ? _dDebugModToObsError : 0,
                         opts.debugModToObsNorm ? _dDebugModToObsNorm : 0);
    }

    cudaMemcpy(_result->hostPtr(),_result->devicePtr(),(dims + JTJSize(dims) + 1)*sizeof(float),cudaMemcpyDeviceToHost);
    unpack(eJ,JTJ,error,_result->hostPtr(),opts.lambdaModToObs,dims);

}

void Optimizer::computeSelfIntersectionContribution(Eigen::VectorXf & eJ, Eigen::MatrixXf & JTJ, float & error,
                                                    const MirroredModel & model, const Pose & pose,
                                                    const OptimizationOptions & opts,
                                                    const MirroredVector<float4> & collisionCloud,
                                                    const MirroredVector<int> & intersectionPotentialMatrix,
                                                    const int nModels, const int debugOffset) {

    const int dims = pose.getReducedDimensions();
    const int modelNum = model.getModelID();

    if (pose.isReduced()) {
        const LinearPoseReduction * reduction = static_cast<const LinearPoseReduction *>(pose.getReduction());
        if (reduction->isParamMap()) {
            const ParamMapPoseReduction * paramMapReduction = static_cast<const ParamMapPoseReduction *>(reduction);
            normEqnsSelfIntersectionParamMap(collisionCloud.devicePtr(),
                                             collisionCloud.length(),
                                             pose.getDimensions(),
                                             dims,
                                             model,
                                             paramMapReduction->getDeviceMapping(),
                                             intersectionPotentialMatrix.devicePtr(),
                                             _result->devicePtr());
        } else {
            normEqnsSelfIntersectionReduced(collisionCloud.devicePtr(),
                                            collisionCloud.length(),
                                            pose.getDimensions(),
                                            dims,
                                            model,
                                            pose.getDeviceFirstDerivatives(),
                                            intersectionPotentialMatrix.devicePtr(),
                                            _result->devicePtr());
        }
    } else {
        normEqnsSelfIntersection(collisionCloud.devicePtr(),
                                 collisionCloud.length(),
                                 dims,
                                 model,
                                 intersectionPotentialMatrix.devicePtr(),
                                 _result->devicePtr(),
                                 opts.debugIntersectionErr ? _dDebugIntersectionError + debugOffset : 0);
    }

    cudaMemcpy(_result->hostPtr(),_result->devicePtr(),((dims-6) + JTJSize((dims-6)) + 1)*sizeof(float),cudaMemcpyDeviceToHost);

    // TODO
    Eigen::MatrixXf JTJtmp = Eigen::MatrixXf::Zero(dims-6,dims-6);
    Eigen::VectorXf eJtmp = Eigen::VectorXf::Zero(dims-6);
    unpack(eJtmp,JTJtmp,error,_result->hostPtr(),opts.lambdaIntersection[modelNum + modelNum*nModels],(dims-6));

    eJ.tail(dims-6) += eJtmp;
    JTJ.bottomRightCorner(dims-6,dims-6) += JTJtmp;

}

void Optimizer::computeIntersectionContribution(Eigen::VectorXf & eJ, Eigen::MatrixXf & JTJ, float & error,
                                                const MirroredModel & srcModel, const MirroredModel & dstModel,
                                                const Pose & pose, const OptimizationOptions & opts,
                                                const MirroredVector<float4> & collisionCloud,
                                                const int nModels, const int debugOffset) {

    const int dims = pose.getReducedDimensions();
    const int srcModelNum = srcModel.getModelID();
    const int dstModelNum = dstModel.getModelID();

    const SE3 T_ds = dstModel.getTransformCameraToModel()*srcModel.getTransformModelToCamera();
    const SE3 T_sd = SE3Invert(T_ds);

    if (pose.isReduced()) {
        const LinearPoseReduction * reduction = static_cast<const LinearPoseReduction *>(pose.getReduction());
        if (reduction->isParamMap()) {
            const ParamMapPoseReduction * paramMapReduction = static_cast<const ParamMapPoseReduction *>(reduction);
            normEqnsIntersectionParamMap(collisionCloud.devicePtr(), collisionCloud.length(),
                                         pose.getDimensions(),dims,T_ds,T_sd,srcModel,dstModel,
                                         paramMapReduction->getDeviceMapping(),_result->devicePtr());
        } else {
            normEqnsIntersectionReduced(collisionCloud.devicePtr(), collisionCloud.length(),
                                        pose.getDimensions(),dims,T_ds,T_sd,srcModel,dstModel,
                                        pose.getDeviceFirstDerivatives(),_result->devicePtr());
        }
    } else {

        normEqnsIntersection(collisionCloud.devicePtr(), collisionCloud.length(),
                             dims,T_ds,T_sd,srcModel,dstModel,_result->devicePtr(),
                             opts.debugIntersectionErr ? _dDebugIntersectionError + debugOffset : 0);
    }

    cudaMemcpy(_result->hostPtr(),_result->devicePtr(),(dims + JTJSize(dims) + 1)*sizeof(float),cudaMemcpyDeviceToHost);
    unpack(eJ,JTJ,error,_result->hostPtr(),opts.lambdaIntersection[srcModelNum + dstModelNum*nModels ],dims);

}

void Optimizer::optimizePose(MirroredModel & model,
                             Pose & pose,
                             const float4 * dObsVertMap,
                             const float4 * dObsNormMap,
                             const int width,
                             const int height,
                             OptimizationOptions & opts,
                             MirroredVector<float4> & collisionCloud,
                             MirroredVector<int> & intersectionPotentialMatrix,
                             const dart::PosePrior * prior) {

    int fullDims = pose.getDimensions();
    int redDims = pose.getReducedDimensions();

    Eigen::MatrixXf JTJ(redDims,redDims);
    float totalLoss;
    float lmLambda = opts.regularizationScaled[0];

    bool predictionsNeeded = (opts.lambdaModToObs > 0);

    Observation observation(dObsVertMap,dObsNormMap,width,height);

    SE3 T_obsSdf_camera;
    if (opts.lambdaModToObs > 0) {
        generateObsSdf(model, observation, opts);
        T_obsSdf_camera = model.getTransformCameraToModel();
    }

    for (int iteration=0; iteration < opts.numIterations; ++iteration) {

        pose.projectReducedToFull();
        model.setPose(pose);

        if (predictionsNeeded) {
            std::vector<const MirroredModel *> modelPtrs(1,&model);
            _predictionRenderer->raytracePrediction(modelPtrs,_depthPredictStream);
            cudaStreamSynchronize(_depthPredictStream);
        }

        JTJ = Eigen::MatrixXf::Zero(redDims,redDims);
        Eigen::VectorXf eJ = Eigen::VectorXf::Zero(redDims);
        totalLoss = 0.0;

        float obsToModError = 0;
        float modToObsError = 0;
        float intersectionError = 0;

        if (opts.lambdaObsToMod > 0) {
            errorAndDataAssociation(dObsVertMap,dObsNormMap,width,height,model,opts,_dPts->hostPtr()[0],_lastElements->devicePtr(),_lastElements->hostPtr(),
                    opts.debugObsToModDA ? _dDebugDataAssocObsToMod : 0, opts.debugObsToModErr ? _dDebugObsToModError : 0, opts.debugObsToModNorm ? _dDebugObsToModNorm : 0);
            computeObsToModContribution(eJ,JTJ,obsToModError,model,pose,opts,observation);
        }
        if (opts.lambdaModToObs > 0) {
            computeModToObsContribution(eJ,JTJ,modToObsError,model,pose,T_obsSdf_camera,opts);
        }
        if (opts.lambdaIntersection[0] > 0) {
            computeSelfIntersectionContribution(eJ,JTJ,intersectionError,model,pose,opts,collisionCloud,intersectionPotentialMatrix,1,0);
        }

        //make JTJ symmetric
        for (int i=0; i<redDims; i++){
            for (int j=0; j<i; j++){
               JTJ(i,j) = JTJ(j,i);
            }
        }

        // ensure JTJ is full rank
        JTJ += opts.regularization[0] * Eigen::MatrixXf::Identity(redDims,redDims);

        SE3 T_mc = model.getTransformCameraToModel();

        Eigen::MatrixXf A = JTJ;

        for (int i=0; i<redDims; ++i) {
            A(i,i) += lmLambda*A(i,i);
        }

        // compute update
        Eigen::VectorXf dalpha = -A.ldlt().solve(eJ);
        //            std::cout << dalpha.transpose() << std::endl;

        // compute next pose
        SE3 dT_mc = SE3Fromse3(se3(dalpha(0),dalpha(1),dalpha(2),dalpha(3),dalpha(4),dalpha(5)));
        SE3 new_T_mc = dT_mc*T_mc;

        for (int i=0; i<pose.getArticulatedDimensions(); ++i) {
            if (!pose.isReduced()) {
                pose.getReducedArticulation()[i] = std::min(std::max(model.getJointMin(i),pose.getReducedArticulation()[i] + dalpha(i+6)),model.getJointMax(i));
            } else {
                pose.getReducedArticulation()[i] = std::min(std::max(pose.getReducedMin(i),pose.getReducedArticulation()[i] + dalpha(i+6)),pose.getReducedMax(i));
            }
        }

        pose.projectReducedToFull();
        pose.setTransformCameraToModel(new_T_mc);
        model.setPose(pose);

    }

}

void Optimizer::optimizePoses(std::vector<MirroredModel *> & models,
                              std::vector<Pose> & poses,
                              const float4 * dObsVertMap,
                              const float4 * dObsNormMap,
                              const int width,
                              const int height,
                              OptimizationOptions & opts,
                              MirroredVector<SE3> & T_mcs,
                              MirroredVector<SE3 *> & T_fms,
                              MirroredVector<int *> & sdfFrames,
                              MirroredVector<const Grid3D<float> *> & sdfs,
                              MirroredVector<int> & nSdfs,
                              MirroredVector<float> & distanceThresholds,
                              MirroredVector<float> & normalThresholds,
                              MirroredVector<float> & planeOffsets,
                              MirroredVector<float3> &  planeNormals,
                              std::vector<MirroredVector<float4> *> & collisionClouds,
                              std::vector<MirroredVector<int> *> & intersectionPotentialMatrices,
                              std::vector<Eigen::MatrixXf *> & dampingMatrices,
                              std::vector<Prior *> & priors) {

    // resize scratch space if there are more models than we've seen before
    const int nModels = models.size();
    const int nPriors = priors.size();
    if (nModels > _maxModels) {
        _dPts->resize(nModels);
        for (int i=_maxModels; i<nModels; ++i) {
            cudaMalloc(&_dPts->hostPtr()[i],width*height*sizeof(DataAssociatedPoint));
        }
        _dPts->syncHostToDevice();
        _lastElements->resize(nModels);
        _JTJ.resize(nModels);
        for (int i=_maxModels; i<nModels; ++i) {
            _JTJ[i] = new Eigen::MatrixXf();
        }
        _iterationSummaries.resize(nModels);

        _maxModels = nModels;
    }

    bool predictionsNeeded = (opts.lambdaModToObs > 0);
    Observation observation(dObsVertMap,dObsNormMap,width,height);

    memcpy(planeOffsets.hostPtr(),opts.planeOffset.data(),nModels*sizeof(float));
    memcpy(planeNormals.hostPtr(),opts.planeNormal.data(),nModels*sizeof(float3));
    memcpy(distanceThresholds.hostPtr(),opts.distThreshold.data(),nModels*sizeof(float));
    planeOffsets.syncHostToDevice();
    planeNormals.syncHostToDevice();
    distanceThresholds.syncHostToDevice();

    std::vector<SE3> T_obsSdfs_camera;
    if (opts.lambdaModToObs > 0) {
        for (int m=0; m<nModels; ++m) {
            generateObsSdf(*models[m],observation,opts); //,_negInfoStream);
            T_obsSdfs_camera.push_back(models[m]->getTransformCameraToModel());
        }
    }

    int sysSize = 0;
    int modelOffsets[nModels];
    int priorOffsets[nPriors];
    for (int m=0; m<nModels; ++m) {
        _iterationSummaries[m].resize(opts.numIterations);
        modelOffsets[m] = sysSize;
        sysSize += poses[m].getReducedDimensions();
    }
    for (int p=0; p<nPriors; ++p) {
        priorOffsets[p] = sysSize;
        sysSize += priors[p]->getNumPriorParams();
    }

    Eigen::SparseMatrix<float> sparseJTJ(sysSize,sysSize);
    Eigen::VectorXf fullJTe(sysSize);

    for (int iteration=0; iteration < opts.numIterations; ++iteration) {

        sparseJTJ.setZero();
        fullJTe = Eigen::VectorXf::Zero(sysSize);

        for (int m=0; m<nModels; ++m) {
            poses[m].projectReducedToFull();
            models[m]->setPose(poses[m]);
            T_mcs.hostPtr()[m] = models[m]->getTransformCameraToModel();
        }
        T_mcs.syncHostToDevice();

        if (predictionsNeeded) {
            // TODO: fix this
            std::vector<const MirroredModel*> constModelPtrs(nModels);
            for (int m=0; m<nModels; ++m) {
                constModelPtrs[m] = models[m];
            }
            _predictionRenderer->raytracePrediction(constModelPtrs,_depthPredictStream);
            _predictionRenderer->cullUnobservable(dObsVertMap,width,height,_depthPredictStream);
        }
        
        errorAndDataAssociationMultiModel(dObsVertMap,dObsNormMap,width,height,nModels,
                                          T_mcs.devicePtr(),T_fms.devicePtr(),
                                          sdfFrames.devicePtr(),sdfs.devicePtr(),
                                          nSdfs.devicePtr(),distanceThresholds.devicePtr(),
                                          normalThresholds.devicePtr(),planeOffsets.devicePtr(),
                                          planeNormals.devicePtr(),_lastElements->devicePtr(),
                                          _dPts->devicePtr(),
                                          opts.debugObsToModDA ? _dDebugDataAssocObsToMod : 0,
                                          opts.debugObsToModErr ? _dDebugObsToModError : 0,
                                          opts.debugObsToModNorm ? _dDebugObsToModNorm : 0,
                                          _posInfoStream);


//        int sysSize = 3*opts.contactPriors.size();
//        for (int m=0; m<nModels; ++m) { sysSize += poses[m].getReducedDimensions(); }
//        Eigen::MatrixXf JTJ = Eigen::MatrixXf::Zero(sysSize,sysSize);
//        Eigen::VectorXf eJ = Eigen::VectorXf::Zero(sysSize);

        cudaStreamSynchronize(_depthPredictStream);
        cudaStreamSynchronize(_posInfoStream);

        _lastElements->syncDeviceToHost(); // needed in compute obs to mod contribution
//        for (int m=0; m<nModels; ++m) {
//            std::cout << _lastElements->hostPtr()[m] << " points associated to model " << m << std::endl;
//        }

        int debugIntersectionOffset = 0;
        if (opts.debugIntersectionErr) {
            initDebugIntersectionError(_dDebugIntersectionError,_maxIntersectionSites);
//            cudaMemset(_dDebugIntersectionError,0,_maxIntersectionSites*sizeof(float));
        }
        for (int m=0; m<nModels; ++m) {

            MirroredModel & model = *models[m];
            Pose & pose = poses[m];
            const float lmLambda = opts.regularizationScaled[m];

            const int dimensions = pose.getDimensions();
            const int reducedDimensions = pose.getReducedDimensions();
            Eigen::MatrixXf & JTJ = *_JTJ[m];
            JTJ = Eigen::MatrixXf::Zero(reducedDimensions,reducedDimensions);
            Eigen::VectorXf eJ = Eigen::VectorXf::Zero(reducedDimensions);
//            float obsToModErr = 0;
//            float modToObsErr = 0;
            float intersectionError = 0;

            if (opts.lambdaObsToMod > 0) {
                computeObsToModContribution(eJ,JTJ,_iterationSummaries[m][iteration].errObsToMod,model,pose,opts,observation);
                _iterationSummaries[m][iteration].nAssociatedPoints = _lastElements->hostPtr()[m];
            }
            if (opts.lambdaModToObs > 0) {
                computeModToObsContribution(eJ,JTJ,_iterationSummaries[m][iteration].errModToObs,model,pose,T_obsSdfs_camera[m],opts);
            }
            if (opts.lambdaIntersection[m + m*nModels] > 0) {
                computeSelfIntersectionContribution(eJ,JTJ,intersectionError,model,pose,opts,
                                                    *collisionClouds[m],*intersectionPotentialMatrices[m], nModels,
                                                    debugIntersectionOffset);
            }
            for (int d=0; d<nModels; ++d) {
                if (d == m) { continue; }
                if (opts.lambdaIntersection[m + d*nModels] > 0) {
                    computeIntersectionContribution(eJ,JTJ,intersectionError,model,*models[d],pose,opts,
                                                    *collisionClouds[m],nModels,debugIntersectionOffset);
                }
            }

            // TODO: get rid of redundancy
            // make JTJ symmetric
            for (int i=0; i<reducedDimensions; i++){
                for (int j=0; j<i; j++){
                    JTJ(i,j) = JTJ(j,i);
                }
            }

            // ensure JTJ is full rank
            JTJ += opts.regularization[m] * Eigen::MatrixXf::Identity(reducedDimensions,reducedDimensions);

            // add LM damping
            for (int i=0; i<reducedDimensions; ++i) {
                JTJ(i,i) += lmLambda*JTJ(i,i);
            }

            // add damping matrix
            JTJ += *dampingMatrices[m];

            std::cout << JTJ << std::endl << std::endl << std::endl;

            for (int i=0; i<reducedDimensions; ++i) {
                for (int j=i; j<reducedDimensions; ++j) {
                    if (JTJ(i,j) != 0) {
                        sparseJTJ.coeffRef(modelOffsets[m]+i,modelOffsets[m]+j) = JTJ(i,j);
                    }
                }
            }
            fullJTe.segment(modelOffsets[m],reducedDimensions) = eJ;

            debugIntersectionOffset += collisionClouds[m]->length();

        }

        if (opts.lambdaModToObs > 0) {
            _lastElements->syncDeviceToHost();
            for (int m=0; m<nModels; ++m) {
                _iterationSummaries[m][iteration].nPredictedPoints = _lastElements->hostPtr()[m];
            }
        }

        for (int p=0; p<priors.size(); ++p) {
            priors[p]->computeContribution(sparseJTJ,fullJTe,modelOffsets,priorOffsets[p],models,poses,opts);
        }

        Eigen::VectorXf paramUpdate = -sparseJTJ.triangularView<Eigen::Upper>().solve(fullJTe);

        for (int m=0; m<nModels; ++m) {
            MirroredModel & model = *models[m];
            Pose & pose = poses[m];

            SE3 T_mc = model.getTransformCameraToModel();

            SE3 dT_mc = SE3Fromse3(se3(paramUpdate(modelOffsets[m] + 0),paramUpdate(modelOffsets[m] + 1),paramUpdate(modelOffsets[m] + 2),
                                       paramUpdate(modelOffsets[m] + 3),paramUpdate(modelOffsets[m] + 4),paramUpdate(modelOffsets[m] + 5)));
            SE3 new_T_mc = dT_mc*T_mc;

            for (int i=0; i<pose.getReducedArticulatedDimensions(); ++i) {
                if (!pose.isReduced()) {
                    pose.getReducedArticulation()[i] = std::min(std::max(model.getJointMin(i),pose.getArticulation()[i] + paramUpdate(modelOffsets[m] + i + 6)),model.getJointMax(i));
                } else {
                    pose.getReducedArticulation()[i] = std::min(std::max(pose.getReducedMin(i),pose.getReducedArticulation()[i] + paramUpdate(modelOffsets[m] + i + 6)),pose.getReducedMax(i));
                }
            }

            pose.setTransformCameraToModel(new_T_mc);
            pose.projectReducedToFull();
            model.setPose(pose);
        }

        for (int p=0; p<priors.size(); ++p) {
            priors[p]->updatePriorParams(paramUpdate.data() + priorOffsets[p],models);
        }

    }

    if (opts.debugJTJ) {

        Eigen::MatrixXf denseJTJ(sysSize,sysSize);
        for (int i=0; i<sysSize; ++i) {
            for (int j=i; j<sysSize; ++j) {
                denseJTJ(i,j) = sparseJTJ.coeff(i,j);
                denseJTJ(j,i) = denseJTJ(i,j);
            }
        }
        MirroredVector<float> JTJdata(sysSize*sysSize);
        memcpy(JTJdata.hostPtr(),denseJTJ.data(),JTJdata.length()*sizeof(float));
        JTJdata.syncHostToDevice();
        visualizeMatrix(JTJdata.devicePtr(),sysSize,sysSize,_JTJimg->devicePtr(),320,240,make_uchar3(100,0,100),0.0f,500.0f);
        _JTJimg->syncDeviceToHost();
    }

    CheckCudaDieOnError();
}


}
