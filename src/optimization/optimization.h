#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <vector>
#include <cuda_runtime.h>

namespace dart {

struct OptimizationOptions {

    int numIterations;                              /**< The number of solver iterations run. */
    std::vector<float> distThreshold;               /**< The maximum distance (in meters) at which data association of an observed point may be considered valid (one value per model). */
    std::vector<float> regularization;              /**< The amount of regularization to be applied at each iteration of the solver. This value times the identity matrix will be added to the Hessian approximation before solving the normal equations (one value per model). */
    float lambdaObsToMod;                           /**< Determines the weight applied to errors induced by observed points in the model SDF. */
    float lambdaModToObs;                           /**< Determines the weight applied to errors induced by predicted model poins in the observation SDF. */
    std::vector<float> lambdaIntersection;          /**< Determines the weight applied to errors induced by model points intersecting other model SDFs. If N models have been added to the Tracker, there should be N<sup>2</sup> values, where lambdaIntersection[i + j*N] gives the weight assigned to model i intersecting model j.  */
    std::vector<float> planeOffset;                 /**< Together with planeNormal, this allows for specifying a clipping plane for rejecting data association. If the dot product of an observed vertex (transformed into model space) with the planeNormal is less than the planeOffset, the point is rejected (one per model). */
    std::vector<float3> planeNormal;                /**< Together with planeNormal, this allows for specifying a clipping plane for rejecting data association. If the dot product of an observed vertex (transformed into model space) with the planeNormal is less than the planeOffset, the point is rejected (one per model). */
    std::vector<float> regularizationScaled;            /**< The amount of (scaled) regularization to be applied at each iteration of the solver. This value times the diagonal of the Hessian approximation will be added to the Hessian approximation before solving the normal equations (one value per model). */

    float focalLength; // TODO: remove?
    float normThreshold; // TODO: remove?
    float huberDelta;
    float contactThreshold; // TODO: remove?

    // debugging options
    bool debugObsToModDA;                           /**< Tells the optimizer whether to compute a dense data association map for debugging. If true, the map can be accessed with Optimizer::getDeviceDebugDataAssociationObsToMod(). */
    bool debugObsToModErr;                          /**< Tells the optimizer whether to compute a dense error map for debugging. If true, the map can be accessed with Optimizer::getDeviceDebugErrorObsToMod(). */
    bool debugModToObsDA;                           /**< Tells the optimizer whether to compute a dense data association map for debugging. If true, the map can be accessed with Optimizer::getDeviceDebugDataAssociationModToObs(). */
    bool debugModToObsErr;                          /**< Tells the optimizer whether to compute a dense error map for debugging. If true, the map can be accessed with Optimizer::getDeviceDebugErrorModToObs(). */
    bool debugObsToModNorm;
    bool debugModToObsNorm;
    bool debugObsToModJs;
    bool debugIntersectionErr;
    bool debugJTJ;

    OptimizationOptions() :
        numIterations(5),
        focalLength(537.0),
        normThreshold(-0.1),
        distThreshold(1,0.03),
        regularization(1,1e-20),
        lambdaObsToMod(1.0),
        lambdaModToObs(1.0),
        lambdaIntersection(1,0.0),
        planeOffset(1,-0.03),
        planeNormal(1,make_float3(0,0,0)),

        regularizationScaled(1,1),

        debugObsToModDA(false),
        debugObsToModErr(false),
        debugModToObsDA(false),
        debugModToObsErr(false),
        debugObsToModNorm(false),
        debugModToObsNorm(false),

        debugJTJ(false),

        contactThreshold(0.01),
        huberDelta(0.02)

    { }
};

__host__ __device__ inline int JTJSize(const int dimensions) { return ((dimensions*(dimensions+1))>>1); }

__host__ __device__ inline int JTJIndex(const int i, const int j){
    if(i >= j){
        return ((i*(i+1))>>1) + j;
    }
    else{
        return ((j*(j+1))>>1) + i;
    }
}

struct DataAssociatedPoint {
    int index;
    int dataAssociation;
    float error;
};

}

#endif // OPTIMIZATION_H
