#ifndef GPUKINEMATICS_H
#define GPUKINEMATICS_H

#include <vector_types.h>

#include <vector>
#include "geometry/grid_3d.h"
#include "geometry/SE3.h"
#include "model/mirrored_model.h"
#include "util/dart_types.h"

namespace dart {

// Computes forward kkinematics for a single joint configuration on the GPU
// puts it in the mirrored model passed in
void computeForwardKinematics(
        float *pose,
        int poselen,
        MirroredModel &robot,
        MirroredVector<float2> *limits);

// poses is a contiguous block of 1d arrays (e.g. floats 0-poselen are pose 1,
// poselen-2*poselen is pose 2. up to nposes
// benchmark is to turn on benchmark logging
// returns a 2D array of transforms, caller is responsible for freeing the result
// as well as array of poses passed in
// WARNING: make sure to free the array that is returned once finished
SE3 **computeForwardKinematicsBatch(float *poses, int nposes,
        int poselen,
        MirroredModel &robot,
        MirroredVector<float2> *limits,
        bool benchmark=false);

// will compute both forward and backward kinematics, will put FK in t_mfs,
// inverse kinematics in t_fms
// WARNING: caller is responsible for freeing t_mfs and t_fms once they are
// no longer needed.
void computeKinematicsBatch(
    float *poses,
    int nposes,
    int poselen,
    MirroredModel &robot,
    MirroredVector<float2> *limits,
    SE3 **&t_mfs,
    SE3 **&t_fms);

// same as above except the pointers passed in are set to the pointers for the gpu
// results and the gpu results are not freed.
// Note that the type is SE3*, the memory pointed to is a contiguous block of 
// results in order of the poses given
// pose 1 is at t_mfs, t_fms
// pose 2 is at t_mfs + numframes
// etc...
// WARNING: caller is responsible for eventually freeing the gpu memory of t_mfs, t_fms
// (use cudaFree)
void computeKinematicsBatchGPU(
    float *poses,
    int nposes,
    int poselen,
    MirroredModel &robot,
    MirroredVector<float2> *limits,
    SE3 *&t_mfs,
    SE3 *&t_fms);

}

#endif // MODTOOBS_H
