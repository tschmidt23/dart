#ifndef GPUKINEMATICS_H
#define GPUKINEMATICS_H

#include <vector_types.h>

#include <vector>
#include "geometry/grid_3d.h"
#include "geometry/SE3.h"
#include "model/mirrored_model.h"
#include "util/dart_types.h"

namespace dart {

void computeForwardKinematics(
        float *pose,
        int poselen,
        MirroredModel &robot,
        MirroredVector<float2> *limits);

SE3 **computeForwardKinematicsBatch(float *poses, int nposes,
        int poselen,
        MirroredModel &robot,
        MirroredVector<float2> *limits);

}

#endif // MODTOOBS_H
