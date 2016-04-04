#ifndef DART_TYPES_H
#define DART_TYPES_H

namespace dart {

enum JointType {
    RotationalJoint = 0,
    PrismaticJoint = 1
};

enum GeomType {
    PrimitiveSphereType = 0,
    PrimitiveCylinderType = 1,
    PrimitiveCubeType = 2,
    NumPrimitives = 3,
    MeshType = 4
};

enum LossFunctionType {
    SquaredLoss,
    HuberLoss
};

}

#endif // DART_TYPES_H
