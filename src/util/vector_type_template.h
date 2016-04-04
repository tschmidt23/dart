#ifndef VECTOR_TYPE_TEMPLATE_H
#define VECTOR_TYPE_TEMPLATE_H

namespace dart {

template <typename Real>
struct VectorTypeTemplate {};

template <>
struct VectorTypeTemplate<float> {
    typedef float2 type2;
    typedef float3 type3;
    typedef float4 type4;
    static float2 (*const2)(float,float);
    static float3 (*const3)(float,float,float);
    static float4 (*const4)(float,float,float,float);
};

template <>
struct VectorTypeTemplate<double> {
    typedef double2 type2;
    typedef double3 type3;
    typedef double4 type4;
    static double2 (*const2)(double,double);
    static double3 (*const3)(double,double,double);
    static double4 (*const4)(double,double,double,double);
};

}

#endif // VECTOR_TYPE_TEMPLATE_H
