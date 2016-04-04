#ifndef SE3_H
#define SE3_H

#include <math.h>
#include <string.h>

#define REAL_SE3

#include "util/prefix.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>

namespace dart {

// structs and raw constructors
struct SE3 {
    float4 r0, r1, r2;
    inline PREFIX SE3();
    inline PREFIX SE3(float4 r0_, float4 r1_, float4 r2_);
    inline PREFIX SE3(float * data);
    inline PREFIX SE3(const float * data);
};

struct se3 {
    float p[6];
    inline PREFIX se3();
    inline PREFIX se3(const float x, const float y, const float z, const float wx, const float wy, const float wz);
    inline PREFIX se3(const float3 translation, const float3 rotation);
    inline PREFIX se3(float * data);
    inline PREFIX se3(const float * data);
};

// methods for trnaslating between SE3 and alternate transformation representations
inline PREFIX SE3 SE3FromDH(const float theta, const float d, const float a, const float alpha);
inline PREFIX SE3 SE3FromPosAxis(const float3 position, const float3 axis, const float theta);
inline PREFIX SE3 SE3FromTranslation(const float3 translation);
inline PREFIX SE3 SE3FromTranslation(const float x, const float y, const float z);
inline PREFIX SE3 SE3FromRotationX(const float theta);
inline PREFIX SE3 SE3FromRotationY(const float theta);
inline PREFIX SE3 SE3FromRotationZ(const float theta);
inline PREFIX SE3 SE3Fromse3(const se3 & t);
inline PREFIX se3 se3FromSE3(const SE3 & A);
inline PREFIX float3 eulerFromSE3(const SE3 & T);
inline PREFIX void eulerFromSE3(const SE3 & T, float & phi, float & theta, float & psi );
inline PREFIX SE3 SE3FromEuler(float3 phiThetaPsi);

// geometric operations
inline PREFIX SE3 SE3Transform(const SE3 & A, const SE3 & B);
inline PREFIX float4 SE3Transform(const SE3 & A, const float4 & b);
inline PREFIX float4 SE3Rotate(const SE3 & A, const float4 & b);
inline PREFIX float3 SE3Transform(const SE3 & A, const float3 & b);
inline PREFIX float3 SE3Rotate(const SE3 & A, const float3 & b);
inline PREFIX SE3 SE3Invert(const SE3 & A);

// interpolation
inline PREFIX SE3 SE3Interpolate(const SE3 & A, const SE3 & B, const float t);

// operators
inline PREFIX SE3 operator*(const SE3 & A, const SE3 & B);
inline PREFIX float4 operator*(const SE3 & A, const float4 & b);
inline PREFIX float3 operator*(const SE3 & A, const float3 & b);
inline PREFIX se3 operator*(const se3 & a, const float c);
inline PREFIX se3 operator*(const float c, const se3& a);

// ----------------------------------------------------------------------------------------------------------------- //
// Inline function definitions
// ----------------------------------------------------------------------------------------------------------------- //
inline PREFIX SE3::SE3() : r0(make_float4(1,0,0,0)), r1(make_float4(0,1,0,0)), r2(make_float4(0,0,1,0)) { }

inline PREFIX SE3::SE3(float4 r0_, float4 r1_, float4 r2_) : r0(r0_), r1(r1_), r2(r2_) { }

inline PREFIX SE3::SE3(float * data) :
    r0(make_float4(data[0],data[3],data[6],data[9])),
    r1(make_float4(data[1],data[4],data[7],data[10])),
    r2(make_float4(data[2],data[5],data[8],data[11])) { }

inline PREFIX SE3::SE3(const float * data) :
    r0(make_float4(data[0],data[3],data[6],data[9])),
    r1(make_float4(data[1],data[4],data[7],data[10])),
    r2(make_float4(data[2],data[5],data[8],data[11])) { }

inline PREFIX se3::se3() { memset(p,0,6*sizeof(float)); }

inline PREFIX se3::se3(const float x, const float y, const float z, const float wx, const float wy, const float wz) {
    p[0] = x; p[1] = y; p[2] = z; p[3] = wx; p[4] = wy; p[5] = wz;
}

inline PREFIX se3::se3(const float3 translation, const float3 rotation) {
    p[0] = translation.x; p[1] = translation.y; p[2] = translation.z; p[3] = rotation.x; p[4] = rotation.y; p[5] = rotation.z;
}

inline PREFIX se3::se3(float * data) { memcpy(p,data,6*sizeof(float)); }

inline PREFIX se3::se3(const float * data) { memcpy(p,data,6*sizeof(float)); }

inline PREFIX SE3 SE3FromDH(const float theta, const float d, const float a, const float alpha) {
    return SE3(make_float4(cosf(theta),   -sinf(theta)*cosf(alpha),   sinf(theta)*sinf(alpha),  a*cosf(theta)),
               make_float4(sinf(theta),   cosf(theta)*cosf(alpha),    -cosf(theta)*sinf(alpha), a*sinf(theta)),
               make_float4(0,             sinf(alpha),                cosf(alpha),              d));
}

inline PREFIX float3 eulerFromSE3(const SE3 & T) {
    float3 phiThetaPsi;
    eulerFromSE3(T,phiThetaPsi.x,phiThetaPsi.y,phiThetaPsi.z);
    return phiThetaPsi;
}

inline PREFIX void eulerFromSE3(const SE3 & T, float & phi, float & theta, float & psi ) {
    // R = Rz(phi)Ry(theta)Rx(psi)
    if ((fabsf(T.r2.x) - 1.f) < -1e-6) {
        theta = -asin(T.r2.x);
        psi = atan2(T.r2.y/cos(theta),T.r2.z/cos(theta));
        phi = atan2(T.r1.x/cos(theta),T.r0.x/cos(theta));
    } else {
        phi = 0.f;
        if (T.r2.x > 0) {
            theta = -M_PI_2;
            psi = atan2(-T.r0.y,-T.r0.z);
        } else {
            theta = M_PI_2;
            psi = atan2(T.r0.y,T.r0.z);
        }
    }
}

inline PREFIX SE3 SE3FromEuler(float3 phiThetaPsi) {
    const float & phi = phiThetaPsi.x;
    const float & theta = phiThetaPsi.y;
    const float & psi = phiThetaPsi.z;
    const float cPhi = cos(phi); const float sPhi = sin(phi);
    const float cTheta = cos(theta); const float sTheta = sin(theta);
    const float cPsi = cos(psi); const float sPsi = sin(psi);
    return SE3(make_float4(cTheta*cPhi, sPsi*sTheta*cPhi - cPsi*sPhi, cPsi*sTheta*cPhi + sPsi*sPhi, 0),
               make_float4(cTheta*sPhi, sPsi*sTheta*sPhi + cPsi*cPhi, cPsi*sTheta*sPhi - sPsi*cPhi, 0),
               make_float4(-sTheta,     sPsi*cTheta,                  cPsi*cTheta,                  0));
}

#ifdef REAL_SE3
inline PREFIX SE3 SE3Fromse3(const se3 & t) {

    const float & wx = t.p[3];
    const float & wy = t.p[4];
    const float & wz = t.p[5];
    const float theta = sqrtf(wx*wx + wy*wy + wz*wz);

//    std::cout << "->theta: "  << theta << std::endl;
//    std::cout << "->displacement: " << sqrt(t.p[0]*t.p[0] + t.p[1]*t.p[1] + t.p[2]*t.p[2]) << std::endl;

    if (theta == 0) {
        return SE3(make_float4(1,0,0,t.p[0]),
                   make_float4(0,1,0,t.p[1]),
                   make_float4(0,0,1,t.p[2]));
    }

    const float Va = (1-cos(theta)) / (theta*theta);
    const float Vb = (theta - sin(theta)) / (theta*theta*theta);

    const float tx = (1 +          Vb*(- wz*wz - wy*wy))*t.p[0] + (     Va*-wz + Vb*(wx*wy)          )*t.p[1] + (     Va* wy + Vb*(wx*wz)          )*t.p[2];
    const float ty = (    Va* wz + Vb*(wx*wy)          )*t.p[0] + (1 +           Vb*(- wz*wz - wx*wx))*t.p[1] + (     Va*-wx + Vb*(wy*wz)          )*t.p[2];
    const float tz = (    Va*-wy + Vb*(wx*wz)          )*t.p[0] + (     Va* wx + Vb*(wy*wz)          )*t.p[1] + (1 +           Vb*(- wy*wy - wx*wx))*t.p[2];

//    std::cout << tx << ", " << ty << ", " << tz << std::endl;
//    std::cout << sqrtf(tx*tx + ty*ty + tz*tz) << std::endl;

    const float a = sinf(theta) / theta;
    const float b = (1-cosf(theta)) / (theta*theta);

    return SE3(make_float4(1 +         b*(- wz*wz - wy*wy),     a*-wz + b*(wx*wy)          ,     a* wy + b*(wx*wz)          ,tx),
               make_float4(    a* wz + b*(wx*wy)          , 1 +         b*(- wz*wz - wx*wx),     a*-wx + b*(wy*wz)          ,ty),
               make_float4(    a*-wy + b*(wx*wz)          ,     a* wx + b*(wy*wz)          , 1 +       + b*(- wy*wy - wx*wx),tz));

}
#else
inline PREFIX SE3 SE3Fromse3(const se3 &t) {
    const float theta = sqrtf(t.p[3]*t.p[3] + t.p[4]*t.p[4] + t.p[5]*t.p[5]);
    if (theta == 0) {
        return SE3(make_float4(1,0,0,t.p[0]),
                   make_float4(0,1,0,t.p[1]),
                   make_float4(0,0,1,t.p[2]));
    }

    const float rx = t.p[3] / theta;
    const float ry = t.p[4] / theta;
    const float rz = t.p[5] / theta;

    const float s_theta = sinf(theta);
    const float omc_theta = 1-cosf(theta);

    return SE3(make_float4( 1 + omc_theta*(rx*rx-1),          s_theta*-rz + omc_theta*(rx*ry),    s_theta*ry +omc_theta*(rx*rz),      t.p[0]),
               make_float4( s_theta*rz + omc_theta*(rx*ry),   1 + omc_theta*(ry*ry-1),            s_theta*-rx + omc_theta*(ry*rz),    t.p[1]),
               make_float4( s_theta*-ry + omc_theta*(rx*rz),  s_theta*rx + omc_theta*(ry*rz),     1 + omc_theta*(rz*rz-1),            t.p[2]));
}
#endif

inline PREFIX SE3 SE3FromTranslation(const float3 translation) {
    return SE3(make_float4(1,0,0,translation.x),
               make_float4(0,1,0,translation.y),
               make_float4(0,0,1,translation.z));
}

inline PREFIX SE3 SE3FromTranslation(const float x, const float y, const float z) {
    return SE3(make_float4(1,0,0,x),
               make_float4(0,1,0,y),
               make_float4(0,0,1,z));
}

inline PREFIX SE3 SE3FromRotationX(const float theta) {
    return SE3(make_float4(1,         0,          0,0),
               make_float4(0,cos(theta),-sin(theta),0),
               make_float4(0,sin(theta), cos(theta),0));
}

inline PREFIX SE3 SE3FromRotationY(const float theta) {
    return SE3(make_float4( cos(theta),0,sin(theta),0),
               make_float4(          0,1,         0,0),
               make_float4(-sin(theta),0,cos(theta),0));
}

inline PREFIX SE3 SE3FromRotationZ(const float theta) {
    return SE3(make_float4(cos(theta),-sin(theta),0,0),
               make_float4(sin(theta), cos(theta),0,0),
               make_float4(         0,          0,1,0));
}

inline PREFIX SE3 SE3FromPosAxis(const float3 position, const float3 axis, const float theta) {

    if (theta == 0) {
        return SE3(make_float4(1,0,0,position.x),
                   make_float4(0,1,0,position.y),
                   make_float4(0,0,1,position.z));
    }

    const float s_theta = sinf(theta);
    const float omc_theta = 1-cosf(theta);

    return SE3(make_float4( 1 + omc_theta*(axis.x*axis.x-1),              s_theta*-axis.z + omc_theta*(axis.x*axis.y),    s_theta*axis.y +omc_theta*(axis.x*axis.z),      position.x),
               make_float4( s_theta*axis.z + omc_theta*(axis.x*axis.y),   1 + omc_theta*(axis.y*axis.y-1),                s_theta*-axis.x + omc_theta*(axis.y*axis.z),    position.y),
               make_float4( s_theta*-axis.y + omc_theta*(axis.x*axis.z),  s_theta*axis.x + omc_theta*(axis.y*axis.z),     1 + omc_theta*(axis.z*axis.z-1),                position.z));
}

#ifdef REAL_SE3
inline PREFIX se3 se3FromSE3(const SE3 & A) {

    se3 a;
    float & wx = a.p[3];
    float & wy = a.p[4];
    float & wz = a.p[5];

    const float cosTheta = (A.r0.x + A.r1.y + A.r2.z - 1)/2.0f;
    const float theta = cosTheta >= 0.9999 ? 0 : acos(cosTheta);
//    std::cout << "<-theta: " << theta << std::endl;
//    std::cout << "<-displacement: " << sqrt(A.r0.w*A.r0.w + A.r1.w*A.r1.w + A.r2.w*A.r2.w) << std::endl;
    if (theta == 0) { wx = wy = wz = 0; a.p[0] = A.r0.w; a.p[1] = A.r1.w; a.p[2] = A.r2.w; return a; }

    wx = (A.r2.y - A.r1.z) * theta / (2*sin(theta));
    wy = (A.r0.z - A.r2.x) * theta / (2*sin(theta));
    wz = (A.r1.x - A.r0.y) * theta / (2*sin(theta));

//    const float Va = (1-cosf(theta)) / (theta*theta);
//    const float Vb = (theta - sinf(theta)) / (theta*theta*theta);

//    const float v11 = 1 +          Vb*(- wz*wz - wy*wy); const float v12 =      Va*-wz + Vb*(wx*wy)          ; const float v13 =      Va* wy + Vb*wx*wz            ;
//    const float v21 =     Va* wz + Vb*(wx*wy)          ; const float v22 = 1 +           Vb*(- wz*wz - wx*wx); const float v23 =      Va*-wx + Vb*wy*wz            ;
//    const float v31 =     Va*-wy + Vb*(wx*wz)          ; const float v32 =      Va* wx + Vb*(wy*wz)          ; const float v33 = 1 +           Vb*(- wy*wy - wx*wx);

//    const float vDet = v11*(v22*v33 - v23*v32) - v12*(v21*v33 - v23*v31) + v13*(v23*v32 - v22*v31);

//    a.p[0] = (1/vDet)*( (v22*v33 - v23*v32)*A.r0.w +
//                        (v13*v32 - v12*v33)*A.r1.w +
//                        (v12*v23 - v13*v22)*A.r2.w);
//    a.p[1] = (1/vDet)*( (v23*v31 - v21*v33)*A.r0.w +
//                        (v11*v33 - v13*v31)*A.r1.w +
//                        (v13*v21 - v11*v23)*A.r2.w);
//    a.p[2] = (1/vDet)*( (v21*v32 - v22*v31)*A.r0.w +
//                        (v12*v31 - v11*v32)*A.r1.w +
//                        (v11*v22 - v12*v21)*A.r2.w);

    const float Va = -0.5;
    const float Vb = (1/(theta*theta))*(1 - (sin(theta)/theta)/(2*(1-cos(theta))/(theta*theta)));

    float & tx = a.p[0];
    float & ty = a.p[1];
    float & tz = a.p[2];

    tx = (1 +          Vb*(- wz*wz - wy*wy))*A.r0.w + (     Va*-wz + Vb*(wx*wy)          )*A.r1.w + (     Va* wy + Vb*(wx*wz)          )*A.r2.w;
    ty = (    Va* wz + Vb*(wx*wy)          )*A.r0.w + (1 +           Vb*(- wz*wz - wx*wx))*A.r1.w + (     Va*-wx + Vb*(wy*wz)          )*A.r2.w;
    tz = (    Va*-wy + Vb*(wx*wz)          )*A.r0.w + (     Va* wx + Vb*(wy*wz)          )*A.r1.w + (1 +           Vb*(- wy*wy - wx*wx))*A.r2.w;

    return a;

}

/*inline PREFIX se3 se3FromSE3Debug(const SE3& A) {

    se3 a;
    float &wx = a.p[3];
    float &wy = a.p[4];
    float &wz = a.p[5];

    const float cosTheta = (A.r0.x + A.r1.y + A.r2.z - 1)/2.0f;
    const float theta = cosTheta == 1 ? 0 : acos(cosTheta);
    std::cout << "theta: " << theta << " = acos(" << (A.r0.x + A.r1.y + A.r2.z - 1)/2.0f << ")" << std::endl;
//    std::cout << "<-displacement: " << sqrt(A.r0.w*A.r0.w + A.r1.w*A.r1.w + A.r2.w*A.r2.w) << std::endl;
    if (theta == 0) { wx = wy = wz = 0; a.p[0] = A.r0.w; a.p[1] = A.r1.w; a.p[2] = A.r2.w; return a; }

    wx = (A.r2.y - A.r1.z) * theta / (2*sin(theta));
    wy = (A.r0.z - A.r2.x) * theta / (2*sin(theta));
    wz = (A.r1.x - A.r0.y) * theta / (2*sin(theta));

//    const float Va = (1-cosf(theta)) / (theta*theta);
//    const float Vb = (theta - sinf(theta)) / (theta*theta*theta);

//    const float v11 = 1 +          Vb*(- wz*wz - wy*wy); const float v12 =      Va*-wz + Vb*(wx*wy)          ; const float v13 =      Va* wy + Vb*wx*wz            ;
//    const float v21 =     Va* wz + Vb*(wx*wy)          ; const float v22 = 1 +           Vb*(- wz*wz - wx*wx); const float v23 =      Va*-wx + Vb*wy*wz            ;
//    const float v31 =     Va*-wy + Vb*(wx*wz)          ; const float v32 =      Va* wx + Vb*(wy*wz)          ; const float v33 = 1 +           Vb*(- wy*wy - wx*wx);

//    const float vDet = v11*(v22*v33 - v23*v32) - v12*(v21*v33 - v23*v31) + v13*(v23*v32 - v22*v31);

//    a.p[0] = (1/vDet)*( (v22*v33 - v23*v32)*A.r0.w +
//                        (v13*v32 - v12*v33)*A.r1.w +
//                        (v12*v23 - v13*v22)*A.r2.w);
//    a.p[1] = (1/vDet)*( (v23*v31 - v21*v33)*A.r0.w +
//                        (v11*v33 - v13*v31)*A.r1.w +
//                        (v13*v21 - v11*v23)*A.r2.w);
//    a.p[2] = (1/vDet)*( (v21*v32 - v22*v31)*A.r0.w +
//                        (v12*v31 - v11*v32)*A.r1.w +
//                        (v11*v22 - v12*v21)*A.r2.w);

    const float Va = -0.5;
    const float Vb = (1/(theta*theta))*(1 - (sin(theta)/theta)/(2*(1-cos(theta))/(theta*theta)));

    float &tx = a.p[0];
    float &ty = a.p[1];
    float &tz = a.p[2];

    tx = (1 +          Vb*(- wz*wz - wy*wy))*A.r0.w + (     Va*-wz + Vb*(wx*wy)          )*A.r1.w + (     Va* wy + Vb*(wx*wz)          )*A.r2.w;
    ty = (    Va* wz + Vb*(wx*wy)          )*A.r0.w + (1 +           Vb*(- wz*wz - wx*wx))*A.r1.w + (     Va*-wx + Vb*(wy*wz)          )*A.r2.w;
    tz = (    Va*-wy + Vb*(wx*wz)          )*A.r0.w + (     Va* wx + Vb*(wy*wz)          )*A.r1.w + (1 +           Vb*(- wy*wy - wx*wx))*A.r2.w;

    return a;

}*/
#else
inline PREFIX se3 se3FromSE3(const SE3 & A) {

    se3 a;

//    const float theta = acos((A.r0.x + A.r1.y + A.r2.z - 1)/2.0f);
//    if (theta == 0) {
//        a.p[3] = a.p[4] = a.p[5] = 0;
//    }
//    else {
//        const float c = 1/(2*sinf(theta));
//        a.p[3] = theta*c*(A.r2.y - A.r1.z);
//        a.p[4] = theta*c*(A.r0.z - A.r2.x);
//        a.p[5] = theta*c*(A.r1.x - A.r0.y);
//    }
    const float x = A.r2.y - A.r1.z;
    const float y = A.r0.z - A.r2.x;
    const float z = A.r1.x - A.r0.y;

    const float r = sqrtf(x*x + y*y + z*z);

    if (r == 0) {
        a.p[3] = a.p[4] = a.p[5] = 0;
    }
    else {
        const float t = A.r0.x + A.r1.y + A.r2.z;
        const float theta = atan2f(r,t-1);
        a.p[3] = x/r*theta;
        a.p[4] = y/r*theta;
        a.p[5] = z/r*theta;
    }

    a.p[0] = A.r0.w;
    a.p[1] = A.r1.w;
    a.p[2] = A.r2.w;

    return a;

}
#endif // REAL_SE3

inline PREFIX float3 translationFromSE3(const SE3 & A) {
    return make_float3(A.r0.w,A.r1.w,A.r2.w);
}

inline PREFIX SE3 SE3Transform(const SE3 & A, const SE3 & B) {
    return SE3(make_float4( A.r0.x*B.r0.x + A.r0.y*B.r1.x + A.r0.z*B.r2.x,
                            A.r0.x*B.r0.y + A.r0.y*B.r1.y + A.r0.z*B.r2.y,
                            A.r0.x*B.r0.z + A.r0.y*B.r1.z + A.r0.z*B.r2.z,
                            A.r0.x*B.r0.w + A.r0.y*B.r1.w + A.r0.z*B.r2.w + A.r0.w),
               make_float4( A.r1.x*B.r0.x + A.r1.y*B.r1.x + A.r1.z*B.r2.x,
                            A.r1.x*B.r0.y + A.r1.y*B.r1.y + A.r1.z*B.r2.y,
                            A.r1.x*B.r0.z + A.r1.y*B.r1.z + A.r1.z*B.r2.z,
                            A.r1.x*B.r0.w + A.r1.y*B.r1.w + A.r1.z*B.r2.w + A.r1.w),
               make_float4( A.r2.x*B.r0.x + A.r2.y*B.r1.x + A.r2.z*B.r2.x,
                            A.r2.x*B.r0.y + A.r2.y*B.r1.y + A.r2.z*B.r2.y,
                            A.r2.x*B.r0.z + A.r2.y*B.r1.z + A.r2.z*B.r2.z,
                            A.r2.x*B.r0.w + A.r2.y*B.r1.w + A.r2.z*B.r2.w + A.r2.w));
}

inline PREFIX float4 SE3Transform(const SE3 & A, const float4 & b) {
    return make_float4(A.r0.x*b.x + A.r0.y*b.y + A.r0.z*b.z + A.r0.w*b.w,
                       A.r1.x*b.x + A.r1.y*b.y + A.r1.z*b.z + A.r1.w*b.w,
                       A.r2.x*b.x + A.r2.y*b.y + A.r2.z*b.z + A.r2.w*b.w,
                       b.w);
}

inline PREFIX float4 SE3Rotate(const SE3 & A, const float4 & b) {
    return make_float4(A.r0.x*b.x + A.r0.y*b.y + A.r0.z*b.z,
                       A.r1.x*b.x + A.r1.y*b.y + A.r1.z*b.z,
                       A.r2.x*b.x + A.r2.y*b.y + A.r2.z*b.z,
                       b.w);
}

inline PREFIX SE3 SE3Rotate(const SE3 & A, const SE3 & B) {
    float3 c1 = SE3Rotate(A,make_float3(B.r0.x,B.r1.x,B.r2.x));
    float3 c2 = SE3Rotate(A,make_float3(B.r0.y,B.r1.y,B.r2.y));
    float3 c3 = SE3Rotate(A,make_float3(B.r0.z,B.r1.z,B.r2.z));
    float3 c4 = SE3Rotate(A,make_float3(B.r0.w,B.r1.w,B.r2.w));
    return SE3(make_float4(c1.x,c2.x,c3.x,c4.x),
               make_float4(c1.y,c2.y,c3.y,c4.y),
               make_float4(c1.z,c2.z,c3.z,c4.z));
}

inline PREFIX float3 SE3Transform(const SE3 & A, const float3 & b) {
    return make_float3(A.r0.x*b.x + A.r0.y*b.y + A.r0.z*b.z + A.r0.w,
                       A.r1.x*b.x + A.r1.y*b.y + A.r1.z*b.z + A.r1.w,
                       A.r2.x*b.x + A.r2.y*b.y + A.r2.z*b.z + A.r2.w);
}

inline PREFIX float3 SE3Rotate(const SE3 & A, const float3 & b) {
    return make_float3(A.r0.x*b.x + A.r0.y*b.y + A.r0.z*b.z,
                       A.r1.x*b.x + A.r1.y*b.y + A.r1.z*b.z,
                       A.r2.x*b.x + A.r2.y*b.y + A.r2.z*b.z);
}

inline PREFIX SE3 SE3Invert(const SE3 & A) {
    return SE3(make_float4(A.r0.x, A.r1.x, A.r2.x, -(A.r0.x*A.r0.w + A.r1.x*A.r1.w + A.r2.x*A.r2.w)),
               make_float4(A.r0.y, A.r1.y, A.r2.y, -(A.r0.y*A.r0.w + A.r1.y*A.r1.w + A.r2.y*A.r2.w)),
               make_float4(A.r0.z, A.r1.z, A.r2.z, -(A.r0.z*A.r0.w + A.r1.z*A.r1.w + A.r2.z*A.r2.w)));
}

inline PREFIX SE3 SE3Interpolate(const SE3 & A, const SE3 & B, const float t) {
    se3 transform = t*se3FromSE3(B*SE3Invert(A));
    return SE3Fromse3(transform)*A;
}

inline PREFIX SE3 SE3Interpolate2(const SE3 & A, const SE3 & B, const float t) {
    const float3 transA = translationFromSE3(A);
    const float3 transB = translationFromSE3(B);
    se3 transform = t*se3FromSE3(SE3Rotate(B,SE3Invert(A)));
    memset(transform.p,0,3*sizeof(float));
    return SE3FromTranslation(make_float3(0.5*(transA.x + transB.x),0.5*(transA.y + transB.y),0.5*(transA.z + transB.z)))*SE3Fromse3(transform)*A;
}

//inline PREFIX SE3 SE3InterpolateDebug(const SE3& A, const SE3& B, const float t) {
//    SE3 relative = B*SE3Invert(A);
//    se3 transform = t*se3FromSE3Debug(B*SE3Invert(A));
//    return SE3Fromse3(transform)*A;
//}

inline PREFIX SE3 operator*(const SE3 & A, const SE3 & B) {
    return SE3Transform(A,B);
}

inline PREFIX float4 operator*(const SE3 & A, const float4 & b) {
    return SE3Transform(A,b);
}

inline PREFIX float3 operator*(const SE3 & A, const float3 & b) {
    return SE3Transform(A,b);
}

inline PREFIX se3 operator*(const se3 & a, const float c) {
    se3 b;
    for (int i=0; i<6; ++i) {
        b.p[i] = c*a.p[i];
    }
    return b;
}

inline PREFIX se3 operator*(const float c, const se3 & a) {
    return a*c;
}


}

#endif // SE3_H
