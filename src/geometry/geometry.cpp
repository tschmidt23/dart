#include "geometry.h"

#include <limits>
#include <math.h>
#include <vector>
#include <map>
#include <iostream>
#include <vector_functions.h>
#include <helper_math.h>
#include <Eigen/Eigen>

namespace dart {

template <typename Real>
Real distancePointEllipseSpecial (const Real e[2], const Real y[2], Real x[2])
{
    Real distance;
    if (y[1] > (Real)0)
    {
        if (y[0] > (Real)0)
        {
            // Bisect to compute the root of F(t) for t >= -e1*e1.
            Real esqr[2] = { e[0]*e[0], e[1]*e[1] };
            Real ey[2] = { e[0]*y[0], e[1]*y[1] };
            Real t0 = -esqr[1] + ey[1];
            Real t1 = -esqr[1] + sqrt(ey[0]*ey[0] + ey[1]*ey[1]);
            Real t = t0;
            const int imax = 2*std::numeric_limits<Real>::max_exponent;
            for (int i = 0; i < imax; ++i)
            {
                t = ((Real)0.5)*(t0 + t1);
                if (t == t0 || t == t1)
                {
                    break;
                }
                Real r[2] = { ey[0]/(t + esqr[0]), ey[1]/(t + esqr[1]) };
                Real f = r[0]*r[0] + r[1]*r[1] - (Real)1;
                if (f > (Real)0)
                {
                    t0 = t;
                }
                else if (f < (Real)0)
                {
                    t1 = t;
                }
                else
                {
                    break;
                }
            }
            x[0] = esqr[0]*y[0]/(t + esqr[0]);
            x[1] = esqr[1]*y[1]/(t + esqr[1]);
            Real d[2] = { x[0] - y[0], x[1] - y[1] };
            distance = sqrt(d[0]*d[0] + d[1]*d[1]);
        }
        else // y0 == 0
        {
            x[0] = (Real)0;
            x[1] = e[1];
            distance = fabs(y[1] - e[1]);
        }
    }
    else // y1 == 0
    {
        Real denom0 = e[0]*e[0] - e[1]*e[1];
        Real e0y0 = e[0]*y[0];
        if (e0y0 < denom0)
        {
            // y0 is inside the subinterval.
            Real x0de0 = e0y0/denom0;
            Real x0de0sqr = x0de0*x0de0;
            x[0] = e[0]*x0de0;
            x[1] = e[1]*sqrt(fabs((Real)1 - x0de0sqr));
            Real d0 = x[0] - y[0];
            distance = sqrt(d0*d0 + x[1]*x[1]);
        }
        else
        {
            // y0 is outside the subinterval. The closest ellipse point has
            // x1 == 0 and is on the domain-boundary interval (x0/e0)^2 = 1.
            x[0] = e[0];
            x[1] = (Real)0;
            distance = fabs(y[0] - e[0]);
        }
    }
    return distance;
}

template <typename Real>
Real distancePointEllipse (const Real e[2], const Real y[2], Real x[2])
{
    // Determine reflections for y to the first quadrant.
    bool reflect[2];
    int i, j;
    for (i = 0; i < 2; ++i)
    {
        reflect[i] = (y[i] < (Real)0);
    }
    // Determine the axis order for decreasing extents.
    int permute[2];
    if (e[0] < e[1])
    {
        permute[0] = 1; permute[1] = 0;
    }
    else
    {
        permute[0] = 0; permute[1] = 1;
    }
    int invpermute[2];
    for (i = 0; i < 2; ++i)
    {
        invpermute[permute[i]] = i;
    }
    Real locE[2], locY[2];
    for (i = 0; i < 2; ++i)
    {
        j = permute[i];
        locE[i] = e[j];
        locY[i] = y[j];
        if (reflect[j])
        {
            locY[i] = -locY[i];
        }
    }
    Real locX[2];
    Real distance = distancePointEllipseSpecial(locE, locY, locX);
    // Restore the axis order and reflections.
    for (i = 0; i < 2; ++i)
    {
        j = invpermute[i];
        if (reflect[j])
        {
            locX[j] = -locX[j];
        }
        x[i] = locX[j];
    }
    return distance;
}

template <typename Real>
Real distancePointEllipsoidSpecial(const Real e[3], const Real y[3], Real x[3])
{
    Real distance;
    if (y[2] > (Real)0)
    {
        if (y[1] > (Real)0)
        {
            if (y[0] > (Real)0)
            {
                // Bisect to compute the root of F(t) for t >= -e2*e2.
                Real esqr[3] = { e[0]*e[0], e[1]*e[1], e[2]*e[2] };
                Real ey[3] = { e[0]*y[0], e[1]*y[1], e[2]*y[2] };
                Real t0 = -esqr[2] + ey[2];
                Real t1 = -esqr[2] + sqrt(ey[0]*ey[0] + ey[1]*ey[1] +
                                          ey[2]*ey[2]);
                Real t = t0;
                const int imax = 2*std::numeric_limits<Real>::max_exponent;
                for (int i = 0; i < imax; ++i)
                {
                    t = ((Real)0.5)*(t0 + t1);
                    if (t == t0 || t == t1)
                    {
                        break;
                    }
                    Real r[3] = { ey[0]/(t + esqr[0]), ey[1]/(t + esqr[1]),
                                  ey[2]/(t + esqr[2]) };
                    Real f = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] - (Real)1;
                    if (f > (Real)0)
                    {
                        t0 = t;
                    }
                    else if (f < (Real)0)
                    {
                        t1 = t;
                    }
                    else
                    {
                        break;
                    }
                }
                x[0] = esqr[0]*y[0]/(t + esqr[0]);
                x[1] = esqr[1]*y[1]/(t + esqr[1]);
                x[2] = esqr[2]*y[2]/(t + esqr[2]);
                Real d[3] = { x[0] - y[0], x[1] - y[1], x[2] - y[2] };
                distance = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
            }
            else // y0 == 0
            {
                x[0] = (Real)0;
                Real etmp[2] = { e[1], e[2] };
                Real ytmp[2] = { y[1], y[2] };
                Real xtmp[2];
                distance = distancePointEllipseSpecial<Real>(etmp, ytmp, xtmp);
                x[1] = xtmp[0];
                x[2] = xtmp[1];
            }
        }
        else // y1 == 0
        {
            x[1] = (Real)0;
            if (y[0] > (Real)0)
            {
                Real etmp[2] = { e[0], e[2] };
                Real ytmp[2] = { y[0], y[2] };
                Real xtmp[2];
                distance = distancePointEllipseSpecial<Real>(etmp, ytmp, xtmp);
                x[0] = xtmp[0];
                x[2] = xtmp[1];
            }
            else // y0 == 0
            {
                x[0] = (Real)0;
                x[2] = e[2];
                distance = fabs(y[2] - e[2]);
            }
        }
    }
    else // y2 == 0
    {
        Real denom[2] = { e[0]*e[0] - e[2]*e[2], e[1]*e[1] - e[2]*e[2] };
        Real ey[2] = { e[0]*y[0], e[1]*y[1] };
        if (ey[0] < denom[0] && ey[1] < denom[1])
        {
            // (y0,y1) is inside the axis-aligned bounding rectangle of the
            // subellipse. This intermediate test is designed to guard
            // against the division by zero when e0 == e2 or e1 == e2.
            Real xde[2] = { ey[0]/denom[0], ey[1]/denom[1] };
            Real xdesqr[2] = { xde[0]*xde[0], xde[1]*xde[1] };
            Real discr = (Real)1 - xdesqr[0] - xdesqr[1];
            if (discr > (Real)0)
            {
                // (y0,y1) is inside the subellipse. The closest ellipsoid
                // point has x2 > 0.
                x[0] = e[0]*xde[0];
                x[1] = e[1]*xde[1];
                x[2] = e[2]*sqrt(discr);
                Real d[2] = { x[0] - y[0], x[1] - y[1] };
                distance = sqrt(d[0]*d[0] + d[1]*d[1] + x[2]*x[2]);
            }
            else
            {
                // (y0,y1) is outside the subellipse. The closest ellipsoid
                // point has x2 == 0 and is on the domain-boundary ellipse
                // (x0/e0)^2 + (x1/e1)^2 = 1.
                x[2] = (Real)0;
                distance = distancePointEllipseSpecial<Real>(e, y, x);
            }
        }
        else
        {
            // (y0,y1) is outside the subellipse. The closest ellipsoid
            // point has x2 == 0 and is on the domain-boundary ellipse
            // (x0/e0)^2 + (x1/e1)^2 = 1.
            x[2] = (Real)0;
            distance = distancePointEllipseSpecial<Real>(e, y, x);
        }
    }
    return distance;
}

template <typename Real>
Real distancePointEllipsoid(const Real e[3], const Real y[3], Real x[3])
{
    // Determine reflections for y to the first octant.
    bool reflect[3];
    int i, j;
    for (i = 0; i < 3; ++i)
    {
        reflect[i] = (y[i] < (Real)0);
    }
    // Determine the axis order for decreasing extents.
    int permute[3];
    if (e[0] < e[1])
    {
        if (e[2] < e[0])
        {
            permute[0] = 1; permute[1] = 0; permute[2] = 2;
        }
        else if (e[2] < e[1])
        {
            permute[0] = 1; permute[1] = 2; permute[2] = 0;
        }
        else
        {
            permute[0] = 2; permute[1] = 1; permute[2] = 0;
        }
    }
    else
    {
        if (e[2] < e[1])
        {
            permute[0] = 0; permute[1] = 1; permute[2] = 2;
        }
        else if (e[2] < e[0])
        {
            permute[0] = 0; permute[1] = 2; permute[2] = 1;
        }
        else
        {
            permute[0] = 2; permute[1] = 0; permute[2] = 1;
        }
    }

    int invpermute[3];
    for (i = 0; i < 3; ++i)
    {
        invpermute[permute[i]] = i;
    }
    Real locE[3], locY[3];
    for (i = 0; i < 3; ++i)
    {
        j = permute[i];
        locE[i] = e[j];
        locY[i] = y[j];
        if (reflect[j])
        {
            locY[i] = -locY[i];
        }
    }
    Real locX[3];
    Real distance = distancePointEllipsoidSpecial(locE, locY, locX);

    // Restore the axis order and reflections.
    for (i = 0; i < 3; ++i)
    {
        j = invpermute[i];
        if (reflect[j])
        {
            locX[j] = -locX[j];
        }
        x[i] = locX[j];
    }
    return distance;
}

float distancePointTriangle(const float3 P, const float3 A, const float3 B, const float3 C) {
    float3 D;
    return distancePointTriangle(P,A,B,C,D);
}

float distancePointTriangle(const float3 P, const float3 A, const float3 B, const float3 C, float3 & point) {

    float3 E0 = A-B;
    float3 E1 = C-B;

    float3 D = B-P;
    float a = dot(E0,E0);
    float b = dot(E0,E1);
    float c = dot(E1,E1);
    float d = dot(E0,D);
    float e = dot(E1,D);
    float f = dot(D,D);

    float det = a*c-b*b;
    float s = b*e - c*d;
    float t = b*d - a*e;

    int region;
    if ( s+t <= det) {
        if ( s < 0 ) {
            if ( t < 0 ) {
                region = 4;
            } else {
                region = 3;
            }
        } else if ( t < 0 ) {
            region = 5;
        } else {
            region = 0;
        }
    } else {
        if ( s < 0 ) {
            region = 2;
        } else if ( t < 0) {
            region = 6;
        } else {
            region = 1;
        }
    }

//    std::cout << region << std::endl;

    switch (region) {
        case 0:
            {
                float invDet = 1/det;
                s*= invDet;
                t*= invDet;
            }
            break;
        case 1:
            {
                float numer = c + e - b - d;
                if (numer <= 0) {
                    s = 0;
                } else {
                    float denom = a - 2*b + c;
                    s = ( numer >= denom ? 1 : numer/denom );
                }
                t = 1-s;
            }
            break;
        case 2:
            {
                float tmp0 = b+d;
                float tmp1 = c+e;
                if ( tmp1 > tmp0 ) { // min on edge s+1=1
                    float numer = tmp1 - tmp0;
                    float denom = a - 2*b + c;
                    s = ( numer >= denom ? 1 : numer/denom );
                    t = 1-s;
                } else { // min on edge s=0
                    s = 0;
                    t = ( tmp1 <= 0 ? 1 : ( e >= 0 ? 0 : -e/c ) );
                }
            }
            break;
        case 3:
            s = 0;
            t = ( e >= 0 ? 0 :
                           ( -e >= c ? 1 : -e/c ) );
            break;
        case 4:
            if ( d < 0 ) { // min on edge t=0
                t = 0;
                s = ( d >= 0 ? 0 :
                               ( -d >= a ? 1 : -d/a ) );
            } else { // min on edge s = 0
                s = 0;
                t = ( e >= 0 ? 0 :
                               ( -e >= c ? 1 : -e/c ) );
            }
            break;
        case 5:
            t = 0;
            s = ( d >= 0 ? 0 :
                           ( -d >= a ? 1 : -d/a ) );
            break;
        case 6:
            {
                float tmp0 = a+d;
                float tmp1 = b+e;
                if (tmp0 > tmp1) { // min on edge s+1=1
                    float numer = c + e - b - d;
                    float denom = a -2*b + c;
                    s = ( numer >= denom ? 1 : numer/denom );
                    t = 1-s;
                } else { // min on edge t=1
                    t = 0;
                    s = ( tmp0 <= 0 ? 1 : ( d >= 0 ? 0 : -d/a ));
                }
            }
            break;
    }
    point = B + s*E0 + t*E1;
    float3 v = point-P;
    return dot(v,v);
}

template <typename Real>
Real distancePointLineSegment2D(const typename VectorTypeTemplate<Real>::type2 p,
                                const typename VectorTypeTemplate<Real>::type2 a,
                                const typename VectorTypeTemplate<Real>::type2 b ) {

    typedef typename VectorTypeTemplate<Real>::type2 T2;

    const T2 v = b - a;
    const T2 w = p - a;

    float c1 = dot(w,v);
    if (c1 <= 0) {
        return length(w);
    }

    float c2 = dot(v,v);
    if (c2 <= c1) {
        return length(p - b);
    }

    float t = c1 / c2;
    T2 closest = a + t*v;
    return length(p - closest);

}

template <typename Real>
Real signedDistancePointLineSegment2D(const typename VectorTypeTemplate<Real>::type2 p,
                                      const typename VectorTypeTemplate<Real>::type2 a,
                                      const typename VectorTypeTemplate<Real>::type2 b ) {

    typedef typename VectorTypeTemplate<Real>::type2 T2;

    const T2 v = b - a;
    const T2 w = p - a;
    T2 n;
    n.x = v.y;
    n.y = -v.x;

    int sign = dot(n,w) > 0 ? 1 : -1;

    float c1 = dot(w,v);
    if (c1 <= 0) {
        return sign*length(w);
    }

    float c2 = dot(v,v);
    if (c2 <= c1) {
        return sign*length(p - b);
    }

    float t = c1 / c2;
    T2 closest = a + t*v;
    return sign*length(p - closest);

}


template float distancePointLineSegment2D<float>(const float2, const float2, const float2);

template float signedDistancePointLineSegment2D<float>(const float2, const float2, const float2);


template <typename Real, typename Derived>
inline void rotationMatrixFromRodrigues(const Real w[3], Eigen::MatrixBase<Derived> const & R) {

    Real theta = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    typedef typename Derived::Scalar Scalar;

    if (theta == 0) {
        const_cast< Eigen::MatrixBase<Derived>& >(R) = Eigen::Matrix<Scalar,3,3>::Identity(3,3);
        return;
    }

    Real rx = w[0] / theta;
    Real ry = w[1] / theta;
    Real rz = w[2] / theta;

    Eigen::Matrix<Scalar,3,3> H;
    H << 0, -rz, ry, rz, 0, -rx, -ry, rx, 0;

    Eigen::Matrix<Scalar,3,3> H2;
    H2 << rx*rx-1, rx*ry, rx*rz, rx*ry, ry*ry-1, ry*rz, rx*rz, ry*rz, rz*rz-1;

    const_cast< Eigen::MatrixBase<Derived> &>(R) = Eigen::Matrix<Scalar,3,3>::Identity(3,3) + sin(theta)*H + (1-cos(theta))*H2;

}

template <typename Real, typename Derived>
inline void rodriguesFromRotationMatrix(Real w[3], Eigen::MatrixBase<Derived> const & R) {

    Real x = R(2,1) - R(1,2);
    Real y = R(0,2) - R(2,0);
    Real z = R(1,0) - R(0,1);

    Real r = sqrt(x*x + y*y + z*z);

    if (r == 0) {
        memset(w,0,3*sizeof(Real));
        return;
    }

    Real t = R(0,0) + R(1,1) + R(2,2);

    Real theta = atan2(r,t-1);

    w[0] = x/r*theta;
    w[1] = y/r*theta;
    w[2] = z/r*theta;

}

template <typename Real>
void rotationMatrixJacobianFromRodrigues(const Real w[3], Eigen::Matrix<Real,9,3> & J) {

    Real theta = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]); 

    if (theta == 0) {
        J <<    0, 0, 0, 0, 0, -1, 0, 1, 0,
                0, 0, 1, 0, 0, 0, -1, 0, 0,
                0, -1, 0, 1, 0, 0, 0, 0, 0;
        return;
    }

    Real rx = w[0] / theta;
    Real ry = w[1] / theta;
    Real rz = w[2] / theta;

    Eigen::Matrix<Real,3,3> H;
    H << 0, -rz, ry, rz, 0, -rx, -ry, rx, 0;

    Eigen::Matrix<Real,3,3> H2;
    H2 << rx*rx-1, rx*ry, rx*rz, rx*ry, ry*ry-1, ry*rz, rx*rz, ry*rz, rz*rz-1;

    Eigen::Matrix<Real,3,3> dH_drx, dH_dry, dH_drz;
    dH_drx << 0, 0, 0, 0, 0, -1, 0, 1, 0;
    dH_dry << 0, 0, 1, 0, 0, 0, -1, 0, 0;
    dH_drz << 0, -1, 0, 1, 0, 0, 0, 0, 0;

    Eigen::Matrix<Real,3,3> dH2_drx, dH2_dry, dH2_drz;
    dH2_drx << 2*rx, ry, rz, ry, 0, 0, rz, 0, 0;
    dH2_dry << 0, rx, 0, rx, 2*ry, rz, 0, rz, 0;
    dH2_drz << 0, 0, rx, 0, 0, ry, rx, ry, 2*rz;

    J.block(0,0,3,3) = H*rx*(cos(theta) - sin(theta)/theta) + dH_drx*sin(theta)/theta +
            H*H*rx*(sin(theta)-2*(1-cos(theta))/theta) + (1-cos(theta))/theta*(dH2_drx - 2*rx*Eigen::Matrix<Real,3,3>::Identity(3,3));
    J.block(3,0,3,3) = H*ry*(cos(theta) - sin(theta)/theta) + dH_dry*sin(theta)/theta +
            H*H*ry*(sin(theta)-2*(1-cos(theta))/theta) + (1-cos(theta))/theta*(dH2_dry - 2*ry*Eigen::Matrix<Real,3,3>::Identity(3,3));
    J.block(6,0,3,3) = H*rz*(cos(theta) - sin(theta)/theta) + dH_drz*sin(theta)/theta +
            H*H*rz*(sin(theta)-2*(1-cos(theta))/theta) + (1-cos(theta))/theta*(dH2_drz - 2*rz*Eigen::Matrix<Real,3,3>::Identity(3,3));

}

template <typename Real>
void aabbEllipsoid(const Real e[3], const Real c[3], const Real w[3], Real o[3], Real s[3]) {

    Eigen::Matrix<Real,3,3> R;
    rotationMatrixFromRodrigues<Real>(w,R);

    Real deltax = sqrt(e[0]*e[0]*R(0,0)*R(0,0) + e[1]*e[1]*R(0,1)*R(0,1) + e[2]*e[2]*R(0,2)*R(0,2));
    Real deltay = sqrt(e[0]*e[0]*R(1,0)*R(1,0) + e[1]*e[1]*R(1,1)*R(1,1) + e[2]*e[2]*R(1,2)*R(1,2));
    Real deltaz = sqrt(e[0]*e[0]*R(2,0)*R(2,0) + e[1]*e[1]*R(2,1)*R(2,1) + e[2]*e[2]*R(2,2)*R(2,2));

    o[0] = c[0] - deltax;
    o[1] = c[1] - deltay;
    o[2] = c[2] - deltaz;

    s[0] = 2*deltax;
    s[1] = 2*deltay;
    s[2] = 2*deltaz;
}

template <typename Real>
void aabbEllipticCylinder(const Real e[2], const Real h, const Real c[3], const Real w[3], Real o[3], Real s[3]) {

    Eigen::Matrix<Real,3,3> R;
    rotationMatrixFromRodrigues<Real>(w,R);

    Eigen::Matrix<Real,3,1> u, v, c2;
    u << e[0], 0, 0;
    v << 0, e[1], 0;
    c2 << 0, 0, h;

    u = R*u;
    v = R*v;
    c2 = R*c2;

    Real rx = sqrt(u(0)*u(0) + v(0)*v(0));
    Real ry = sqrt(u(1)*u(1) + v(1)*v(1));
    Real rz = sqrt(u(2)*u(2) + v(2)*v(2));

    o[0] = std::min(c[0]-rx, c[0]+c2(0)-rx);
    o[1] = std::min(c[1]-ry, c[1]+c2(1)-ry);
    o[2] = std::min(c[2]-rz, c[2]+c2(2)-rz);

    s[0] = std::max(c[0]+rx, c[0]+c2(0)+rx) - o[0];
    s[1] = std::max(c[1]+ry, c[1]+c2(1)+ry) - o[1];
    s[2] = std::max(c[2]+rz, c[2]+c2(2)+rz) - o[2];
}

template <typename Real>
void aabbRectangularPrism(const Real l[3], const Real c[3], const Real w[3], Real o[3], Real s[3]) {

    Eigen::Matrix<Real,3,3> R;
    rotationMatrixFromRodrigues<Real>(w,R);

    Eigen::Matrix<Real,3,1> corners[8];
    corners[0] << -l[0], -l[1], -l[2];
    corners[1] << -l[0], -l[1],  l[2];
    corners[2] << -l[0],  l[1], -l[2];
    corners[3] << -l[0],  l[1],  l[2];
    corners[4] <<  l[0], -l[1], -l[2];
    corners[5] <<  l[0], -l[1],  l[2];
    corners[6] <<  l[0],  l[1], -l[2];
    corners[7] <<  l[0],  l[1],  l[2];

    for (int i=0; i<8; i++)
        corners[i] = R*corners[i];

    o[0] = s[0] = corners[0](0);
    o[1] = s[1] = corners[0](1);
    o[2] = s[2] = corners[0](2);

    for (int i=1; i<8; i++) {
        o[0] = std::min(o[0],corners[i](0));
        o[1] = std::min(o[1],corners[i](1));
        o[2] = std::min(o[2],corners[i](2));
        s[0] = std::max(s[0],corners[i](0));
        s[1] = std::max(s[1],corners[i](1));
        s[2] = std::max(s[2],corners[i](2));
    }

    s[0] -= o[0];
    s[1] -= o[1];
    s[2] -= o[2];

    o[0] += c[0];
    o[1] += c[1];
    o[2] += c[2];
}

void generateUnitIcosphere(float3 * &verts, int3 * & indxs, int & nverts, int & nfaces, const int splits) {

    std::vector<float3> * vertVec = new std::vector<float3>();
    std::vector<int3> * faceVec = new std::vector<int3>();

    // generate initial 12 vertices
    float t = (1.0f + sqrtf(5.0f)) / 2.0f;

    vertVec->push_back(normalize(make_float3(-1, t, 0)));
    vertVec->push_back(normalize(make_float3( 1, t, 0)));
    vertVec->push_back(normalize(make_float3(-1,-1, 0)));
    vertVec->push_back(normalize(make_float3( 1,-t, 0)));

    vertVec->push_back(normalize(make_float3( 0,-1, t)));
    vertVec->push_back(normalize(make_float3( 0, 1, t)));
    vertVec->push_back(normalize(make_float3( 0,-1,-t)));
    vertVec->push_back(normalize(make_float3( 0, 1,-t)));

    vertVec->push_back(normalize(make_float3( t, 0,-1)));
    vertVec->push_back(normalize(make_float3( t, 0, 1)));
    vertVec->push_back(normalize(make_float3(-t, 0,-1)));
    vertVec->push_back(normalize(make_float3(-t, 0, 1)));

    // generate intitial 20 faces
    faceVec->push_back(make_int3( 0,11, 5));
    faceVec->push_back(make_int3( 0, 5, 1));
    faceVec->push_back(make_int3( 0, 1, 7));
    faceVec->push_back(make_int3( 0, 7,10));
    faceVec->push_back(make_int3( 0,10,11));

    faceVec->push_back(make_int3( 1, 5, 9));
    faceVec->push_back(make_int3( 5,11, 4));
    faceVec->push_back(make_int3(11,10, 2));
    faceVec->push_back(make_int3(10, 7, 6));
    faceVec->push_back(make_int3( 7, 1, 8));

    faceVec->push_back(make_int3( 3, 9, 4));
    faceVec->push_back(make_int3( 3, 4, 2));
    faceVec->push_back(make_int3( 3, 2, 6));
    faceVec->push_back(make_int3( 3, 6, 8));
    faceVec->push_back(make_int3( 3, 8, 9));

    faceVec->push_back(make_int3( 4, 9, 5));
    faceVec->push_back(make_int3( 2, 4,11));
    faceVec->push_back(make_int3( 6, 2,10));
    faceVec->push_back(make_int3( 8, 6, 7));
    faceVec->push_back(make_int3( 9, 8, 1));

    // map of already split vertices
    std::map<int64_t,int> split_verts;

    for (int i=0; i<splits; i++) {
        std::vector<int3>* new_faces = new std::vector<int3>();

        for (unsigned int f = 0; f < faceVec->size(); f++) {

            const int v1 = faceVec->at(f).x;
            const int v2 = faceVec->at(f).y;
            const int v3 = faceVec->at(f).z;
            int64_t key;
            std::map<int64_t,int>::iterator it;
            int p12, p23, p31;

            // edge 12
            key = (v1 < v2) ? (((int64_t)v1 << 32) | v2) : (((int64_t)v2 << 32) | v1);
            it = split_verts.find(key);
            if (it != split_verts.end()) { // check if we've already split this edge
                p12 = it->second;
            }
            else {
                p12 = vertVec->size();
                split_verts[key] = p12;
                vertVec->push_back(normalize(vertVec->at(v1) + vertVec->at(v2)));
            }

            // edge 23
            key = (v2 < v3) ? (((int64_t)v2 << 32) | v3) : (((int64_t)v3 << 32) | v2);
            it = split_verts.find(key);
            if (it != split_verts.end()) { // check if we've already split this edge
                p23 = it->second;
            }
            else {
                p23 = vertVec->size();
                split_verts[key] = p23;
                vertVec->push_back(normalize(vertVec->at(v2) + vertVec->at(v3)));
            }

            // edge 31
            key = (v3 < v1) ? (((int64_t)v3 << 32) | v1) : (((int64_t)v1 << 32) | v3);
            it = split_verts.find(key);
            if (it != split_verts.end()) { // check if we've already split this edge
                p31 = it->second;
            }
            else {
                p31 = vertVec->size();
                split_verts[key] = p31;
                vertVec->push_back(normalize(vertVec->at(v3) + vertVec->at(v1)));
            }

            // add new faces
            new_faces->push_back(make_int3(v1,p12,p31));
            new_faces->push_back(make_int3(v2,p23,p12));
            new_faces->push_back(make_int3(v3,p31,p23));
            new_faces->push_back(make_int3(p12,p23,p31));

        }

        delete faceVec;
        faceVec = new_faces;
    }

    // convert STL containers to raw arrays
    verts = new float3[vertVec->size()];
    memcpy(verts,vertVec->data(),vertVec->size()*sizeof(float3));
    nverts = vertVec->size();

    indxs = new int3[faceVec->size()];
    memcpy(indxs,faceVec->data(),faceVec->size()*sizeof(int3));
    nfaces = faceVec->size();

    // memory cleanup
    delete vertVec;
    delete faceVec;

}

// generate functions from template for the linker (just float or double for now)
template double distancePointEllipse<double>(const double[], const double[], double[]);
template float distancePointEllipse<float>(const float[], const float[], float[]);

template double distancePointEllipsoid<double>(const double[], const double[], double[]);
template float distancePointEllipsoid<float>(const float[], const float[], float[]);

template void aabbEllipsoid<double>(const double e[3], const double c[3], const double w[3], double o[3], double s[3]);
template void aabbEllipsoid<float>(const float e[3], const float c[3], const float w[3], float o[3], float s[3]);

template void aabbEllipticCylinder<double>(const double e[2], const double h, const double c[3], const double w[3], double o[3], double s[3]);
template void aabbEllipticCylinder<float>(const float e[2], const float h, const float c[3], const float w[3], float o[3], float s[3]);

template void aabbRectangularPrism<double>(const double l[3], const double c[3], const double w[3], double o[3], double s[3]);
template void aabbRectangularPrism<float>(const float l[3], const float c[3], const float w[3], float o[3], float s[3]);

}
