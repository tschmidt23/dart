#ifndef GEOMETRY_H
#define GEOMETRY_H

//#include <Eigen/Eigen>
#include <vector_types.h>
#include "util/vector_type_template.h"

namespace dart {

//----------------------------------------------------------------------------
//                      Distance to ellipsoid functions
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// The ellipse is (x0/e0)^2 + (x1/e1)^2 = 1 with e0 >= e1. The query point is
// (y0,y1) with y0 >= 0 and y1 >= 0. The function returns the distance from
// the query point to the ellipse. It also computes the ellipse point (x0,x1)
// in the first quadrant that is closest to (y0,y1).
//----------------------------------------------------------------------------
template <typename Real>
Real distancePointEllipseSpecial(const Real e[2], const Real y[2], Real x[2]);

//----------------------------------------------------------------------------
// The ellipse is (x0/e0)^2 + (x1/e1)^2 = 1. The query point is (y0,y1).
// The function returns the distance from the query point to the ellipse.
// It also computes the ellipse point (x0,x1) that is closest to (y0,y1).
//----------------------------------------------------------------------------
template <typename Real>
Real distancePointEllipse(const Real e[2], const Real y[2], Real x[2]);

//----------------------------------------------------------------------------
// The ellipsoid is (x0/e0)^2 + (x1/e1)^2 + (x2/e2)^2 = 1 with e0 >= e1 >= e2.
// The query point is (y0,y1,y2) with y0 >= 0, y1 >= 0, and y2 >= 0. The
// function returns the distance from the query point to the ellipsoid. It
// also computes the ellipsoid point (x0,x1,x2) in the first octant that is
// closest to (y0,y1,y2).
//----------------------------------------------------------------------------
template <typename Real>
Real distancePointEllipsoidSpecial(const Real e[3], const Real y[3], Real x[3]);

//----------------------------------------------------------------------------
// The ellipsoid is (x0/e0)^2 + (x1/e1)^2 + (x2/e2)^2 = 1. The query point is
// (y0,y1,y2). The function returns the distance from the query point to the
// ellipsoid. It also computes the ellipsoid point (x0,x1,x2) that is
// closest to (y0,y1,y2).
//----------------------------------------------------------------------------
template <typename Real>
Real distancePointEllipsoid(const Real e[3], const Real y[3], Real x[3]);


float distancePointTriangle(const float3 p, const float3 A, const float3 B, const float3 C);

float distancePointTriangle(const float3 p, const float3 A, const float3 B, const float3 C, float & s, float & t);

float distancePointTriangle(const float3 p, const float3 A, const float3 B, const float3 C, float3 & D);

template <typename Real>
Real distancePointLineSegment2D(const typename VectorTypeTemplate<Real>::type2 p,
                                 const typename VectorTypeTemplate<Real>::type2 a,
                                 const typename VectorTypeTemplate<Real>::type2 b );

template <typename Real>
Real signedDistancePointLineSegment2D(const typename VectorTypeTemplate<Real>::type2 p,
                                      const typename VectorTypeTemplate<Real>::type2 a,
                                      const typename VectorTypeTemplate<Real>::type2 b );


//----------------------------------------------------------------------------
//                  Rodrigues (axis-angle) matrix functions
//----------------------------------------------------------------------------
//template <typename Real, typename Derived>
//inline void rotationMatrixFromRodrigues(const Real w[3], Eigen::MatrixBase<Derived> const &R);

//template <typename Real, typename Derived>
//inline void rodriguesFromRotationMatrix(Real w[3], Eigen::MatrixBase<Derived> const &R);

//----------------------------------------------------------------------------
// This calculates the derivative of a rotation matrix derived from the given
// set of Rodrigues parameters, relative to those three parameters. The
// computed Jacobian is stacked such that the top 3x3 matrix is the derivative
// of the rotation w.r.t. wx, the next 3x3 matrix is the derivative w.r.t. wy,
// and the bottom 3x3 is the derivative w.r.t. wz.
//----------------------------------------------------------------------------
//template <typename Real>
//void rotationMatrixJacobianFromRodrigues(const Real w[3], Eigen::Matrix<Real,9,3> &J);

//----------------------------------------------------------------------------
//              Axis-aligned bounding box (AABB) functions
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// This function calculates the axis aligned bounding box of an ellipse
// specified by ((x0-c0)/e0)^2 + ((x1-c1)/e1)^2 + ((x2-c2)/e2)^2 <= 1, which
// has also undergone a rotation specified by Rodrigues parameters (wx,wy,wz).
// The bounding box has origin (o0,o1,o2) and size (s0,s1,s2).
//----------------------------------------------------------------------------
template <typename Real>
void aabbEllipsoid(const Real e[3], const Real c[3], const Real w[3], Real o[3], Real s[3]);

//----------------------------------------------------------------------------
// This function calculates the axis aligned bounding box of an elliptic
// cylinder with end caps specified by ((x0-c0)/e0)^2 + ((x1-c1)/e1)^2 <= 1.
// The end caps are centered at (c0,c1,0) and (c0,c1,h), and the cylinder has
// also undergone a rotation specified by Rodrigues parameters (wx,wy,wx). It
// is assumed the rotation is about the point (c0,c1,0), i.e. the center of
// the lower end cap.
//----------------------------------------------------------------------------
template <typename Real>
void aabbEllipticCylinder(const Real e[2], const Real h, const Real c[3], const Real w[3], Real o[3], Real s[3]);

template <typename Real>
void aabbRectangularPrism(const Real l[3], const Real c[3], const Real w[3], Real o[3], Real s[3]);

//----------------------------------------------------------------------------
//                      Geometry-generating functions
//----------------------------------------------------------------------------
void generateUnitIcosphere(float3 * & verts, int3 * & indxs, int & nverts, int & nfaces, const int splits);

}

#endif // GEOMETRY_H
