#ifndef GRID_3D_H
#define GRID_3D_H

#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

#include "util/vector_type_template.h"
#include "util/prefix.h"

namespace dart {

template <typename T>
class Grid3D {

private:
    typedef typename VectorTypeTemplate<T>::type3 T3;

public:

    // ===== constructors / destructor =====
    PREFIX Grid3D() : dim(make_uint3(0,0,0)), offset(make_float3(0,0,0)), resolution(0), data(0) {}
    PREFIX Grid3D(uint3 dim) : dim(dim), offset(make_float3(0,0,0)), resolution(0), data(new T[dim.x*dim.y*dim.z]) {}
    PREFIX Grid3D(uint3 dim, float3 offset, float resolution) : dim(dim), offset(offset), resolution(resolution), data(new T[dim.x*dim.y*dim.z]) {}
    PREFIX Grid3D(const Grid3D & grid) : dim(grid.dim), offset(grid.offset), resolution(grid.resolution), data(new T[grid.dim.x*grid.dim.y*grid.dim.z]) {
        memcpy(data, grid.data, dim.x*dim.y*dim.z*sizeof(T));
    }
    PREFIX ~Grid3D() { delete [] data; }

    // ===== inline member functions =====
    inline PREFIX Grid3D<T> & operator= (const Grid3D<float> & grid) {
        if (this == &grid) {
            return *this;
        }
        delete [] data;
        dim = grid.dim;
        offset = grid.offset;
        resolution = grid.resolution;
        data = new T[dim.x*dim.y*dim.z];
        memcpy(grid.data,data,dim.x*dim.y*dim.z*sizeof(T));
        return *this;
    }

    inline PREFIX float3 getGridCoords(const float3 & pWorld) const {
        return (pWorld - offset)/resolution;
    }

    inline PREFIX float3 getWorldCoords(const float3 & pGrid) const {
        return resolution*pGrid + offset;
    }

    inline PREFIX bool isInBounds(const float3 & pGrid) const {
        return (pGrid.x > 0 && pGrid.x < dim.x &&
                pGrid.y > 0 && pGrid.y < dim.y &&
                pGrid.z > 0 && pGrid.z < dim.z);
    }

    inline PREFIX bool isInBoundsInterp(const float3 & pGrid) const {
        return (pGrid.x > 0.50001 && pGrid.x < dim.x - 0.50001 &&
                pGrid.y > 0.50001 && pGrid.y < dim.y - 0.50001 &&
                pGrid.z > 0.50001 && pGrid.z < dim.z - 0.50001);
    }

    inline PREFIX bool isInBoundsGradient(const float3 & pGrid) const {
        return (pGrid.x > 1.50001 && pGrid.x < dim.x - 1.50001 &&
                pGrid.y > 1.50001 && pGrid.y < dim.y - 1.50001 &&
                pGrid.z > 1.50001 && pGrid.z < dim.z - 1.50001);
    }

    // TODO: figure out what these values should really be
    inline PREFIX bool isInBoundsGradientInterp(const float3 & pGrid) const {
        return (pGrid.x > 2.50001 && pGrid.x < dim.x - 2.50001 &&
                pGrid.y > 2.50001 && pGrid.y < dim.y - 2.50001 &&
                pGrid.z > 2.50001 && pGrid.z < dim.z - 2.50001);
    }

    inline PREFIX T & getValue(const int3 & v) const {
        return data[v.x + dim.x*(v.y + dim.y*v.z)];
    }

    inline PREFIX T getValueInterpolated(const float3 & pGrid) const {

        const int x0 = (int)(pGrid.x - 0.5); const float fx = (pGrid.x - 0.5) - x0;
        const int y0 = (int)(pGrid.y - 0.5); const float fy = (pGrid.y - 0.5) - y0;
        const int z0 = (int)(pGrid.z - 0.5); const float fz = (pGrid.z - 0.5) - z0;

        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const int z1 = z0 + 1;

        if ( !(x0 >= 0 && x1 < dim.x && y0 >= 0 && y1 < dim.y && z0 >=0 && z1 < dim.z) ) {
            printf("nope (%d)\n",isInBoundsInterp(pGrid));
        }

        const float dx00 = lerp( getValue(make_int3(x0,y0,z0)), getValue(make_int3(x1,y0,z0)), fx);
        const float dx01 = lerp( getValue(make_int3(x0,y0,z1)), getValue(make_int3(x1,y0,z1)), fx);
        const float dx10 = lerp( getValue(make_int3(x0,y1,z0)), getValue(make_int3(x1,y1,z0)), fx);
        const float dx11 = lerp( getValue(make_int3(x0,y1,z1)), getValue(make_int3(x1,y1,z1)), fx);

        const float dxy0 = lerp( dx00, dx10, fy );
        const float dxy1 = lerp( dx01, dx11, fy );

        const float dxyz = lerp( dxy0, dxy1, fz );

        return dxyz;

    }

    inline PREFIX T getValueInterpolated_(const float3 & pGrid) const {

        const int x0 = (int)(pGrid.x - 0.5); const float fx = (pGrid.x - 0.5) - x0;
        const int y0 = (int)(pGrid.y - 0.5); const float fy = (pGrid.y - 0.5) - y0;
        const int z0 = (int)(pGrid.z - 0.5); const float fz = (pGrid.z - 0.5) - z0;;

        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const int z1 = z0 + 1;

        if ( !(x0 >= 0 && x1 < dim.x && y0 >= 0 && y1 < dim.y && z0 >=0 && z1 < dim.z) ) {
            printf("nope_ (%d)\n",isInBoundsInterp(pGrid));
        }

        const float dx00 = lerp( getValue(make_int3(x0,y0,z0)), getValue(make_int3(x1,y0,z0)), fx);
        const float dx01 = lerp( getValue(make_int3(x0,y0,z1)), getValue(make_int3(x1,y0,z1)), fx);
        const float dx10 = lerp( getValue(make_int3(x0,y1,z0)), getValue(make_int3(x1,y1,z0)), fx);
        const float dx11 = lerp( getValue(make_int3(x0,y1,z1)), getValue(make_int3(x1,y1,z1)), fx);

        const float dxy0 = lerp( dx00, dx10, fy );
        const float dxy1 = lerp( dx01, dx11, fy );

        const float dxyz = lerp( dxy0, dxy1, fz );

        return dxyz;

    }

    inline PREFIX T getValueInterpolated1(const float3 & pGrid) const {

        const int x0 = (int)(pGrid.x - 0.5); const float fx = (pGrid.x - 0.5) - x0;
        const int y0 = (int)(pGrid.y - 0.5); const float fy = (pGrid.y - 0.5) - y0;
        const int z0 = (int)(pGrid.z - 0.5); const float fz = (pGrid.z - 0.5) - z0;

        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const int z1 = z0 + 1;

        if ( !(x0 >= 0 && x1 < dim.x && y0 >= 0 && y1 < dim.y && z0 >=0 && z1 < dim.z) ) {
            printf("nope1 (%d)\n",isInBoundsInterp(pGrid));
        }

        const float dx00 = lerp( getValue(make_int3(x0,y0,z0)), getValue(make_int3(x1,y0,z0)), fx);
        const float dx01 = lerp( getValue(make_int3(x0,y0,z1)), getValue(make_int3(x1,y0,z1)), fx);
        const float dx10 = lerp( getValue(make_int3(x0,y1,z0)), getValue(make_int3(x1,y1,z0)), fx);
        const float dx11 = lerp( getValue(make_int3(x0,y1,z1)), getValue(make_int3(x1,y1,z1)), fx);

        const float dxy0 = lerp( dx00, dx10, fy );
        const float dxy1 = lerp( dx01, dx11, fy );

        const float dxyz = lerp( dxy0, dxy1, fz );

        return dxyz;

    }

    inline PREFIX T getValueInterpolated2(const float3 & pGrid) const {

        const int x0 = (int)(pGrid.x - 0.5); const float fx = (pGrid.x - 0.5) - x0;
        const int y0 = (int)(pGrid.y - 0.5); const float fy = (pGrid.y - 0.5) - y0;
        const int z0 = (int)(pGrid.z - 0.5); const float fz = (pGrid.z - 0.5) - z0;

        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const int z1 = z0 + 1;

        if ( !(x0 >= 0 && x1 < dim.x && y0 >= 0 && y1 < dim.y && z0 >=0 && z1 < dim.z) ) {
            printf("nope2 (%d)\n",isInBoundsInterp(pGrid));
        }

        const float dx00 = lerp( getValue(make_int3(x0,y0,z0)), getValue(make_int3(x1,y0,z0)), fx);
        const float dx01 = lerp( getValue(make_int3(x0,y0,z1)), getValue(make_int3(x1,y0,z1)), fx);
        const float dx10 = lerp( getValue(make_int3(x0,y1,z0)), getValue(make_int3(x1,y1,z0)), fx);
        const float dx11 = lerp( getValue(make_int3(x0,y1,z1)), getValue(make_int3(x1,y1,z1)), fx);

        const float dxy0 = lerp( dx00, dx10, fy );
        const float dxy1 = lerp( dx01, dx11, fy );

        const float dxyz = lerp( dxy0, dxy1, fz );

        return dxyz;

    }

    inline PREFIX T getValueInterpolated3(const float3 & pGrid) const {

        const int x0 = (int)(pGrid.x - 0.5); const float fx = (pGrid.x - 0.5) - x0;
        const int y0 = (int)(pGrid.y - 0.5); const float fy = (pGrid.y - 0.5) - y0;
        const int z0 = (int)(pGrid.z - 0.5); const float fz = (pGrid.z - 0.5) - z0;

        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const int z1 = z0 + 1;

        if ( !(x0 >= 0 && x1 < dim.x && y0 >= 0 && y1 < dim.y && z0 >=0 && z1 < dim.z) ) {
            printf("nope3 (%d)\n",isInBoundsInterp(pGrid));
        }

        const float dx00 = lerp( getValue(make_int3(x0,y0,z0)), getValue(make_int3(x1,y0,z0)), fx);
        const float dx01 = lerp( getValue(make_int3(x0,y0,z1)), getValue(make_int3(x1,y0,z1)), fx);
        const float dx10 = lerp( getValue(make_int3(x0,y1,z0)), getValue(make_int3(x1,y1,z0)), fx);
        const float dx11 = lerp( getValue(make_int3(x0,y1,z1)), getValue(make_int3(x1,y1,z1)), fx);

        const float dxy0 = lerp( dx00, dx10, fy );
        const float dxy1 = lerp( dx01, dx11, fy );

        const float dxyz = lerp( dxy0, dxy1, fz );

        return dxyz;

    }

    inline PREFIX T getValueInterpolated4(const float3 & pGrid) const {

        const int x0 = (int)(pGrid.x - 0.5); const float fx = (pGrid.x - 0.5) - x0;
        const int y0 = (int)(pGrid.y - 0.5); const float fy = (pGrid.y - 0.5) - y0;
        const int z0 = (int)(pGrid.z - 0.5); const float fz = (pGrid.z - 0.5) - z0;

        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const int z1 = z0 + 1;

        if ( !(x0 >= 0 && x1 < dim.x && y0 >= 0 && y1 < dim.y && z0 >=0 && z1 < dim.z) ) {
            printf("nope4 (%d)\n",isInBoundsInterp(pGrid));
        }

        const float dx00 = lerp( getValue(make_int3(x0,y0,z0)), getValue(make_int3(x1,y0,z0)), fx);
        const float dx01 = lerp( getValue(make_int3(x0,y0,z1)), getValue(make_int3(x1,y0,z1)), fx);
        const float dx10 = lerp( getValue(make_int3(x0,y1,z0)), getValue(make_int3(x1,y1,z0)), fx);
        const float dx11 = lerp( getValue(make_int3(x0,y1,z1)), getValue(make_int3(x1,y1,z1)), fx);

        const float dxy0 = lerp( dx00, dx10, fy );
        const float dxy1 = lerp( dx01, dx11, fy );

        const float dxyz = lerp( dxy0, dxy1, fz );

        return dxyz;

    }

    inline PREFIX T getValueInterpolated5(const float3 & pGrid) const {

        const int x0 = (int)(pGrid.x - 0.5); const float fx = (pGrid.x - 0.5) - x0;
        const int y0 = (int)(pGrid.y - 0.5); const float fy = (pGrid.y - 0.5) - y0;
        const int z0 = (int)(pGrid.z - 0.5); const float fz = (pGrid.z - 0.5) - z0;

        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const int z1 = z0 + 1;

        if ( !(x0 >= 0 && x1 < dim.x && y0 >= 0 && y1 < dim.y && z0 >=0 && z1 < dim.z) ) {
            printf("nope5 (%d)\n",isInBoundsInterp(pGrid));
        }

        const float dx00 = lerp( getValue(make_int3(x0,y0,z0)), getValue(make_int3(x1,y0,z0)), fx);
        const float dx01 = lerp( getValue(make_int3(x0,y0,z1)), getValue(make_int3(x1,y0,z1)), fx);
        const float dx10 = lerp( getValue(make_int3(x0,y1,z0)), getValue(make_int3(x1,y1,z0)), fx);
        const float dx11 = lerp( getValue(make_int3(x0,y1,z1)), getValue(make_int3(x1,y1,z1)), fx);

        const float dxy0 = lerp( dx00, dx10, fy );
        const float dxy1 = lerp( dx01, dx11, fy );

        const float dxyz = lerp( dxy0, dxy1, fz );

        return dxyz;

    }

    inline PREFIX T3 getGradient(const int3 & v) const {

        T3 grad;

        if (v.x == 0) {
            grad.x = data[v.x+1 + v.y*dim.x + v.z*dim.x*dim.y] - data[v.x + v.y*dim.x + v.z*dim.x*dim.y];
        }  else if (v.x == dim.x - 1) {
            grad.x = data[v.x + v.y*dim.x + v.z*dim.x*dim.y] - data[v.x-1 + v.y*dim.x + v.z*dim.x*dim.y];
        }  else {
            grad.x = 0.5*(data[v.x+1 + v.y*dim.x + v.z*dim.x*dim.y] - data[v.x-1 + v.y*dim.x + v.z*dim.x*dim.y]);
        }

        if (v.y == 0) {
            grad.y = data[v.x + (v.y+1)*dim.x + v.z*dim.x*dim.y] - data[v.x + v.y*dim.x + v.z*dim.x*dim.y];
        } else if (v.y == dim.y - 1) {
            grad.y = data[v.x + v.y*dim.x + v.z*dim.x*dim.y] - data[v.x + (v.y-1)*dim.x + v.z*dim.x*dim.y];
        } else {
            grad.y = 0.5*(data[v.x + (v.y+1)*dim.x + v.z*dim.x*dim.y] - data[v.x + (v.y-1)*dim.x + v.z*dim.x*dim.y]);
        }

        if (v.z == 0) {
            grad.z = data[v.x + v.y*dim.x + (v.z+1)*dim.x*dim.y] - data[v.x + v.y*dim.x + v.z*dim.x*dim.y];
        } else if (v.z == dim.z - 1) {
            grad.z = data[v.x + v.y*dim.x + v.z*dim.x*dim.y] - data[v.x + v.y*dim.x + (v.z-1)*dim.x*dim.y];
        } else {
            grad.z = 0.5*(data[v.x + v.y*dim.x + (v.z+1)*dim.x*dim.y] - data[v.x + v.y*dim.x + (v.z-1)*dim.x*dim.y]);
        }

        return grad;

    }

    inline PREFIX T3 getGradientInterpolated(const float3 & pGrid) const {

        T f_px = getValueInterpolated_(pGrid + make_float3(1,0,0));
        T f_py = getValueInterpolated_(pGrid + make_float3(0,1,0));
        T f_pz = getValueInterpolated_(pGrid + make_float3(0,0,1));

        T f_mx = getValueInterpolated_(pGrid - make_float3(1,0,0));
        T f_my = getValueInterpolated_(pGrid - make_float3(0,1,0));
        T f_mz = getValueInterpolated_(pGrid - make_float3(0,0,1));

        T3 grad;
        grad.x = 0.5*(f_px - f_mx);
        grad.y = 0.5*(f_py - f_my);
        grad.z = 0.5*(f_pz - f_mz);
        return grad;

    }

    inline PREFIX T3 getGradientInterpolated1(const float3 & pGrid) const {

        T f_px = getValueInterpolated1(pGrid + make_float3(1,0,0));
        T f_py = getValueInterpolated1(pGrid + make_float3(0,1,0));
        T f_pz = getValueInterpolated1(pGrid + make_float3(0,0,1));

        T f_mx = getValueInterpolated1(pGrid - make_float3(1,0,0));
        T f_my = getValueInterpolated1(pGrid - make_float3(0,1,0));
        T f_mz = getValueInterpolated1(pGrid - make_float3(0,0,1));

        T3 grad;
        grad.x = 0.5*(f_px - f_mx);
        grad.y = 0.5*(f_py - f_my);
        grad.z = 0.5*(f_pz - f_mz);
        return grad;

    }

    inline PREFIX T3 getGradientInterpolated2(const float3 & pGrid) const {

        T f_px = getValueInterpolated2(pGrid + make_float3(1,0,0));
        T f_py = getValueInterpolated2(pGrid + make_float3(0,1,0));
        T f_pz = getValueInterpolated2(pGrid + make_float3(0,0,1));

        T f_mx = getValueInterpolated2(pGrid - make_float3(1,0,0));
        T f_my = getValueInterpolated2(pGrid - make_float3(0,1,0));
        T f_mz = getValueInterpolated2(pGrid - make_float3(0,0,1));

        T3 grad;
        grad.x = 0.5*(f_px - f_mx);
        grad.y = 0.5*(f_py - f_my);
        grad.z = 0.5*(f_pz - f_mz);
        return grad;

    }

    inline PREFIX T3 getGradientInterpolated3(const float3 & pGrid) const {

        T f_px = getValueInterpolated3(pGrid + make_float3(1,0,0));
        T f_py = getValueInterpolated3(pGrid + make_float3(0,1,0));
        T f_pz = getValueInterpolated3(pGrid + make_float3(0,0,1));

        T f_mx = getValueInterpolated3(pGrid - make_float3(1,0,0));
        T f_my = getValueInterpolated3(pGrid - make_float3(0,1,0));
        T f_mz = getValueInterpolated3(pGrid - make_float3(0,0,1));

        T3 grad;
        grad.x = 0.5*(f_px - f_mx);
        grad.y = 0.5*(f_py - f_my);
        grad.z = 0.5*(f_pz - f_mz);
        return grad;

    }

    inline PREFIX T3 getGradientInterpolated4(const float3 & pGrid) const {

        T f_px = getValueInterpolated4(pGrid + make_float3(1,0,0));
        T f_py = getValueInterpolated4(pGrid + make_float3(0,1,0));
        T f_pz = getValueInterpolated4(pGrid + make_float3(0,0,1));

        T f_mx = getValueInterpolated4(pGrid - make_float3(1,0,0));
        T f_my = getValueInterpolated4(pGrid - make_float3(0,1,0));
        T f_mz = getValueInterpolated4(pGrid - make_float3(0,0,1));

        T3 grad;
        grad.x = 0.5*(f_px - f_mx);
        grad.y = 0.5*(f_py - f_my);
        grad.z = 0.5*(f_pz - f_mz);
        return grad;

    }

    inline PREFIX T3 getGradientInterpolated5(const float3 & pGrid) const {

        T f_px = getValueInterpolated5(pGrid + make_float3(1,0,0));
        T f_py = getValueInterpolated5(pGrid + make_float3(0,1,0));
        T f_pz = getValueInterpolated5(pGrid + make_float3(0,0,1));

        T f_mx = getValueInterpolated5(pGrid - make_float3(1,0,0));
        T f_my = getValueInterpolated5(pGrid - make_float3(0,1,0));
        T f_mz = getValueInterpolated5(pGrid - make_float3(0,0,1));

        T3 grad;
        grad.x = 0.5*(f_px - f_mx);
        grad.y = 0.5*(f_py - f_my);
        grad.z = 0.5*(f_pz - f_mz);
        return grad;

    }

    // ===== data members =====
    uint3 dim;
    float3 offset;
    float resolution;
    T * data;

};

}

#endif // GRID_3D_H
