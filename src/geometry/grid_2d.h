#ifndef GRID_2D_H
#define GRID_2D_H

#include <vector_types.h>
#include <vector_functions.h>
#include "util/vector_operators.h"
#include "util/vector_type_template.h"
#include "util/prefix.h"

namespace dart {

template <typename T>
class Grid2D {

private:
    typedef typename VectorTypeTemplate<T>::type2 T2;

public:

    // ===== constructors / destructor =====
    PREFIX Grid2D() : dim(make_uint2(0,0)), offset(make_float2(0,0)), resolution(0), data(0) {}
    PREFIX Grid2D(uint2 dim) : dim(dim), offset(make_float2(0,0)), resolution(0), data(new T[dim.x*dim.y]) {}
    PREFIX Grid2D(uint2 dim, float2 offset, float resolution) : dim(dim), offset(offset), resolution(resolution), data(new T[dim.x*dim.y]) {}
    PREFIX Grid2D(const Grid2D & grid) : dim(grid.dim), offset(grid.offset), resolution(grid.resolution), data(new T[grid.dim.x*grid.dim.y]) {
        memcpy(data, grid.data, dim.x*dim.y*sizeof(T));
    }
    PREFIX ~Grid2D() { delete [] data; }

    // ===== inline member functions =====
    inline PREFIX Grid2D<T> & operator= (const Grid2D<float> & grid) {
        if (this == &grid) {
            return *this;
        }
        delete [] data;
        dim = grid.dim;
        offset = grid.offset;
        resolution = grid.resolution;
        data = new T[dim.x*dim.y];
        memcpy(grid.data,data,dim.x*dim.y*sizeof(T));
        return *this;
    }

    inline PREFIX float2 getGridCoords(const float2 & pWorld) const {
        return (pWorld - offset)/resolution;
    }

    inline PREFIX float2 getWorldCoords(const float2 & pGrid) const {
        return resolution*pGrid + offset;
    }

    inline PREFIX bool isInBounds(const float2 & pGrid) const {
        return (pGrid.x > 0 && pGrid.x < dim.x &&
                pGrid.y > 0 && pGrid.y < dim.y);
    }

    inline PREFIX bool isInBoundsInterp(const float2 & pGrid) const {
        return (pGrid.x > 0.50001 && pGrid.x < dim.x - 0.50001 &&
                pGrid.y > 0.50001 && pGrid.y < dim.y - 0.50001);
    }

    inline PREFIX bool isInBoundsGradient(const float2 & pGrid) const {
        return (pGrid.x > 1.50001 && pGrid.x < dim.x - 1.50001 &&
                pGrid.y > 1.50001 && pGrid.y < dim.y - 1.50001);
    }

    // TODO: figure out what these values should really be
    inline PREFIX bool isInBoundsGradientInterp(const float2 & pGrid) const {
        return (pGrid.x > 2.50001 && pGrid.x < dim.x - 2.50001 &&
                pGrid.y > 2.50001 && pGrid.y < dim.y - 2.50001);
    }

    inline PREFIX T & getValue(const int2 & v) const {
        return data[v.x + dim.x*v.y];
    }

    inline PREFIX T getValueInterpolated(const float2 & pGrid) const {

        const int x0 = (int)(pGrid.x-0.5); const float fx = (pGrid.x-0.5) - x0;
        const int y0 = (int)(pGrid.y-0.5); const float fy = (pGrid.y-0.5) - y0;

        const int x1 = x0 + 1;
        const int y1 = y0 + 1;

        if ( !(x0 >= 0 && x1 < dim.x && y0 >= 0 && y1 < dim.y) ) {
            printf("nope (%d)\n",isInBoundsInterp(pGrid));
        }

        const float dx0 = lerp( getValue(make_int2(x0,y0)), getValue(make_int2(x1,y0)), fx);
        const float dx1 = lerp( getValue(make_int2(x0,y1)), getValue(make_int2(x1,y1)), fx);
        const float dxy = lerp( dx0, dx1, fy );

        return dxy;

    }

    inline PREFIX T2 getGradient(const int2 & v) const {

        T2 grad;

        if (v.x == 0) {
            grad.x = data[v.x+1 + v.y*dim.x] - data[v.x + v.y*dim.x];
        }  else if (v.x == dim.x - 1) {
            grad.x = data[v.x + v.y*dim.x] - data[v.x-1 + v.y*dim.x];
        } else {
            grad.x = 0.5*(data[v.x+1 + v.y*dim.x] - data[v.x-1 + v.y*dim.x]);
        }

        if (v.y == 0) {
            grad.y = data[v.x + (v.y+1)*dim.x] - data[v.x + v.y*dim.x];
        } else if (v.y == dim.y - 1) {
            grad.y = data[v.x + v.y*dim.x] - data[v.x + (v.y-1)*dim.x];
        } else {
            grad.y = 0.5*(data[v.x + (v.y+1)*dim.x] - data[v.x + (v.y-1)*dim.x]);
        }

        return grad;

    }

    inline PREFIX T2 getGradientInterpolated(const float2 & pGrid) const {

        T f_px = getValueInterpolated(pGrid + make_float2(1,0));
        T f_py = getValueInterpolated(pGrid + make_float2(0,1));

        T f_mx = getValueInterpolated(pGrid - make_float2(1,0));
        T f_my = getValueInterpolated(pGrid - make_float2(0,1));

        T2 grad;
        grad.x = 0.5*(f_px - f_mx);
        grad.y = 0.5*(f_py - f_my);
        return grad;

    }

    // ===== data members =====
    uint2  dim;
    float2 offset;
    float  resolution;
    T *    data;

};

} // namespace dart

#endif // GRID_2D_H
