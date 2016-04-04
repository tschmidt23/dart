#ifndef DISTANCE_TRANSFORMS_H
#define DISTANCE_TRANSFORMS_H

#define INF 1e20

namespace dart {

    /**
    * @namespace dart
    * Performs a 1D distance transform using the Felzenszwalb method.
    * @param in The source array, which should be of length greater than or equal to width. Values in this array will be unchanged.
    * @param out The destination array, which should be of length greater than or equal to width. Values in this array will be the final distance values.
    * @param width The length of the arrays.
    * @param takeSqrt If true, the function will compute distances rather than squared distances.
    */
    template <typename Real>
    void distanceTransform1D(const Real * in, Real * out, const unsigned int width, bool takeSqrt = true);

    /**
    * Performs a 1D distance transform using the Felzenszwalb method.
    * @param in The source array, which should be of length greater than or equal to width. Values in this array will be unchanged.
    * @param out The destination array, which should be of length greater than or equal to width. Values in this array will be the final distance values.
    * @param width The length of the arrays.
    * @param takeSqrt If true, the function will compute distances rather than squared distances.
    * @param zScratch A preallocated scratch buffer of length (width + 1).
    * @param vScratch A preallocated scratch buffer of length width.
    */
    template <typename Real>
    void distanceTransform1D(const Real * in, Real * out, const unsigned int width, bool takeSqrt, Real * zScratch, int * vScratch);

    /**
    * Performs a 2D distance transform using the Felzenszwalb method.
    * @param im The source array, which should be of length greater than or equal to width*height. Values in this array will be the final distance values.
    * @param scratch The scratch array, which should be of length greater than or equal to width*height. Values in this array will be intermediate values.
    * @param width The width of the transform.
    * @param height The height of the transform.
    */
    template <typename Real, bool takeSqrt>
    void distanceTransform2D(Real * im, Real * scratch, const unsigned int width, const unsigned int height);

    /**
    * Performs a 2D distance transform using the Felzenszwalb method.
    * @param im The source array, which should be of length greater than or equal to width*height. Values in this array will be the final distance values.
    * @param scratch The scratch array, which should be of length greater than or equal to width*height. Values in this array will be intermediate values.
    * @param width The width of the transform.
    * @param height The height of the transform.
    * @param zScratch A preallocated scratch buffer of length (width + 1)*(height+1).
    * @param vScratch A preallocated scratch buffer of length width*height.
    */
    template <typename Real, bool takeSqrt>
    void distanceTransform2D(Real * im, Real * scratch, const unsigned int width, const unsigned int height, Real * zScratch, int * vScratch);

    /**
    * Performs a 3D distance transform using the Felzenszwalb method.
    * @param in The source array, which should be of length greater than or equal to width*height*depth. Values in this array will be the intermediate values.
    * @param out The destination array, which should be of length greater than or equal to width*height. Values in this array will be the final distance values.
    * @param width The width of the transform.
    * @param height The height of the transform.
    * @param depth The depth of the transform.
    */
    template <typename Real, bool takeSqrt>
    void distanceTransform3D(Real * in, Real * out, const unsigned int width, const unsigned int height, const unsigned int depth);

    /**
    * Performs a 3D distance transform using the Felzenszwalb method.
    * @param in The source array, which should be of length greater than or equal to width*height*depth. Values in this array will be the intermediate values.
    * @param out The distination array, which should be of length greater than or equal to width*height. Values in this array will be the final distance values.
    * @param width The width of the transform.
    * @param height The height of the transform.
    * @param depth The depth of the transform.
    * @param zScratch A preallocated scratch buffer of length (width + 1)*(height+1)*(depth+1).
    * @param vScratch A preallocated scratch buffer of length width*height*depth.
    */
    template <typename Real, bool takeSqrt>
    void distanceTransform3D(Real * in, Real * out, const unsigned int width, const unsigned int height, const unsigned int depth, Real * zScratch, int * vScratch);

    template <typename Real, bool takeSqrt>
    void signedDistanceTransform2D(Real * in, Real * out, const unsigned int width, const unsigned int height);

    template <typename Real, bool takeSqrt>
    void signedDistanceTransform2D(Real * in, Real * out, const unsigned int width, const unsigned int height, Real * zScratch, int * vScratch);

    template <typename Real, bool takeSqrt>
    void signedDistanceTransform2D(Real * in, Real * out, const unsigned int width, const unsigned int height, Real * zScratch, int * vScratch, Real * imScratch1, Real * imScratch2);

    /**
    * Performs a 3D signed distance transform using the Felzenszwalb method in two phases. This function assumes that all interior voxels have initial value 0,
    * and all exterior voxels have a positive initial value.
    * @param in The source array, which should be of length greater than or equal to width*height*depth. Values in this array will be unchanged.
    * @param out The destination array, which should be of length greater than or equal to width*height*depth. Values in this array will be the final signed distance values.
    * @param width The width of the transform.
    * @param height The height of the transform.
    * @param depth The depth of the transform.
    */
    template <typename Real, bool takeSqrt>
    void signedDistanceTransform3D(Real * in, Real * out, const unsigned int width, const unsigned int height, const unsigned int depth);

    /**
    * Performs a 3D signed distance transform using the Felzenszwalb method in two phases. This function assumes that all interior voxels have initial value 0,
    * and all exterior voxels have a positive initial value.
    * @param in The source array, which should be of length greater than or equal to width*height*depth. Values in this array will be unchanged.
    * @param out The destination array, which should be of length greater than or equal to width*height*depth. Values in this array will be the final signed distance values.
    * @param width The width of the transform.
    * @param height The height of the transform.
    * @param depth The depth of the transform.
    * @param zScratch A preallocated scratch buffer of length (width + 1)*(height+1)*(depth+1).
    * @param vScratch A preallocated scratch buffer of length width*height*depth.
    */
    template <typename Real, bool takeSqrt>
    void signedDistanceTransform3D(Real * in, Real * out, const unsigned int width, const unsigned int height, const unsigned int depth, Real * zScratch, int * vScratch);

    /**
    * Performs a 3D signed distance transform using the Felzenszwalb method in two phases. This function assumes that all interior voxels have initial value 0,
    * and all exterior voxels have a positive initial value.
    * @param in The source array, which should be of length greater than or equal to width*height*depth. Values in this array will be unchanged.
    * @param out The destination array, which should be of length greater than or equal to width*height*depth. Values in this array will be the final signed distance values.
    * @param width The width of the transform.
    * @param height The height of the transform.
    * @param depth The depth of the transform.
    * @param zScratch A preallocated scratch buffer of length (width + 1)*(height+1)*(depth+1).
    * @param vScratch A preallocated scratch buffer of length width*height*depth.
    * @param imScratch A preallocated scratch buffer of length width*height*depth.
    */
    template <typename Real, bool takeSqrt>
    void signedDistanceTransform3D(Real * in, Real * out, const unsigned int width, const unsigned int height, const unsigned int depth, Real * zScratch, int * vScratch, Real * imScratch);

}

#endif // DISTANCE_TRANSFORMS_H
