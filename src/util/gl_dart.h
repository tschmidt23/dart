#ifndef GL_DART_H
#define GL_DART_H

#include <GL/glew.h>
#include "geometry/SE3.h"

namespace dart {

/**
 * @brief glLoadSE3 This function loads a 6DoF transformation matrix stored in an SE3 instance. The semantics are the same as glLoadMatrix.
 * @param mx The 6DoF transformation matrix to load.
 */
void glLoadSE3(const dart::SE3 & mx);

/**
 * @brief glMultSE3 This function multiplies a 6DoF transformation matrix stored in an SE3 instance with the current GL matrix stack. The semantics are the same as glMultMatrix.
 * @param mx The 6DoF transformation matrix to load.
 */
void glMultSE3(const dart::SE3 & mx);

/**
 * @brief glVertexFloat4 This function places a 3D vertex at the location given in the float4 struct. The homogenous coordinate is ignored.
 * @param vertex The vertex location.
 */
inline void glVertexFloat4(const float4 & vertex) { glVertex3f(vertex.x, vertex.y, vertex.z); }

/**
 * @brief glColorUchar3 This function sets the current GL color to that given by the uchar3 struct.
 * @param color The color values (0-255).
 */
inline void glColorUchar3(const uchar3 & color) { glColor3ub(color.x, color.y, color.z); }

}

#endif // GL_DART_H
