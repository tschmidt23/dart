
#include "gl_dart.h"

#include <GL/glew.h>

namespace dart {

void glLoadSE3(const dart::SE3 & mx) {
    float mxData[16] = { mx.r0.x, mx.r1.x, mx.r2.x, 0,
                         mx.r0.y, mx.r1.y, mx.r2.y, 0,
                         mx.r0.z, mx.r1.z, mx.r2.z, 0,
                         mx.r0.w, mx.r1.w, mx.r2.w, 1 };
    glLoadMatrixf(mxData);
}

void glMultSE3(const dart::SE3 & mx) {
    float mxData[16] = { mx.r0.x, mx.r1.x, mx.r2.x, 0,
                         mx.r0.y, mx.r1.y, mx.r2.y, 0,
                         mx.r0.z, mx.r1.z, mx.r2.z, 0,
                         mx.r0.w, mx.r1.w, mx.r2.w, 1 };
    glMultMatrixf(mxData);
}

}
