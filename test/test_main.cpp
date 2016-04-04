#include "gtest/gtest.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

int main(int argc, char * * argv) {
    ::testing::InitGoogleTest(&argc, argv);

    glutInit(&argc,argv);
    glutCreateWindow("useless");

    glewInit();

    return RUN_ALL_TESTS();
}
