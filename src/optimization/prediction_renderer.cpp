#include "prediction_renderer.h"

#include <cuda_gl_interop.h>
#include "optimization/kernels/modToObs.h"
#include "optimization/kernels/raycast.h"
#include "util/cuda_utils.h"

#include <sys/time.h>

#include <GL/glx.h>

namespace dart {

const char * depthVertShaderSrc = "#version 130\n"
        "varying vec4 world_pos;\n"
        "uniform mat4 gl_ModelViewMatrix;"
        "void main(void)\n"
        "{\n"
        "       gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
        "       world_pos = gl_ModelViewMatrix*gl_Vertex;\n"
        "}\n";

const char * depthFragShaderSrc = "#version 130\n"
        "out vec4 out_Color;\n"
        "varying vec4 world_pos;\n"
        "void main(void)\n"
        "{\n"
        "   out_Color = vec4(world_pos.z,world_pos.z,world_pos.z,1.0);\n"
        "}\n";

const char * labeledVertShaderSrc = "#version 130\n"
        "varying vec4 world_pos;\n"
        "varying float id;\n"
        "uniform mat4 gl_ModelViewMatrix;"
        "void main()\n"
        "{\n"
        "   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
        "   world_pos = gl_ModelViewMatrix*gl_Vertex;"
        "   id = gl_Color.x + gl_Color.z + 65536;\n"
        "}";

const char * labeledFragShaderSrc = "#version 130\n"
        "out vec4 out_Color;\n"
        "varying vec4 world_pos;\n"
        "varying float id;\n"
        "void main(void)\n"
        "{\n"
        "       out_Color = vec4(world_pos.x,world_pos.y,world_pos.z,id);\n"
        "}\n";

PredictionRenderer::PredictionRenderer(const int width, const int height, const float2 focalLength) :
     _focalLength(focalLength), _debugBoxIntersections(width*height) {

    _width = width;
    _height = height;

    /*cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(err));
    }
    std::cout << "creating shaders" << std::endl;

    _program = glCreateProgramObjectARB();
    _fragShader = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
    glShaderSourceARB(_fragShader, 1, &depthFragShaderSrc, NULL);
    glCompileShader(_fragShader);
    glAttachObjectARB(_program, _fragShader);
    glLinkProgram(_program);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(err));
    }
    std::cout << "creating textures" << std::endl;

    GLenum glErr = glGetError();

    // init texture
    glGenTextures(1,&_tidRgb);

    glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
        std::cerr << "glGenTextures: " << gluErrorString(glErr) << std::endl;
    }

    glBindTexture(GL_TEXTURE_2D,_tidRgb);

    glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
        std::cerr << "glBindTexture: " << gluErrorString(glErr) << std::endl;
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_FLOAT, 0);

    glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
        std::cerr << "glTexImage2D: " << gluErrorString(glErr) << std::endl;
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("> %s\n",cudaGetErrorString(err));
    }

    cudaGraphicsGLRegisterImage(&_resRgb,_tidRgb,GL_TEXTURE_2D,cudaGraphicsMapFlagsReadOnly);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("< %s\n",cudaGetErrorString(err));
    }
    std::cout << "creating render buffer" << std::endl;

    // init render buffer
    glGenRenderbuffersEXT(1, &_rbid);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, _rbid);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, width, height);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT,0);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(err));
    }
    std::cout << "creating frame buffer" << std::endl;

    // init frame buffer
    glGenFramebuffersEXT(1, &_fbid);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _fbid);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, _tidRgb,0);//_texRgb->tid, 0);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, _rbid); //_renderBuffer->rbid);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(err));
    }
    std::cout << "initializing shaders" << std::endl;

    // init shaders
    _program = glCreateProgramObjectARB();
    _vertShader = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
    _fragShader = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
    glShaderSourceARB(_vertShader, 1, &labeledVertShaderSrc, NULL);
    glShaderSourceARB(_fragShader, 1, &labeledFragShaderSrc, NULL);
    glCompileShader(_vertShader);
    glCompileShader(_fragShader);
    glAttachObjectARB(_program, _vertShader);
    glAttachObjectARB(_program, _fragShader);
    glLinkProgram(_program);*/
    cudaMalloc(&_dPrediction,width*height*sizeof(float4));
    cudaMemset(_dPrediction,0,width*height*sizeof(float4));

//    memcpy(_glK,glK,16*sizeof(double));

//    err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        printf("%s\n",cudaGetErrorString(err));
//    }
//    std::cout << "done" << std::endl;

//    CheckGlDieOnError();

}

PredictionRenderer::~PredictionRenderer() {

    /*glDeleteFramebuffersEXT(1,&_fbid);
    glDeleteRenderbuffersEXT(1,&_rbid);
    glDeleteTextures(1,&_tidRgb);

    glDetachObjectARB(_program,_fragShader);
    glDeleteProgramsARB(1,&_program);*/
    cudaFree(_dPrediction);

}

//void PredictionRenderer::renderPrediction(const Model & model,cudaStream_t & stream) {
//    std::vector<const Model*> models(1);
//    models[0] = &model;
//    renderPrediction(models,stream);
//}

void PredictionRenderer::raytracePrediction(const MirroredModel & model, cudaStream_t & stream) {
    std::vector<const MirroredModel *> models(1);
    models[0] = &model;
    raytracePrediction(models,stream);
}

void PredictionRenderer::raytracePrediction(const std::vector<const MirroredModel *> & models, cudaStream_t & stream) {

    for (int m=0; m<models.size(); ++m) {
        raycastPrediction(_focalLength,
                          make_float2(_width/2,_height/2),
                          _width,_height,m,
                          models[m]->getTransformCameraToModel(),
                          models[m]->getDeviceTransformsModelToFrame(),
                          models[m]->getDeviceTransformsFrameToModel(),
                          models[m]->getDeviceSdfFrames(),
                          models[m]->getDeviceSdfs(),
                          models[m]->getNumSdfs(),
                          _dPrediction,0,
                          stream);
    }

    cudaStreamSynchronize(stream);

    _debugBoxIntersections.syncDeviceToHost();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "gpu_raytracePrediction error: %s\n" << cudaGetErrorString(err) << std::endl;
    }
}

//void PredictionRenderer::renderPrediction(const std::vector<const Model *> & models, cudaStream_t & stream) {

//    CheckGlDieOnError();

////    _frameBuffer->Bind();
//    GLuint attachmentBuffer = GL_COLOR_ATTACHMENT0_EXT;
//    glDrawBuffers(1,&attachmentBuffer);

//    CheckGlDieOnError();

//    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _fbid); //_fbid);

//    CheckGlDieOnError();

//    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB,   GL_FALSE);
//    glClampColorARB(GL_CLAMP_READ_COLOR_ARB,     GL_FALSE);
//    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);

//    glEnable(GL_DEPTH_TEST);

//    glDisable(GL_LIGHTING);
//    glDisable(GL_BLEND);
//    glDisable(GL_ALPHA_TEST);

//    glDisable(GL_SCISSOR_TEST);

//    glColor3f(1.0,1.0,1.0);
//    glClearColor(0,0,0,1.0);
//    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

//    glViewport(0, 0, _width, _height);

//    glMatrixMode(GL_PROJECTION);
//    glLoadMatrixd(_glK);

//    glMatrixMode(GL_MODELVIEW);
//    glLoadIdentity();
//    glPushMatrix();

//    glUseProgram(_program);

////    struct timeval start,end;
////    gettimeofday(&start,NULL);
//    for (int m=0; m<models.size(); ++m) {
//        models[m]->renderLabeled(m);
//    }
////    models[1]->renderLabeled(1);
////    gettimeofday(&end,NULL);
////    std::cout << "render time: " << end.tv_usec - start.tv_usec + (1e6)*(end.tv_sec - start.tv_sec) << std::endl;

//    glPopMatrix();

//    glUseProgram(0);

//    glDrawBuffers( 1, &attachmentBuffer );
//    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
////    _frameBuffer->Unbind();

//   // glClearColor(0.0,0.0,0.0,1.0);

//    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB,   GL_TRUE);
//    glClampColorARB(GL_CLAMP_READ_COLOR_ARB,     GL_TRUE);
//    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_TRUE);

////    static float4 *hPrediction = 0;
////    if (hPrediction == 0) { cudaMallocHost(&hPrediction,_width*_height*sizeof(float4)); }
////    glBindTexture(GL_TEXTURE_2D,_tidRgb);
////
////    gettimeofday(&start,NULL);
////    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, hPrediction);
////    gettimeofday(&end,NULL);
////    std::cout << "get image time: " << end.tv_usec - start.tv_usec + (1e6)*(end.tv_sec - start.tv_sec) << std::endl;
////
////    cudaMemcpyAsync(_dPrediction,hPrediction,_width*_height*sizeof(float4),cudaMemcpyHostToDevice,stream);


//    cudaGraphicsMapResources(1,&_resRgb,stream); //&_texRgb->cuda_res);
//    cudaArray * cArray;
//    cudaGraphicsSubResourceGetMappedArray(&cArray,_resRgb,0,0); //_texRgb->cuda_res,0,0);

//    cudaMemcpy2DFromArrayAsync(_dPrediction,_width*sizeof(float4),cArray,0,0,_width*sizeof(float4),_height,cudaMemcpyDeviceToDevice,stream);

//    cudaGraphicsUnmapResources(1,&_resRgb,stream); //&_texRgb->cuda_res);

//}

void PredictionRenderer::cullUnobservable(const float4 * dObsVertMap,
                                          const int width,
                                          const int height,
                                          const cudaStream_t stream) {

    cullUnobservable_(_dPrediction,_width,_height,dObsVertMap,width,height,stream);

}

void PredictionRenderer::debugPredictionRay(const std::vector<const MirroredModel *> & models, const int x, const int y,
                                            std::vector<MirroredVector<float3> > & boxIntersects,
                                            std::vector<MirroredVector<float2> > & raySteps) {

    for (int m=0; m<models.size(); ++m) {
        std::cout << "debugging ray for " << m << std::endl;
        raycastPredictionDebugRay(_focalLength,
                                  make_float2(_width/2,_height/2),
                                  x,y,_width,m,
                                  models[m]->getTransformCameraToModel(),
                                  models[m]->getDeviceTransformsModelToFrame(),
                                  models[m]->getDeviceTransformsFrameToModel(),
                                  models[m]->getDeviceSdfFrames(),
                                  models[m]->getDeviceSdfs(),
                                  models[m]->getNumSdfs(),
                                  _dPrediction,0,
                                  boxIntersects[m].devicePtr(),
                                  raySteps[m].devicePtr(),raySteps[m].length());
        boxIntersects[m].syncDeviceToHost();
        raySteps[m].syncDeviceToHost();
    }
}

}
