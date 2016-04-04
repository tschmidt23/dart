#ifndef PREDICTION_RENDERER_H
#define PREDICTION_RENDERER_H

#include <GL/glew.h>
#include <stdio.h>

#include "model/model.h"
#include "model/mirrored_model.h"
#include "util/mirrored_memory.h"

namespace dart {

class PredictionRenderer {
public:
    PredictionRenderer(const int width, const int height, const float2 focalLength);

    ~PredictionRenderer();

    int getWidth() const { return _width; }

    int getHeight() const { return _height; }

    const float4 * getDevicePrediction() const { cudaDeviceSynchronize(); return _dPrediction; }
//    void renderPrediction(const Model & model,cudaStream_t & stream);
//    void renderPrediction(const std::vector<const Model*> & models, cudaStream_t & stream);

    void raytracePrediction(const MirroredModel & model, cudaStream_t & stream) ;

    void raytracePrediction(const std::vector<const MirroredModel*> & models, cudaStream_t & stream);

    void cullUnobservable(const float4 * dObsVertMap, const int width, const int height, const cudaStream_t stream = 0);

    const unsigned char * getDebugBoxIntersection() const { return _debugBoxIntersections.hostPtr(); }

    void debugPredictionRay(const std::vector<const MirroredModel *> & models, const int x, const int y, std::vector<MirroredVector<float3> > & boxIntersects, std::vector<MirroredVector<float2> > & raySteps);
private:

    int _width, _height;
    float2 _focalLength;
//    double _glK[16];

//    GLuint _tidRgb;
//    cudaGraphicsResource * _resRgb;

//    GLuint _rbid;
//    GLuint _fbid;

    float4 * _dPrediction;
//    GLenum _program;
//    GLhandleARB _fragShader;
//    GLhandleARB _vertShader;

    MirroredVector<unsigned char> _debugBoxIntersections;
};

}

#endif // PREDICTION_RENDERER_H
