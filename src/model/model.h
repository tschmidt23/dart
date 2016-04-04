#ifndef MODELS_H
#define MODELS_H

#include <string.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <sys/types.h>
//#include <Eigen/Dense>

#include <vector_types.h>
#include <vector_functions.h>

#include "geometry/SE3.h"
#include "geometry/grid_3d.h"
#include "pose/pose.h"
#include "util/dart_types.h"
#include "util/model_renderer.h"

namespace dart {

class ModelRenderer;

class Model {
public:

    Model(const int dimensionality) { _dimensionality = dimensionality; }

    Model(const Model & copy);

    int getPoseDimensionality() const { return _dimensionality; }

    virtual void setArticulation(const float * pose) = 0;

    virtual void setPose(const Pose & pose) = 0;

    static void initializeRenderer(MeshReader * meshReader = 0);

    static void shutdownRenderer();

    void render(const char alpha = 0xff) const;
    void renderWireframe() const;
    void renderSkeleton(const float sphereSize = 0.01, const float lineSize=4) const;
    void renderLabeled(const int modelNumber = 0) const;
    void renderVoxels(const float levelSet) const;
    void renderCoordinateFrames(const float size = 0.1) const;
    void renderFrameParents() const;

    virtual uint getNumFrames() const = 0;

    virtual const SE3 & getTransformFrameToModel(const int frame) const = 0;
    virtual const SE3 & getTransformModelToFrame(const int frame) const = 0;
    virtual const SE3 * getTransformsFrameToModel() const = 0;
    virtual const SE3 * getTransformsModelToFrame() const = 0;
    const inline SE3 & getTransformCameraToModel() const { return _T_mc; }
    const inline SE3 & getTransformModelToCamera() const { return _T_cm; }
    const inline SE3 getTransformCameraToFrame(const int frame) const { return getTransformModelToFrame(frame)*getTransformCameraToModel(); }
    const inline SE3 getTransformFrameToCamera(const int frame) const { return getTransformModelToCamera()*getTransformFrameToModel(frame); }

    virtual int getFrameParent(const int frame) const = 0;
    inline int getFrameSdfNumber(const int frame) const { return _frameSdfNumbers[frame]; }
    inline int getFrameNumGeoms(const int frame) const { return _frameGeoms[frame].size(); }
    const inline int * getFrameGeoms(const int frame) const { return _frameGeoms[frame].data(); }

    virtual uint getNumSdfs() const = 0;
    virtual const Grid3D<float> & getSdf(const int sdfNum) const = 0;
    virtual const Grid3D<float> * getSdfs() const = 0;
    virtual uint getSdfFrameNumber(const int sdfNum) const = 0;
    virtual uchar3 getSdfColor(const int sdfNum) const = 0;
    virtual float3 getJointAxis(const int joint) const = 0;
    virtual float3 & getJointPosition(const int joint) = 0;
    virtual float3 & getJointOrientation(const int joint) = 0;
    virtual const float3 & getJointPosition(const int joint) const = 0;
    virtual const float3 & getJointOrientation(const int joint) const = 0;

    virtual int getDependency(const int frame, const int joint) const = 0;
    virtual const int * getDependencies() const = 0;

    inline uint getNumGeoms() const { return _geomTypes.size(); }
    inline GeomType getGeometryType(const int geomNumber) const { return _geomTypes[geomNumber]; }
    virtual uchar3 getGeometryColor(const int geomNumber) const = 0;
    const inline float3 getGeometryScale(const int geomNumber) const { return _geomScales[geomNumber]; }
    const inline SE3 getGeometryTransform(const int geomNumber)  const { return _geomTransforms[geomNumber]; }

    inline int getMeshNumber(const uint geomNumber) const {
        std::map<int,uint>::const_iterator it = _meshNumbers.find(geomNumber);
        if (it == _meshNumbers.end()) {
            return -1;
        }
        return it->second;
    }
    inline const Mesh & getMesh(const int meshNumber) const { return _renderer->getMesh(meshNumber); }

    virtual uint getNumJoints() const = 0;
    virtual int getJointFrame(const int joint) const = 0;
    virtual JointType getJointType(const int joint) const = 0;
    void getModelJacobianOfModelPoint(const float4 & modelPoint, const int associatedFrame, std::vector<float3> & J3D) const;

    inline float getJointMin(const int joint) const { return _jointLimits[joint].x; }
    inline float getJointMax(const int joint) const { return _jointLimits[joint].y; }

    const std::string & getJointName(const int joint) const { return _jointNames[joint]; }
    void renderSdf(const dart::Grid3D<float> & sdf, float levelSet) const;

    void getArticulatedBoundingBox(float3 & mins, float3 & maxs, const float modelSdfPadding, const int nSweepPoints = 4);

    virtual void setSdfColor(const int sdfNum, const uchar3 color) = 0;
    virtual void setGeometryColor(const int geomNum, const uchar3 color) = 0;

    void setModelVersion(const int modelVersion) { _modelVersion = modelVersion; }

    virtual const SE3 getTransformJointAxisToParent(const int joint) const = 0;

protected:
    int _dimensionality;

    // TODO: maybe move some of this to private?
    std::vector<std::vector<int> > _frameGeoms;
    std::vector<GeomType> _geomTypes;
    std::vector<std::string> _jointNames;

    SE3 _T_mc;
    SE3 _T_cm;

    std::vector<float2> _jointLimits;

    std::vector<float3> _geomScales;
    std::vector<SE3> _geomTransforms;

    static ModelRenderer * _renderer;
    std::vector<int> _frameSdfNumbers;



    inline void setMeshNumber(const int geomNumber, const int meshNumber) { _meshNumbers[geomNumber] = meshNumber; }

    int _modelVersion;

private:

    std::map<int,uint> _meshNumbers;

    // geometry-level rendering
    void renderGeometries(void (Model::*prerenderFunc)(const int, const int, const char *) const,
                          const char * args = 0,
                          void (Model::*postrenderFunc)() const = 0) const;

    void renderColoredPrerender(const int frameNumber, const int geomNumber, const char * args) const;

    void renderWireframePrerender(const int frameNumber, const int geomNumber, const char * args) const;
    void renderWireFramePostrender() const;

    void renderLabeledPrerender(const int frameNumber, const int geomNumber, const char * args) const;

    // frame-level rendering
    void renderFrames(void (Model::*renderFrameFunc)(const int,const char*) const,
                      const char * args = 0) const;

    void renderVoxelizedFrame(const int frameNumber, const char * args) const;


    void renderCoordinateFrame(const int frameNumber, const char * args) const;

};

}

#endif // MODELS_H
