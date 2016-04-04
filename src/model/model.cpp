#include "model.h"

#include "GL/glut.h"
#include <Eigen/Eigen>

namespace dart {

ModelRenderer * Model::_renderer;

Model::Model(const Model & copy) {

    _dimensionality = copy._dimensionality;

    _frameGeoms = copy._frameGeoms;
    _frameSdfNumbers = copy._frameSdfNumbers;
    _geomTypes = copy._geomTypes;
    _geomScales = copy._geomScales;
    _geomTransforms = copy._geomTransforms;
    _meshNumbers = copy._meshNumbers;
    _T_mc = copy._T_mc;
    _T_cm = copy._T_cm;
    _jointLimits = copy._jointLimits;
    _jointNames = copy._jointNames;

}

void Model::initializeRenderer(dart::MeshReader * meshReader) {
    Model::_renderer = new dart::ModelRenderer(meshReader);
}

void Model::shutdownRenderer() {
    delete Model::_renderer;
}

void Model::render(const char alpha) const {

    renderGeometries(&Model::renderColoredPrerender, (const char*)&alpha);

}

void Model::renderColoredPrerender(const int frameNumber, const int geomNumber, const char * args) const {

    const char & alpha = *args;
    const uchar3 geomColor = getGeometryColor(geomNumber);
    glColor4ub(geomColor.x,geomColor.y,geomColor.z,alpha);

}

void Model::renderWireframe() const {

    renderGeometries(&Model::renderWireframePrerender,0,&Model::renderWireFramePostrender);

}

void Model::renderSkeleton(const float sphereSize, const float lineSize) const {

    glPushMatrix();

    float mxData[16] = { _T_cm.r0.x, _T_cm.r1.x, _T_cm.r2.x, 0,
                         _T_cm.r0.y, _T_cm.r1.y, _T_cm.r2.y, 0,
                         _T_cm.r0.z, _T_cm.r1.z, _T_cm.r2.z, 0,
                         _T_cm.r0.w, _T_cm.r1.w, _T_cm.r2.w, 1 };
    glMultMatrixf(mxData);

    glEnable(GL_LINE_STIPPLE);
    glLineStipple(1,0xf81f);
    glNormal3f(0,0,1);
    glLineWidth(lineSize);
    glBegin(GL_LINES);
    for (int f=1; f<getNumFrames(); ++f) {
        const dart::SE3& Tme = getTransformFrameToModel(f);
        const dart::SE3& Tparent = getTransformFrameToModel(getFrameParent(f));
        glVertex3f(Tme.r0.w,Tme.r1.w,Tme.r2.w);
        glVertex3f(Tparent.r0.w,Tparent.r1.w,Tparent.r2.w);
    }
    glEnd();
    glDisable(GL_LINE_STIPPLE);
    glLineWidth(1);

    for (int f=0; f<getNumFrames(); ++f) {

        glPushMatrix();

        const dart::SE3& T = getTransformFrameToModel(f);
        glTranslatef(T.r0.w,T.r1.w,T.r2.w);

        glutSolidSphere(sphereSize,15,15);

        glPopMatrix();

    }

    glPopMatrix();

}

void Model::renderWireframePrerender(const int frameNumber,const int geomNumber, const char * args) const {

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

}

void Model::renderWireFramePostrender() const {

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

}

void Model::renderLabeled(const int modelNumber) const {

    renderGeometries(&Model::renderLabeledPrerender,(char *)&modelNumber);

}


void Model::renderLabeledPrerender(const int frameNumber, const int geomNumber, const char* args) const {

    int modelNumber = ((int*)args)[0];
    const int grid = getFrameSdfNumber(frameNumber);
    glColor3f(grid,grid,modelNumber);

}

void Model::renderGeometries(void (Model::*prerenderFunc)(const int, const int, const char *) const,
                             const char* args,
                             void (Model::*postrenderFunc)() const) const {

    glPushMatrix();

    float mxData[16] = { _T_cm.r0.x, _T_cm.r1.x, _T_cm.r2.x, 0,
                         _T_cm.r0.y, _T_cm.r1.y, _T_cm.r2.y, 0,
                         _T_cm.r0.z, _T_cm.r1.z, _T_cm.r2.z, 0,
                         _T_cm.r0.w, _T_cm.r1.w, _T_cm.r2.w, 1 };
    glMultMatrixf(mxData);

    for (int f=0; f<getNumFrames(); f++) {

        // check to make sure there are children
        if (getFrameNumGeoms(f) > 0) {

            glPushMatrix();

            const dart::SE3 & T = getTransformFrameToModel(f);
            float mxData[16] = { T.r0.x, T.r1.x, T.r2.x, 0,
                                 T.r0.y, T.r1.y, T.r2.y, 0,
                                 T.r0.z, T.r1.z, T.r2.z, 0,
                                 T.r0.w, T.r1.w, T.r2.w, 1 };
            glMultMatrixf(mxData);

            for (int g = 0; g < getFrameNumGeoms(f); g++) {
                int geomNumber = getFrameGeoms(f)[g];

                dart::GeomType geomType = getGeometryType(geomNumber);

                glPushMatrix();
                const dart::SE3 geomT = getGeometryTransform(geomNumber); //dart::SE3Fromse3(dart::se3(&geomParams[3]));
                float mxData[16] = { geomT.r0.x, geomT.r1.x, geomT.r2.x, 0,
                                     geomT.r0.y, geomT.r1.y, geomT.r2.y, 0,
                                     geomT.r0.z, geomT.r1.z, geomT.r2.z, 0,
                                     geomT.r0.w, geomT.r1.w, geomT.r2.w, 1 };
                glMultMatrixf(mxData);
                const float3 geomScale = getGeometryScale(geomNumber);
                glScalef(geomScale.x,geomScale.y,geomScale.z);

                (this->*prerenderFunc)(f,geomNumber,args);

                if (geomType == dart::MeshType) {
                    _renderer->renderMesh(getMeshNumber(geomNumber));
                } else {
                    _renderer->renderPrimitive(geomType);
                }
                glPopMatrix();

                if (postrenderFunc != 0) {
                    (this->*postrenderFunc)();
                }

            }

            glPopMatrix();

        }

    }

    glPopMatrix();
}

void Model::renderVoxels(const float levelSet) const {

    renderFrames(&Model::renderVoxelizedFrame,(char*)&levelSet);

}

void Model::renderVoxelizedFrame(const int frameNumber, const char* args) const {

    const float levelSet = *((float*)args);

    if (getFrameNumGeoms(frameNumber) > 0) {
        const int sdfNum = getFrameSdfNumber(frameNumber);
        const dart::Grid3D<float> &sdf = getSdf(sdfNum);
        const uchar3 color = getSdfColor(sdfNum);
        glColor4ub(color.x,color.y,color.z,72);
        renderSdf(sdf,levelSet);
    }
}

void Model::renderSdf(const dart::Grid3D<float> & sdf, float levelSet) const {

    glDisable(GL_BLEND);

    for (int z=0; z<sdf.dim.z; z++) {
        for (int y=0; y<sdf.dim.y; y++) {
            for (int x=0; x<sdf.dim.x; x++) {
                if (sdf.data[x + y*sdf.dim.x + z*sdf.dim.x*sdf.dim.y] <= levelSet) {
                    glPushMatrix();

                    glTranslated(sdf.offset.x + (x+0.5)*sdf.resolution, sdf.offset.y + (y+0.5)*sdf.resolution, sdf.offset.z + (z+0.5)*sdf.resolution);

                    glutSolidCube(sdf.resolution);

                    glPopMatrix();
                }
            }
        }
    }

    glEnable(GL_BLEND);

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH);
    glEnable(GL_LINE_STIPPLE);
    glLineStipple(1, 0xffff);
    glLineWidth(2);
    glBegin(GL_LINES);

    glVertex3d(sdf.offset.x,sdf.offset.y,sdf.offset.z);
    glVertex3d(sdf.offset.x,sdf.offset.y,sdf.offset.z + sdf.dim.z*sdf.resolution);

    glVertex3d(sdf.offset.x,sdf.offset.y + sdf.dim.y*sdf.resolution,sdf.offset.z);
    glVertex3d(sdf.offset.x,sdf.offset.y + sdf.dim.y*sdf.resolution,sdf.offset.z + sdf.dim.z*sdf.resolution);

    glVertex3d(sdf.offset.x + sdf.dim.x*sdf.resolution,sdf.offset.y,sdf.offset.z);
    glVertex3d(sdf.offset.x + sdf.dim.x*sdf.resolution,sdf.offset.y,sdf.offset.z + sdf.dim.z*sdf.resolution);

    glVertex3d(sdf.offset.x + sdf.dim.x*sdf.resolution,sdf.offset.y + sdf.dim.y*sdf.resolution,sdf.offset.z);
    glVertex3d(sdf.offset.x + sdf.dim.x*sdf.resolution,sdf.offset.y + sdf.dim.y*sdf.resolution,sdf.offset.z + sdf.dim.z*sdf.resolution);


    glVertex3d(sdf.offset.x,sdf.offset.y,sdf.offset.z);
    glVertex3d(sdf.offset.x,sdf.offset.y + sdf.dim.y*sdf.resolution,sdf.offset.z);

    glVertex3d(sdf.offset.x,sdf.offset.y,sdf.offset.z + sdf.dim.z*sdf.resolution);
    glVertex3d(sdf.offset.x,sdf.offset.y + sdf.dim.y*sdf.resolution,sdf.offset.z + sdf.dim.z*sdf.resolution);

    glVertex3d(sdf.offset.x + sdf.dim.x*sdf.resolution,sdf.offset.y,sdf.offset.z);
    glVertex3d(sdf.offset.x + sdf.dim.x*sdf.resolution,sdf.offset.y + sdf.dim.y*sdf.resolution,sdf.offset.z);

    glVertex3d(sdf.offset.x + sdf.dim.x*sdf.resolution,sdf.offset.y,sdf.offset.z + sdf.dim.z*sdf.resolution);
    glVertex3d(sdf.offset.x + sdf.dim.x*sdf.resolution,sdf.offset.y + sdf.dim.y*sdf.resolution,sdf.offset.z + sdf.dim.z*sdf.resolution);


    glVertex3d(sdf.offset.x,sdf.offset.y,sdf.offset.z);
    glVertex3d(sdf.offset.x + sdf.dim.x*sdf.resolution,sdf.offset.y,sdf.offset.z);

    glVertex3d(sdf.offset.x,sdf.offset.y,sdf.offset.z + sdf.dim.z*sdf.resolution);
    glVertex3d(sdf.offset.x + sdf.dim.x*sdf.resolution,sdf.offset.y,sdf.offset.z + sdf.dim.z*sdf.resolution);

    glVertex3d(sdf.offset.x,sdf.offset.y + sdf.dim.y*sdf.resolution,sdf.offset.z);
    glVertex3d(sdf.offset.x + sdf.dim.x*sdf.resolution,sdf.offset.y + sdf.dim.y*sdf.resolution,sdf.offset.z);

    glVertex3d(sdf.offset.x,sdf.offset.y + sdf.dim.y*sdf.resolution,sdf.offset.z + sdf.dim.z*sdf.resolution);
    glVertex3d(sdf.offset.x + sdf.dim.x*sdf.resolution,sdf.offset.y + sdf.dim.y*sdf.resolution,sdf.offset.z + sdf.dim.z*sdf.resolution);

    glEnd();
    glDisable(GL_LINE_STIPPLE);
    glEnable(GL_LIGHTING);
    glLineWidth(1);

    glEnable(GL_DEPTH);
    glDisable(GL_BLEND);

}

void Model::renderCoordinateFrames(const float size) const {

    glPushMatrix();

    float mxData[16] = { _T_cm.r0.x, _T_cm.r1.x, _T_cm.r2.x, 0,
                         _T_cm.r0.y, _T_cm.r1.y, _T_cm.r2.y, 0,
                         _T_cm.r0.z, _T_cm.r1.z, _T_cm.r2.z, 0,
                         _T_cm.r0.w, _T_cm.r1.w, _T_cm.r2.w, 1 };
    glMultMatrixf(mxData);

    glColor3ub(0x80,0x80,0x80);
    glLineWidth(1.0);
    glLineStipple(1, 0xff00);
    glEnable(GL_LINE_STIPPLE);

    glBegin(GL_LINES);
    for (int f=1; f<getNumFrames(); ++f) {
        const int p = getFrameParent(f);
        const float4 e0 = getTransformFrameToModel(f)*make_float4(0,0,0,1);
        const float4 e1 = getTransformFrameToModel(p)*make_float4(0,0,0,1);
        glVertex3f(e0.x,e0.y,e0.z);
        glVertex3f(e1.x,e1.y,e1.z);
    }
    glEnd();

    glDisable(GL_LINE_STIPPLE);

    glPopMatrix();

    glLineWidth(4.0);

//    glDisable(GL_DEPTH_TEST);
    renderFrames(&Model::renderCoordinateFrame,(char*)&size);
//    glEnable(GL_DEPTH_TEST);

}

void Model::renderCoordinateFrame(const int frameNumber, const char * args) const {

    const float size = *((float*)args);

    glPushMatrix();

    glRotated(90,0,1.0,0);
    glScalef(0.0025,0.0025,size);

    glColor3f(1.0,0.0,0.0);
    _renderer->renderPrimitive(dart::PrimitiveCylinderType);

    glPopMatrix();

    glPushMatrix();

    glRotated(-90,1.0,0,0);
    glScalef(0.0025,0.0025,size);

    glColor3f(0.0,1.0,0.0);
    _renderer->renderPrimitive(dart::PrimitiveCylinderType);

    glPopMatrix();

    glPushMatrix();

    glScalef(0.0025,0.0025,size);
    glColor3f(0.0,0.0,1.0);
    _renderer->renderPrimitive(dart::PrimitiveCylinderType);

    glPopMatrix();

}

void Model::renderFrames(void (Model::*renderFrameFunc)(const int,const char *) const, const char * args) const {

    glPushMatrix();

    float mxData[16] = { _T_cm.r0.x, _T_cm.r1.x, _T_cm.r2.x, 0,
                         _T_cm.r0.y, _T_cm.r1.y, _T_cm.r2.y, 0,
                         _T_cm.r0.z, _T_cm.r1.z, _T_cm.r2.z, 0,
                         _T_cm.r0.w, _T_cm.r1.w, _T_cm.r2.w, 1 };
    glMultMatrixf(mxData);

    for (int f=0; f<getNumFrames(); f++) {

        glPushMatrix();

        const dart::SE3 & T = getTransformFrameToModel(f);
        float mxData[16] = { T.r0.x, T.r1.x, T.r2.x, 0,
                             T.r0.y, T.r1.y, T.r2.y, 0,
                             T.r0.z, T.r1.z, T.r2.z, 0,
                             T.r0.w, T.r1.w, T.r2.w, 1 };
        glMultMatrixf(mxData);

        (this->*renderFrameFunc)(f,args);

        glPopMatrix();

    }

    glPopMatrix();

}

void Model::renderFrameParents() const {

    glPushMatrix();

    float mxData[16] = { _T_cm.r0.x, _T_cm.r1.x, _T_cm.r2.x, 0,
                         _T_cm.r0.y, _T_cm.r1.y, _T_cm.r2.y, 0,
                         _T_cm.r0.z, _T_cm.r1.z, _T_cm.r2.z, 0,
                         _T_cm.r0.w, _T_cm.r1.w, _T_cm.r2.w, 1 };
    glMultMatrixf(mxData);

    glDisable(GL_DEPTH_TEST);

    glColor3f(0.5,0.5,0.5);
    glBegin(GL_LINES);
    for (int frame=1; frame<getNumFrames(); ++frame) {

        const dart::SE3 & T = getTransformFrameToModel(frame); //_T_ml[frame];
        const int parentFrame = getFrameParent(frame);
        const dart::SE3 & parent_T = getTransformFrameToModel(parentFrame);

        glVertex3f(parent_T.r0.w,parent_T.r1.w,parent_T.r2.w);
        glVertex3f(T.r0.w,T.r1.w,T.r2.w);

    }
    glEnd();

    glEnable(GL_DEPTH_TEST);

    glPopMatrix();

}

void Model::getModelJacobianOfModelPoint(const float4 & modelPoint, const int frame, std::vector<float3> & J3D) const {

    J3D.resize(getPoseDimensionality());

    // fill x,y,z derivatives
    J3D[0] = make_float3(-1, 0, 0);
    J3D[1] = make_float3( 0,-1, 0);
    J3D[2] = make_float3( 0, 0,-1);
    J3D[3] = make_float3(          0.0, modelPoint.z,-modelPoint.y);
    J3D[4] = make_float3(-modelPoint.z,          0.0, modelPoint.x);
    J3D[5] = make_float3( modelPoint.y,-modelPoint.x,          0.0);

//    std::cout << "-=-=-=-=-=-=-=-=-=-=-" << std::endl;
//    std::cout << "modelPoint: " << modelPoint.x << ", " << modelPoint.y << ", " << modelPoint.z << std::endl << std::endl;

    // fill joint parameter derivatives
    for (int joint = 0; joint<getNumJoints(); ++joint) {
        const int i = 6+joint;

        // check if the link is dependent on this joint
        if (!getDependency(frame,joint)) {
            J3D[i] = make_float3(0.0, 0.0, 0.0);
            continue;
        }

        const int jointFrame = getJointFrame(joint);
        const JointType jointType = getJointType(joint);
        const int jointParent = getFrameParent(jointFrame);
        const SE3 & T_mp = getTransformFrameToModel(jointParent);

        // get joint axis
        const float3 & jointPosition = getJointPosition(joint);
        const float3 & jointOrientation = getJointOrientation(joint);
        SE3 T_pa = SE3FromTranslation(jointPosition)*SE3Fromse3(se3(0,0,0,
                                  jointOrientation.x, jointOrientation.y, jointOrientation.z));

        if (jointType == RotationalJoint) {

            // get the point representing link's origin relative to the joint link's parent's frame
            float4 parentLinkPoint = getTransformModelToFrame(jointParent)*modelPoint; //parentT_mf*modelPoint;
            float4 axisRelativePoint = getTransformModelToFrame(jointFrame)*modelPoint;

            float3 derivative_a = cross(getJointAxis(joint),make_float3(axisRelativePoint));

            float3 derivative = SE3Rotate(getTransformFrameToModel(jointFrame),derivative_a);

//            std::cout << "joint " << joint << ": " << std::endl;
//            std::cout << "\taxisRelativePoint = " << axisRelativePoint.x << ", " << axisRelativePoint.y << ", " << axisRelativePoint.z << std::endl;
//            std::cout << "\tderivative_a = " << derivative_a.x << ", " << derivative_a.y << ", " << derivative_a.z << std::endl;
//            std::cout << "\tderivative = " << derivative.x << ", " << derivative.y << ", " << derivative.z << std::endl;
//            std::cout << std::endl << std::endl;


            //            // convert the point to a vector and rotate it into the frame of joint link's parent
//            parentLinkPoint.w = 0;
//            float4 pointVec = parentT_mf*parentLinkPoint;

//            // take cross-produce of z vector and vector from joint link's parent to point
//            float3 derivative = -1*cross(axis_m,make_float3(pointVec));

            J3D[i] = make_float3(derivative.x, derivative.y, derivative.z);
        }
        else if (jointType == PrismaticJoint) {

            const float3 axis_p = SE3Rotate(T_pa,getJointAxis(joint));
            const float3 axis_m = SE3Rotate(getTransformFrameToModel(jointParent),axis_p);

            // prismatic joint: translation around z axis
            J3D[i] = make_float3(axis_m.x, axis_m.y, axis_m.z);

        }

    }

}

void Model::getArticulatedBoundingBox(float3 & mins, float3 & maxs, const float modelSdfPadding, const int nSweepPoints) {

    mins = getSdf(0).offset + make_float3(modelSdfPadding);
    maxs = getSdf(0).offset - make_float3(modelSdfPadding) + getSdf(0).resolution*make_float3(getSdf(0).dim.x,getSdf(0).dim.y,getSdf(0).dim.z);

    for (int s=1; s<getNumSdfs(); ++s) {

        const Grid3D<float> & sdf = getSdf(s);
        const int frame = getSdfFrameNumber(s);

        std::vector<int> jointDependencies;
        for (int j=0; j<getNumJoints(); ++j) {
            if (getDependency(frame,j)) {
                jointDependencies.push_back(j);
            }
        }
        if (jointDependencies.size() == 0) { continue; }

        float4 sdfMin_f = make_float4(sdf.offset + make_float3(modelSdfPadding),1);
        float4 sdfMax_f = make_float4(sdf.offset - make_float3(modelSdfPadding) + sdf.resolution*make_float3(sdf.dim.x,sdf.dim.y,sdf.dim.z),1);

        float4 corners[8];
        for (int i=0; i<8; ++i) {
            corners[i].w = 1;
            corners[i].x = (i & 4) ? sdfMin_f.x : sdfMax_f.x;
            corners[i].y = (i & 2) ? sdfMin_f.y : sdfMax_f.y;
            corners[i].z = (i & 1) ? sdfMin_f.z : sdfMax_f.z;
        }

        std::vector<std::vector<float> > jointAngleCombinations(1);
        for (int j=0; j<jointDependencies.size(); ++j) {
            std::vector<std::vector<float> > newJointAngleCombinations;
            const int joint = jointDependencies[j];
            const float jointMin = getJointMin(joint);
            const float jointMax = getJointMax(joint);
            for (int c=0; c<jointAngleCombinations.size(); ++c) {
                const std::vector<float> & combinations = jointAngleCombinations[c];
                for (int a=0; a<nSweepPoints; ++a) {
                    float jointAngle = jointMin + ((float)a/(nSweepPoints-1.f))*(jointMax-jointMin);
                    std::vector<float> newCombination = combinations;
                    newCombination.push_back(jointAngle);
                    newJointAngleCombinations.push_back(newCombination);
                }
            }
            jointAngleCombinations = newJointAngleCombinations;
        }

//        std::cout << jointDependencies.size() << " joint dependencies leads to " << jointAngleCombinations.size() <<
//                     " joint angle combinations" << std::endl;

        float pose[getPoseDimensionality()];
        for (int c=0; c<jointAngleCombinations.size(); ++c) {

            for (int j=0; j<jointDependencies.size(); ++j) {
                const int joint = jointDependencies[j];
                pose[joint+6] = jointAngleCombinations[c][j];
            }

            setArticulation(pose);

            for (int p=0; p<8; ++p) {
                float4 corner_m = getTransformFrameToModel(frame)*corners[p];
                mins.x = std::min(mins.x,corner_m.x);
                mins.y = std::min(mins.y,corner_m.y);
                mins.z = std::min(mins.z,corner_m.z);
                maxs.x = std::max(maxs.x,corner_m.x);
                maxs.y = std::max(maxs.y,corner_m.y);
                maxs.z = std::max(maxs.z,corner_m.z);
            }

        }

    }

}

}
