#include "dart_io.h"

#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <sstream>

#define TIXML_USE_STL
#include <tinyxml.h>

namespace dart {

const static int currentFileVersion = 1;

void readFrameXML(const int parent, HostOnlyModel & model, TiXmlElement * element, std::string & dir, const int version) {

    for (TiXmlElement * child = element->FirstChildElement(); child != 0; child = child->NextSiblingElement()) {

        if (strcmp(child->Value(),"frame") == 0) {
            std::string jointName = child->Attribute("jointName");
            std::string jointTypeStr = child->Attribute("jointType");
            dart::JointType jointType;
            if (strcmp(jointTypeStr.c_str(),"rotational") == 0) {
                jointType = dart::RotationalJoint;
            } else if (strcmp(jointTypeStr.c_str(),"prismatic") == 0) {
                jointType = dart::PrismaticJoint;
            }  else {
                std::cerr << jointTypeStr << " is not a recognized joint type; the options are 'rotational' and 'prismatic.'" << std::endl;
            }
            std::string jointMin = child->Attribute("jointMin");
            std::string jointMax = child->Attribute("jointMax");
            std::string posX, posY, posZ, oriX, oriY, oriZ, axisX, axisY, axisZ;
            for (TiXmlElement * grandChild = child->FirstChildElement(); grandChild != 0; grandChild = grandChild->NextSiblingElement()) {
                if (strcmp(grandChild->Value(),"position") == 0) {
                    posX = grandChild->Attribute("x");
                    posY = grandChild->Attribute("y");
                    posZ = grandChild->Attribute("z");
                } else if (strcmp(grandChild->Value(),"orientation") == 0) {
                    oriX = grandChild->Attribute("x");
                    oriY = grandChild->Attribute("y");
                    oriZ = grandChild->Attribute("z");
                } else if (strcmp(grandChild->Value(),"axis") == 0) {
                    axisX = grandChild->Attribute("x");
                    axisY = grandChild->Attribute("y");
                    axisZ = grandChild->Attribute("z");
                }
            }
            int ID = model.addFrame(parent,jointType,posX,posY,posZ,oriX,oriY,oriZ,axisX,axisY,axisZ,jointMin,jointMax,jointName);
            readFrameXML(ID,model,child,dir,version);
        }
        else if (strcmp(child->Value(),"geom") == 0) {
            GeomType type;
            std::string typeStr = child->Attribute("type");
            if (strcmp(typeStr.c_str(),"sphere") == 0) {
                type = PrimitiveSphereType;
            } else if (strcmp(typeStr.c_str(),"cylinder") == 0) {
                type = PrimitiveCylinderType;
            } else if (strcmp(typeStr.c_str(),"cube") == 0) {
                type = PrimitiveCubeType;
            } else if (strcmp(typeStr.c_str(),"mesh") == 0) {
                type = MeshType;
            }
            int color[3];
            child->Attribute("red",&color[0]);
            child->Attribute("green",&color[1]);
            child->Attribute("blue",&color[2]);
            if (type == MeshType) {
                model.addGeometry(parent,type,
                                  child->Attribute("sx"),
                                  child->Attribute("sy"),
                                  child->Attribute("sz"),
                                  child->Attribute("tx"),
                                  child->Attribute("ty"),
                                  child->Attribute("tz"),
                                  child->Attribute("rx"),
                                  child->Attribute("ry"),
                                  child->Attribute("rz"),
                                  color[0],color[1],color[2],
                        dir + "/" + child->Attribute("meshFile"));
            }
            else {
                model.addGeometry(parent,type,
                                  child->Attribute("sx"),
                                  child->Attribute("sy"),
                                  child->Attribute("sz"),
                                  child->Attribute("tx"),
                                  child->Attribute("ty"),
                                  child->Attribute("tz"),
                                  child->Attribute("rx"),
                                  child->Attribute("ry"),
                                  child->Attribute("rz"),
                                  color[0],color[1],color[2]);
            }
        }
        else if (strcmp(child->Value(),"param") == 0) {

        }
        else {
            // TODO
//            std::cerr << "unexpected node: " << child->Value() << std::endl;
        }

    }

}

bool readModelXML(const char * filename, HostOnlyModel & model) {

    struct stat buffer;
    if (stat (filename, &buffer) != 0) {
        std::cerr << "file " << filename << " does not exist" << std::endl;
        return false;
    }

    TiXmlDocument doc( filename );
    doc.LoadFile();

    TiXmlElement * root = doc.FirstChildElement();

    if (strcmp(root->Value(),"model") != 0) {
        std::cerr << "expected 'model' at root node; got '" << root->Value() << "'" << std::endl;
        return false;
    }

    // check version
    int modelVersion = 0;
    root->QueryIntAttribute("version",&modelVersion);
    std::cout << filename << " is model version " << modelVersion << std::endl;
    model.setModelVersion(modelVersion);

    // read size params
    for (TiXmlElement * child = root->FirstChildElement(); child != 0; child = child->NextSiblingElement()) {
        if (strcmp(child->Value(),"param") == 0) {
            std::string name = std::string(child->Attribute("name"));
            double value;
            child->Attribute("value",&value);
            model.addSizeParam(name,value);
        }
    }

    std::string dir(filename);
    dir = dir.substr(0,dir.find_last_of('/'));

    readFrameXML(0,model,root,dir,modelVersion);

    return true;
}

void writeGeomXML(const HostOnlyModel & model, const int geom, TiXmlElement * parent) {
    TiXmlElement * geomEl = new TiXmlElement("geom");
    switch (model.getGeometryType(geom)) {
    case PrimitiveSphereType:
        geomEl->SetAttribute("type","sphere");
        break;
    case PrimitiveCylinderType:
        geomEl->SetAttribute("type","cylinder");
        break;
    case PrimitiveCubeType:
        geomEl->SetAttribute("type","cube");
        break;
    case MeshType:
        geomEl->SetAttribute("type","mesh");
        break;
    }
    geomEl->SetDoubleAttribute("sx",model.getGeometryScale(geom).x);
    geomEl->SetDoubleAttribute("sy",model.getGeometryScale(geom).y);
    geomEl->SetDoubleAttribute("sz",model.getGeometryScale(geom).z);
    const SE3 T = model.getGeometryTransform(geom);
    geomEl->SetDoubleAttribute("tx",T.r0.w);
    geomEl->SetDoubleAttribute("ty",T.r1.w);
    geomEl->SetDoubleAttribute("tz",T.r2.w);
    const float3 phiThetaPsi = eulerFromSE3(T);
    geomEl->SetDoubleAttribute("rx",phiThetaPsi.z);
    geomEl->SetDoubleAttribute("ry",phiThetaPsi.y);
    geomEl->SetDoubleAttribute("rz",phiThetaPsi.x);
    geomEl->SetAttribute("red",model.getGeometryColor(geom).x);
    geomEl->SetAttribute("green",model.getGeometryColor(geom).y);
    geomEl->SetAttribute("blue",model.getGeometryColor(geom).z);
    if (model.getGeometryType(geom) == MeshType) {
        geomEl->SetAttribute("meshFile",model.getGeometryMeshFilename(geom).c_str());
    }
    parent->LinkEndChild(geomEl);
}

void writeFrameXML(const HostOnlyModel & model, const int frame, TiXmlElement * parent) {

    int joint = frame-1;

    // create element for this frame
    TiXmlElement * frameEl = new TiXmlElement("frame");
    frameEl->SetAttribute("jointName",model.getJointName(joint).c_str());
    switch (model.getJointType(joint)) {
    case RotationalJoint:
        frameEl->SetAttribute("jointType","rotational");
        break;
    case PrismaticJoint:
        frameEl->SetAttribute("jointType","prismatic");
        break;
    }
    frameEl->SetDoubleAttribute("jointMin",model.getJointMin(joint));
    frameEl->SetDoubleAttribute("jointMax",model.getJointMax(joint));

    TiXmlElement * jointPos = new TiXmlElement("position");
    SE3 T_pf = model.getTransformJointAxisToParent(joint);
    jointPos->SetDoubleAttribute("x",T_pf.r0.w);
    jointPos->SetDoubleAttribute("y",T_pf.r1.w);
    jointPos->SetDoubleAttribute("z",T_pf.r2.w);
//    jointPos->SetDoubleAttribute("x",model.getJointPosition(joint).x);
//    jointPos->SetDoubleAttribute("y",model.getJointPosition(joint).y);
//    jointPos->SetDoubleAttribute("z",model.getJointPosition(joint).z);
    frameEl->LinkEndChild(jointPos);

    TiXmlElement * jointOrientation = new TiXmlElement("orientation");

    float3 phiThetaPsi = eulerFromSE3(T_pf);
    jointOrientation->SetDoubleAttribute("x",phiThetaPsi.z);
    jointOrientation->SetDoubleAttribute("y",phiThetaPsi.y);
    jointOrientation->SetDoubleAttribute("z",phiThetaPsi.x);
//    jointOrientation->SetDoubleAttribute("x",model.getJointOrientation(joint).x);
//    jointOrientation->SetDoubleAttribute("y",model.getJointOrientation(joint).y);
//    jointOrientation->SetDoubleAttribute("z",model.getJointOrientation(joint).z);
    frameEl->LinkEndChild(jointOrientation);

    TiXmlElement * jointAxis = new TiXmlElement("axis");
    jointAxis->SetDoubleAttribute("x",model.getJointAxis(joint).x);
    jointAxis->SetDoubleAttribute("y",model.getJointAxis(joint).y);
    jointAxis->SetDoubleAttribute("z",model.getJointAxis(joint).z);
    frameEl->LinkEndChild(jointAxis);

    parent->LinkEndChild(frameEl);

    // add child geometry
    for (int g=0; g<model.getFrameNumGeoms(frame); ++g) {
        const int geom = model.getFrameGeoms(frame)[g];
        writeGeomXML(model,geom,frameEl);
    }

    // recursively add child frames
    for (int c=0; c<model.getFrameNumChildren(frame); ++c) {
        writeFrameXML(model,model.getFrameChildren(frame)[c],frameEl);
    }

}

void writeModelXML(const HostOnlyModel & model, const char * filename) {

    TiXmlDocument doc;
    TiXmlDeclaration * declaration = new TiXmlDeclaration("1.0","","");
    doc.LinkEndChild(declaration);

    TiXmlElement * root = new TiXmlElement("model");
    doc.LinkEndChild(root);

    root->SetAttribute("version",currentFileVersion);

    const std::map<std::string,float> & sizeParams = model.getSizeParams();
    for (std::map<std::string,float>::const_iterator it = sizeParams.begin(); it != sizeParams.end(); ++it) {
        TiXmlElement * paramElement = new TiXmlElement("param");
        paramElement->SetAttribute("name",it->first.c_str());
        paramElement->SetDoubleAttribute("value",it->second);
        root->LinkEndChild(paramElement);
    }

    // write root geometry
    for (int g=0; g<model.getFrameNumGeoms(0); ++g) {
        const int geom = model.getFrameGeoms(0)[g];
        writeGeomXML(model,geom,root);
    }

    // write child frames (recursively)
    for (int c=0; c<model.getFrameNumChildren(0); ++c) {
        writeFrameXML(model,model.getFrameChildren(0)[c],root);
    }

    doc.SaveFile(filename);

}

void saveState(const float * pose, const int dimensions, const int frame, std::string filename) {

    int filesize = sizeof(int)+dimensions*sizeof(float);
    char data[filesize];

    int * frame_data = (int*)(data);
    frame_data[0] = frame;
    float * pose_data = (float*)(data + sizeof(int));
    for (int i=0; i<dimensions; i++)
        pose_data[i] = pose[i];

    std::ofstream fstream;
    fstream.open(filename.c_str(),std::ios_base::out | std::ios_base::binary);
    fstream.write(data,filesize);
    fstream.close();

}

void loadState(float * pose, const int dimensions, int & frame, std::string filename) {

    std::ifstream fstream;
    fstream.open(filename.c_str(),std::ios_base::in | std::ios_base::binary);
    fstream.read((char *)&frame,sizeof(int));

    fstream.read((char *)pose,dimensions*sizeof(float));

    fstream.close();

}

LinearPoseReduction * loadLinearPoseReduction(std::string filename) {

    std::ifstream fstream;
    fstream.open(filename.c_str(),std::ios_base::in);

    int fullDimensions;
    int reducedDimensions;

    // read dimensions
    fstream >> fullDimensions;
    fstream >> reducedDimensions;

    float * A = new float[fullDimensions*reducedDimensions];
    float * b = new float[fullDimensions];
    float * mins = new float[reducedDimensions];
    float * maxs = new float[reducedDimensions];

    // read A
    for (int f=0; f<fullDimensions; ++f) {
        for (int r=0; r<reducedDimensions; ++r) {
            fstream >> A[r + f*reducedDimensions];
        }
    }

    // read b
    for (int f=0; f<fullDimensions; ++f) {
        fstream >> b[f];
    }

    // read mins
    for (int r=0; r<reducedDimensions; ++r) {
        fstream >> mins[r];
    }

    // read maxs
    for (int r=0; r<reducedDimensions; ++r) {
        fstream >> maxs[r];
    }

    // read dof names
    std::vector<std::string> names;

    std::string name;
    getline(fstream,name); // clear newline
    for (int r=0; r<reducedDimensions; ++r) {
        getline(fstream,name);
        names.push_back(name);
//        std::cout << r << ": " << name << std::endl;
    }

    fstream.close();

    LinearPoseReduction * reduction = new LinearPoseReduction(fullDimensions,reducedDimensions,A,b,mins,maxs,names.data());

    delete [] A;
    delete [] b;
    delete [] mins;
    delete [] maxs;

    return reduction;

}

ParamMapPoseReduction * loadParamMapPoseReduction(std::string filename) {

    std::ifstream fstream;
    fstream.open(filename.c_str(),std::ios_base::in);

    int fullDimensions;
    int reducedDimensions;

    // read dimensions
    fstream >> fullDimensions;
    fstream >> reducedDimensions;

    // read mapping
    int * mapping = new int[fullDimensions];
    float * mins = new float[reducedDimensions];
    float * maxs = new float[reducedDimensions];

    for (int f=0; f<fullDimensions; ++f) {
        fstream >> mapping[f];
    }

    // read mins
    for (int r=0; r<reducedDimensions; ++r) {
        fstream >> mins[r];
    }

    // read maxs
    for (int r=0; r<reducedDimensions; ++r) {
        fstream >> maxs[r];
    }

    // read dof names
    std::vector<std::string> names;

    std::string name;
    getline(fstream,name); // clear newline
    for (int r=0; r<reducedDimensions; ++r) {
        getline(fstream,name);
        names.push_back(name);
//        std::cout << r << ": " << name << std::endl;
    }

    fstream.close();

    ParamMapPoseReduction * reduction = new ParamMapPoseReduction(fullDimensions,reducedDimensions,mapping,mins,maxs,names.data());

    delete [] mapping;
    delete [] mins;
    delete [] maxs;

    return reduction;

}

//LinearPoseReduction * loadLinearPoseReduction(std::string filename, const int fullDimensions, const int reducedDimensions) {

//    std::ifstream fstream;
//    fstream.open(filename.c_str(),std::ios_base::in);

//    float * A = new float[fullDimensions*reducedDimensions];
//    float * b = new float[fullDimensions];

//    // read A
//    std::string line;
//    for (int f=0; f<fullDimensions; ++f) {
//        std::getline(fstream,line);
//        std::stringstream ss(line);
//        for (int r=0; r<reducedDimensions; ++r) {
//            ss >> A[r + f*reducedDimensions];
//        }
//    }

//    // read b
//    std::getline(fstream,line);
//    std::stringstream ss(line);
//    for (int j=0; j<fullDimensions; ++j) {
//        ss >> b[j];
//    }

//    fstream.close();

//    LinearPoseReduction* reduction; //= new LinearPoseReduction(A,b,fullDimensions,reducedDimensions);

//    delete [] A;
//    delete [] b;

//    return reduction;

//}

int * loadSelfIntersectionMatrix(const std::string filename, const int numSdfs) {

    std::ifstream fstream;
    fstream.open(filename.c_str(),std::ios_base::in);

    int * matrix = new int[numSdfs*numSdfs];

    for (int i=0; i<numSdfs; ++i) {
        for (int j=0; j<numSdfs; ++j) {
            fstream >> matrix[j + i*numSdfs];
        }
    }

    fstream.close();
    return matrix;
}

}
