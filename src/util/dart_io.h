#ifndef DART_IO_H
#define DART_IO_H

#include <string>
#include "model/host_only_model.h"
#include "pose/pose.h"

namespace dart {

void writeModelXML(const HostOnlyModel & model, const char * filename);
bool readModelXML(const char * filename, HostOnlyModel & model);

// TODO: maybe make this look nicer?
void saveState(const float * pose, const int dimensions, const int frame, std::string filename);
void loadState(float * pose, const int dimensions, int & frame, std::string filename);

LinearPoseReduction * loadLinearPoseReduction(std::string filename);

ParamMapPoseReduction * loadParamMapPoseReduction(std::string filename);

//// TODO: put the number of dimensions in the file?
///**
// * @brief loadLinearPoseReduction Loads a LinearPoseReduction instance from a text file. The returned pointer is owned by the caller and will therefore need to be deleted by the caller.
// * @param filename The name of the file storing the LinearPoseReduction data.
// * @param fullDimensions The dimensionality of the space the reduction projects to.
// * @param reducedDimensions The dimensionality of the reduced pose space.
// * @return A pointer to the newly allocated LinearPoseReduction instance.
// */
//LinearPoseReduction * loadLinearPoseReduction(std::string filename, const int fullDimensions, const int reducedDimensions);

int * loadSelfIntersectionMatrix(const std::string filename, const int numSdfs);

}

#endif // DART_IO_H
