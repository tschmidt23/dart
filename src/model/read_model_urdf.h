#ifndef READ_MODEL_URDF_H
#define READ_MODEL_URDF_H

#include <dart/model/host_only_model.h>

namespace dart {

/**
 * @brief readModelURDF parse URDF model description and store kinematic and meshes in DART model format
 * @param path relative or absolute path to URDF file
 * @param model DART model that will be extended with frames from URDF model
 * @param root_link_name optional root link name from where to start, if not provided the root link will be determined automatically
 * @param mesh_extension_surrogate optional file extension to load meshes from different file types as provided in the URDF model description
 * @return true on success
 * @return false on failure
 */
bool readModelURDF(const std::string path, HostOnlyModel & model, const std::string root_link_name = "", const std::string mesh_extension_surrogate = "");

} // namespace dart

#endif // READ_MODEL_URDF_H
