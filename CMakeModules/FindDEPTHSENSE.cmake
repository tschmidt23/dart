################################################################################
# Find DEPTHSENSE
#
# DEPTHSENSE_FOUND - True if DepthSense was found
# DEPTHSENSE_INCLUDE_DIR - Directories containing the DepthSense include files
# DEPTHSENSE_LIBRARIES - Librarires needed to use DepthSense
#
################################################################################

IF (WIN32)
ELSE ()
    FIND_PATH(DEPTHSENSE_INCLUDE_DIR DepthSense.hxx
        /opt/DepthSenseSDK/include
        /opt/softkinetic/DepthSenseSDK/include
        ${PROJECT_SOURCE_DIR}/external_lib/include
        DOC "The directory where DepthSense.hxx resides"
    )
    FIND_LIBRARY(DEPTHSENSE_LIBRARIES
        NAMES DepthSense
        PATHS
        /opt/DepthSenseSDK/lib/libDepthSense.so
        /opt/softkinetic/DepthSenseSDK/lib/
        ${PROJECT_SOURCE_DIR}/external_lib/lib
        DOC "The DepthSense library"
    )
ENDIF()

IF(DEPTHSENSE_INCLUDE_DIR AND DEPTHSENSE_LIBRARIES)
    SET(DEPTHSENSE_FOUND ON)
    message(STATUS "DepthSense found: ${DEPTHSENSE_INCLUDE_DIR}")
ELSE()
    SET(DEPTHSENSE_FOUND OFF)
ENDIF()

MARK_AS_ADVANCED(DEPTHSENSE_FOUND)
