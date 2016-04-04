find_package(PkgConfig QUIET)

# Find LibUSB
if(NOT WIN32)
  pkg_check_modules(PC_USB_10 libusb-1.0)
  find_path(USB_10_INCLUDE_DIR libusb-1.0/libusb.h
            HINTS ${PC_USB_10_INCLUDEDIR} ${PC_USB_10_INCLUDE_DIRS} "${USB_10_ROOT}" "$ENV{USB_10_ROOT}"
            PATH_SUFFIXES libusb-1.0)

  find_library(USB_10_LIBRARY
               NAMES usb-1.0
               HINTS ${PC_USB_10_LIBDIR} ${PC_USB_10_LIBRARY_DIRS} "${USB_10_ROOT}" "$ENV{USB_10_ROOT}"
               PATH_SUFFIXES lib)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(USB_10 DEFAULT_MSG USB_10_LIBRARY USB_10_INCLUDE_DIR)

  if(NOT USB_10_FOUND)
    message(STATUS "OpenNI 2 disabled because libusb-1.0 not found.")
    return()
  else()
    include_directories(SYSTEM ${USB_10_INCLUDE_DIR})
  endif()
endif(NOT WIN32)

if(${CMAKE_VERSION} VERSION_LESS 2.8.2)
  pkg_check_modules(PC_OPENNI2 libopenni2)
else()
  pkg_check_modules(PC_OPENNI2 QUIET libopenni2)
endif()

set(OPENNI2_DEFINITIONS ${PC_OPENNI_CFLAGS_OTHER})

set(OPENNI2_SUFFIX)
if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(OPENNI2_SUFFIX 64)
endif(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)

find_path(OPENNI2_INCLUDE_DIRS OpenNI.h
    PATHS
    "$ENV{OPENNI2_INCLUDE${OPENNI2_SUFFIX}}"  # Win64 needs '64' suffix
    /usr/include/openni2  # common path for deb packages
)

find_library(OPENNI2_LIBRARY
             NAMES OpenNI2  # No suffix needed on Win64
             libOpenNI2     # Linux
             PATHS "$ENV{OPENNI2_LIB${OPENNI2_SUFFIX}}"  # Windows default path, Win64 needs '64' suffix
             "$ENV{OPENNI2_REDIST}"                      # Linux install does not use a separate 'lib' directory
             )

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(OPENNI2_LIBRARIES ${OPENNI2_LIBRARY} ${LIBUSB_1_LIBRARIES})
else()
  set(OPENNI2_LIBRARIES ${OPENNI2_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenNI2 DEFAULT_MSG OPENNI2_LIBRARY OPENNI2_INCLUDE_DIRS)

mark_as_advanced(OPENNI2_LIBRARY OPENNI2_INCLUDE_DIRS)

if(OPENNI2_FOUND)
  # Add the include directories
  set(OPENNI2_INCLUDE_DIRS ${OPENNI2_INCLUDE_DIR})
  set(OPENNI2_REDIST_DIR $ENV{OPENNI2_REDIST${OPENNI2_SUFFIX}})
  message(STATUS "OpenNI 2 found (include: ${OPENNI2_INCLUDE_DIRS}, lib: ${OPENNI2_LIBRARY}, redist: ${OPENNI2_REDIST_DIR})")
endif(OPENNI2_FOUND)
#find_path(OpenNI2_INCLUDE_DIR OpenNI.h
#    /usr/local/OpenNI-2.2/Include
#    /usr/local/OpenNI-2.1/Include
    
#)
#find_path(OpenNI2_LIBRARY_PATH libOpenNI2.so
#    /usr/local/OpenNI-2.2/Redist
#    /usr/local/OpenNI-2.1/Redist
#)
#if(OpenNI2_INCLUDE_DIR)
#    set(OpenNI2_FOUND ON)
#    set(OpenNI2_LIBRARIES libOpenNI2.so libOniFile.so libPS1080.so)
#    set(OpenNI2_LIBRARY_COPY_PATH ${OpenNI2_LIBRARY_PATH}/OpenNI2)
#    set(OpenNI2_DRIVER_LIBRARY_PATH ${OpenNI2_LIBRARY_PATH}/OpenNI2/Drivers)
#endif()
