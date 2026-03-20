################################################################################
# CMakeLists.txt - Camera Calibration & Augmented Reality System
# Author: Krushna Sanjay Sharma
# Description: CMake build configuration for AR Calibration system with OpenCV
# Date: March 2026
################################################################################

cmake_minimum_required(VERSION 3.15)
project(ARCalibration VERSION 1.0.0 LANGUAGES CXX)

################################################################################
# Build Configuration
################################################################################

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

################################################################################
# Compiler Flags
################################################################################

if(MSVC)
    # Visual Studio compiler flags
    add_compile_options(/W4 /WX-)          # Warning level 4, warnings not as errors
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)  # Disable secure warnings
else()
    # GCC/Clang compiler flags
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

################################################################################
# Dependencies
################################################################################

# Find OpenCV (required)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "========================================")
message(STATUS "OpenCV Configuration")
message(STATUS "========================================")
message(STATUS "OpenCV version:      ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries:    ${OpenCV_LIBS}")
message(STATUS "========================================")

################################################################################
# Include Directories
################################################################################

include_directories(${CMAKE_SOURCE_DIR}/include)

################################################################################
# Source Files - AR Core Library
#
# Files are added here as each task is implemented.
# Uncomment each entry when the corresponding source file is created.
################################################################################

set(AR_CORE_SOURCES
    # Task 1-3: Camera Calibration
    src/CameraCalibration.cpp

    # Task 4-6: Pose Estimation & Virtual Object
    src/PoseEstimator.cpp
    src/VirtualObject.cpp

    # Multi-target extension: Eagle object + SIFT tracker
    src/VirtualEagleObject.cpp
    src/SIFTTracker.cpp

    # Task 7: Feature Detection
    src/FeatureDetector.cpp

    # Shared Utilities
    # src/Utils.cpp
)

set(AR_CORE_HEADERS
    # Task 1-3: Camera Calibration
    include/CameraCalibration.h

    # Task 4-6: Pose Estimation & Virtual Object
    include/PoseEstimator.h
    include/VirtualObject.h

    # Multi-target extension: Eagle object + SIFT tracker
    include/VirtualEagleObject.h
    include/SIFTTracker.h

    # Task 7: Feature Detection
    include/FeatureDetector.h

    # Shared Utilities
    # include/Utils.h
)

################################################################################
# AR Core Library
#
# Static library containing all reusable AR classes.
# Apps link against this library.
################################################################################

add_library(ar_core STATIC
    ${AR_CORE_SOURCES}
    ${AR_CORE_HEADERS}
)

target_link_libraries(ar_core PUBLIC ${OpenCV_LIBS})
target_include_directories(ar_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

################################################################################
# Application Executables
#
# Executables are added here as each task is implemented.
# Uncomment each entry when the corresponding app file is created.
################################################################################

# Application 1: Camera Calibration (Tasks 1-3)
add_executable(calibrateCamera apps/calibrateCamera.cpp)
target_link_libraries(calibrateCamera ar_core ${OpenCV_LIBS})

# Application 2: Augmented Reality (Tasks 4-6)
add_executable(augmentedReality apps/augmentedReality.cpp)
target_link_libraries(augmentedReality ar_core ${OpenCV_LIBS})

# Application 5: Multi-Target AR (Extension)
add_executable(multiTargetAR apps/multiTargetAR.cpp)
target_link_libraries(multiTargetAR ar_core ${OpenCV_LIBS})

# Application 4: SIFT AR (Uber Extension 2)
add_executable(siftAR apps/siftAR.cpp)
target_link_libraries(siftAR ar_core ${OpenCV_LIBS})

# Application 3: Feature Detector (Task 7)
add_executable(featureDetector apps/featureDetector.cpp)
target_link_libraries(featureDetector ar_core ${OpenCV_LIBS})

################################################################################
# Data Directories
################################################################################

# Create output directories for calibration data.
# Visual Studio is a multi-config generator — it appends Release/Debug to
# CMAKE_RUNTIME_OUTPUT_DIRECTORY automatically, so we must create both.
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data/calibration)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data/calibration)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release/data/calibration)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/data/calibration)

################################################################################
# Build Summary
################################################################################

message(STATUS "")
message(STATUS "========================================")
message(STATUS "AR Calibration System Build Configuration")
message(STATUS "========================================")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "Executables output: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Libraries output:   ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "========================================")
message(STATUS "Core library: ar_core")
message(STATUS "Applications (enabled as tasks are completed):")
message(STATUS "  - calibrateCamera  (Tasks 1-3: detection + calibration)")
message(STATUS "  - augmentedReality (Tasks 4-6: pose + virtual object)")
message(STATUS "  - featureDetector  (Task  7:   feature detection)")
message(STATUS "========================================")
message(STATUS "")
