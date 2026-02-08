################################################################################
# CMakeLists.txt - Content-Based Image Retrieval System
# Author: Krushna Sanjay Sharma
# Description: CMake build configuration for CBIR system with OpenCV
# Date: February 2026
################################################################################

cmake_minimum_required(VERSION 3.15)
project(CBIRSystem VERSION 1.0.0 LANGUAGES CXX)

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
    add_compile_options(/W4 /WX-)  # Warning level 4, warnings not as errors
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
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "========================================")

################################################################################
# Include Directories
################################################################################

include_directories(${CMAKE_SOURCE_DIR}/include)

################################################################################
# Source Files - CBIR Core Library
################################################################################

set(CBIR_CORE_SOURCES
    # Base classes
    src/FeatureExtractor.cpp
    src/DistanceMetric.cpp
    
    # Feature extractors
    src/BaselineFeature.cpp
    src/HistogramFeature.cpp
    src/MultiHistogramFeature.cpp
    src/TextureColorFeature.cpp
    src/GaborTextureColorFeature.cpp
    src/MultiRegionHistogramIntersection.cpp
    
    # Distance metrics
    src/SSDMetric.cpp
    src/HistogramIntersection.cpp
    src/WeightedHistogramIntersection.cpp
    src/MultiRegionHistogramIntersection.cpp
    
    # Database and retrieval
    src/FeatureDatabase.cpp
    src/ImageRetrieval.cpp
    
    # Utilities
    src/Utils.cpp
)

set(CBIR_CORE_HEADERS
    include/FeatureExtractor.h
    include/DistanceMetric.h
    include/BaselineFeature.h.
    include/HistogramFeature.h
    include/MultiHistogramFeature.h
    include/TextureColorFeature.h
    include/GaborTextureColorFeature.h
    include/SSDMetric.h
    include/HistogramIntersection.h
    include/WeightedHistogramIntersection.h
    include/MultiRegionHistogramIntersection.h
    include/FeatureDatabase.h
    include/ImageRetrieval.h
    include/Utils.h
)

################################################################################
# CBIR Core Library
################################################################################

add_library(cbir_core STATIC 
    ${CBIR_CORE_SOURCES}
    ${CBIR_CORE_HEADERS}
)

target_link_libraries(cbir_core PUBLIC ${OpenCV_LIBS})
target_include_directories(cbir_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

################################################################################
# Application Executables
################################################################################

# Application 1: Build Feature Database
add_executable(buildFeatureDB apps/buildFeatureDB.cpp)
target_link_libraries(buildFeatureDB cbir_core ${OpenCV_LIBS})

# Application 2: Query Image
add_executable(queryImage apps/queryImage.cpp)
target_link_libraries(queryImage cbir_core ${OpenCV_LIBS})

################################################################################
# Copy Data Files
################################################################################

# Create data directories in build output
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data/features)

# Copy image database if it exists
if(EXISTS ${CMAKE_SOURCE_DIR}/data/images)
    file(COPY ${CMAKE_SOURCE_DIR}/data/images 
         DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data/)
    message(STATUS "Image database will be copied to build directory")
endif()

################################################################################
# Build Summary
################################################################################

message(STATUS "")
message(STATUS "========================================")
message(STATUS "CBIR System Build Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Executables output: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Libraries output: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "========================================")
message(STATUS "Core library: cbir_core")
message(STATUS "Applications:")
message(STATUS "  - buildFeatureDB (Build feature database)")
message(STATUS "  - queryImage (Query system)")
message(STATUS "========================================")
message(STATUS "")
