cmake_minimum_required(VERSION 3.15)
project(ObjectRecognition VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# Project: 2D Object Recognition System
# Author:  Krushna Sanjay Sharma
# Date:    February 2026
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# -----------------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------------
if(MSVC)
    add_compile_options(/W4 /MP)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------------------------------------------------------
# OpenCV (required)
# -----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# -----------------------------------------------------------------------------
# ONNX Runtime (optional)
# -----------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "C:/lib/onnxruntime" CACHE PATH "ONNX Runtime root dir")
find_package(ONNXRuntime QUIET)
if(ONNXRuntime_FOUND)
    message(STATUS "ONNX Runtime: ${ONNXRuntime_LIBRARIES}")
    add_definitions(-DUSE_ONNXRUNTIME)
else()
    message(WARNING "ONNX Runtime not found — CNN embedding disabled.")
endif()

# -----------------------------------------------------------------------------
# GLFW (manual install at C:/lib/glfw)
# Explicitly locate lib and header — do not rely on find_package
# -----------------------------------------------------------------------------
set(GLFW_ROOT "C:/lib/glfw" CACHE PATH "GLFW root dir")

find_library(GLFW_LIB
    NAMES glfw3 glfw
    PATHS
        "${GLFW_ROOT}/lib-vc2022"
        "${GLFW_ROOT}/lib-vc2019"
        "${GLFW_ROOT}/lib"
    NO_DEFAULT_PATH
)

find_path(GLFW_INCLUDE_DIR
    NAMES GLFW/glfw3.h
    PATHS "${GLFW_ROOT}/include"
    NO_DEFAULT_PATH
)

if(GLFW_LIB AND GLFW_INCLUDE_DIR)
    message(STATUS "GLFW lib    : ${GLFW_LIB}")
    message(STATUS "GLFW include: ${GLFW_INCLUDE_DIR}")
    set(GLFW_FOUND TRUE)
else()
    message(STATUS "GLFW not found at ${GLFW_ROOT} — GUI disabled.")
    set(GLFW_FOUND FALSE)
endif()

# -----------------------------------------------------------------------------
# ImGui detection
# -----------------------------------------------------------------------------
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui")
if(GLFW_FOUND AND EXISTS "${IMGUI_DIR}/imgui.h")
    set(IMGUI_FOUND TRUE)
    message(STATUS "ImGui found : ${IMGUI_DIR}")
else()
    set(IMGUI_FOUND FALSE)
    message(STATUS "ImGui not found — GUI disabled.")
endif()

# -----------------------------------------------------------------------------
# Include directories
# -----------------------------------------------------------------------------
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
if(ONNXRuntime_FOUND)
    include_directories(${ONNXRuntime_INCLUDE_DIRS})
endif()

# -----------------------------------------------------------------------------
# Pipeline sources
# -----------------------------------------------------------------------------
set(PIPELINE_SOURCES
    src/Threshold.cpp
    src/Morphology.cpp
    src/ConnectedComponents.cpp
    src/RegionFeatures.cpp
    src/ObjectDB.cpp
    src/Classifier.cpp
    src/Evaluator.cpp
    src/utilities.cpp
    src/Embedding.cpp
    src/EmbeddingPlot.cpp
)

if(IMGUI_FOUND)
    list(APPEND PIPELINE_SOURCES src/GUI.cpp)
    add_definitions(-DUSE_IMGUI)
endif()

add_library(pipeline STATIC ${PIPELINE_SOURCES})
target_link_libraries(pipeline PUBLIC ${OpenCV_LIBS})

if(ONNXRuntime_FOUND)
    target_include_directories(pipeline PUBLIC ${ONNXRuntime_INCLUDE_DIRS})
    target_link_libraries(pipeline PUBLIC ${ONNXRuntime_LIBRARIES})
endif()

# -----------------------------------------------------------------------------
# ImGui library (pure ImGui sources — no project headers)
# -----------------------------------------------------------------------------
if(IMGUI_FOUND)
    set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
        ${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp
    )

    add_library(imgui_lib STATIC ${IMGUI_SOURCES})
    target_include_directories(imgui_lib PUBLIC
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${IMGUI_DIR}/misc/cpp
        ${GLFW_INCLUDE_DIR}
    )
    target_link_libraries(imgui_lib PUBLIC ${GLFW_LIB} opengl32)

    # Give pipeline access to ImGui headers (needed by GUI.cpp)
    target_include_directories(pipeline PUBLIC
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${IMGUI_DIR}/misc/cpp
        ${GLFW_INCLUDE_DIR}
    )
    target_link_libraries(pipeline PUBLIC imgui_lib)
    message(STATUS "ImGui + GLFW compiled and linked.")
endif()

# -----------------------------------------------------------------------------
# Main executable
# -----------------------------------------------------------------------------
add_executable(objectRecognition src/main.cpp)
target_link_libraries(objectRecognition PRIVATE pipeline ${OpenCV_LIBS})
if(ONNXRuntime_FOUND)
    target_link_libraries(objectRecognition PRIVATE ${ONNXRuntime_LIBRARIES})
endif()

# -----------------------------------------------------------------------------
# Post-build: copy data folder
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/data")
    file(COPY "${CMAKE_SOURCE_DIR}/data/"
         DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data/")
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/data/models/resnet18-v2-7.onnx")
    add_custom_command(TARGET objectRecognition POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/data/models/resnet18-v2-7.onnx"
            "$<TARGET_FILE_DIR:objectRecognition>/data/models/resnet18-v2-7.onnx"
        COMMENT "Copying resnet18-v2-7.onnx"
    )
endif()

if(WIN32 AND ONNXRuntime_FOUND)
    file(GLOB ONNX_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll")
    foreach(DLL ${ONNX_DLLS})
        add_custom_command(TARGET objectRecognition POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DLL}" "$<TARGET_FILE_DIR:objectRecognition>"
            COMMENT "Copying ONNX DLL: ${DLL}"
        )
    endforeach()
endif()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== ObjectRecognition Build Config ===")
message(STATUS "  C++ standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV       : ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime : ${ONNXRuntime_FOUND}")
message(STATUS "  GLFW found   : ${GLFW_FOUND}")
message(STATUS "  ImGui found  : ${IMGUI_FOUND}")
message(STATUS "  Output dir   : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "======================================")
