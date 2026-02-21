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
message(STATUS "OpenCV libs:    ${OpenCV_LIBS}")

# -----------------------------------------------------------------------------
# ONNX Runtime (optional — Task 9 CNN embedding)
#
# Expected layout at ONNXRUNTIME_ROOT:
#   include/onnxruntime_cxx_api.h
#   lib/onnxruntime.lib
#   lib/onnxruntime.dll
# -----------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "C:/lib/onnxruntime" CACHE PATH "ONNX Runtime root dir")
find_package(ONNXRuntime QUIET)

if(ONNXRuntime_FOUND)
    message(STATUS "ONNX Runtime found")
    message(STATUS "  Include : ${ONNXRuntime_INCLUDE_DIRS}")
    message(STATUS "  Libs    : ${ONNXRuntime_LIBRARIES}")
    add_definitions(-DUSE_ONNXRUNTIME)
else()
    message(WARNING "ONNX Runtime not found — CNN embedding (Task 9) disabled.")
endif()

# -----------------------------------------------------------------------------
# GLFW3 (optional — window + OpenGL context backend for Dear ImGui)
#
# GLFW creates the OS window and OpenGL context that ImGui renders into.
# Without it the GUI phase is skipped; all pipeline logic still works.
#
# Expected layout at GLFW_ROOT:
#   include/GLFW/glfw3.h
#   lib-vc2022/glfw3.lib          (MSVC pre-built)
#   -- OR installed via vcpkg --
#
# Install options:
#   vcpkg : vcpkg install glfw3:x64-windows
#   Manual: https://www.glfw.org/download  → extract to C:\lib\glfw
# -----------------------------------------------------------------------------
set(GLFW_ROOT "C:/lib/glfw" CACHE PATH "GLFW root dir (manual install)")

# Let CMake find glfw3 via vcpkg toolchain or manual hint
find_package(glfw3 QUIET HINTS "${GLFW_ROOT}/lib/cmake")

if(NOT glfw3_FOUND)
    # Fallback: locate manually from GLFW_ROOT
    find_library(GLFW_LIB
        NAMES glfw3 glfw
        PATHS "${GLFW_ROOT}/lib-vc2022"
              "${GLFW_ROOT}/lib-vc2019"
              "${GLFW_ROOT}/lib"
    )
    find_path(GLFW_INCLUDE
        NAMES GLFW/glfw3.h
        PATHS "${GLFW_ROOT}/include"
    )
    if(GLFW_LIB AND GLFW_INCLUDE)
        # Wrap into an imported target so the rest of CMake stays uniform
        add_library(glfw3 STATIC IMPORTED)
        set_target_properties(glfw3 PROPERTIES
            IMPORTED_LOCATION             "${GLFW_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE}"
        )
        set(glfw3_FOUND TRUE)
        message(STATUS "GLFW found (manual): ${GLFW_LIB}")
    endif()
endif()

if(glfw3_FOUND)
    message(STATUS "GLFW ready — ImGui GUI will be compiled.")
    add_definitions(-DUSE_IMGUI)
else()
    message(STATUS "GLFW not found — ImGui GUI skipped (pipeline still works).")
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
# Add each task's .cpp here as it is implemented.
# -----------------------------------------------------------------------------
set(PIPELINE_SOURCES
    src/Threshold.cpp
    src/Morphology.cpp            # Task 2
    src/ConnectedComponents.cpp   # Task 3
    src/RegionFeatures.cpp        # Task 4
    src/ObjectDB.cpp              # Task 5
    src/Classifier.cpp            # Task 6
    src/Evaluator.cpp             # Task 7
    src/utilities.cpp             # Task 9 (Modified from Prof. Bruce Maxwell)
    src/Embedding.cpp             # Task 9
    # src/RegionFeatures.cpp      # Task 4
    # src/ObjectDB.cpp            # Task 5
    # src/Classifier.cpp          # Task 6
    # src/Evaluator.cpp           # Task 7
    # src/Embedding.cpp           # Task 9
)

add_library(pipeline STATIC ${PIPELINE_SOURCES})
target_link_libraries(pipeline PUBLIC ${OpenCV_LIBS})

if(ONNXRuntime_FOUND)
    target_include_directories(pipeline PUBLIC ${ONNXRuntime_INCLUDE_DIRS})
    target_link_libraries(pipeline PUBLIC ${ONNXRuntime_LIBRARIES})
endif()

# -----------------------------------------------------------------------------
# Main executable
# -----------------------------------------------------------------------------
add_executable(objectRecognition src/main.cpp)
target_link_libraries(objectRecognition PRIVATE pipeline ${OpenCV_LIBS})

# -----------------------------------------------------------------------------
# Dear ImGui — compiled only when GLFW + imgui sources are present
#
# ImGui source files live in third_party/imgui/ (git clone ocornut/imgui).
# Backends used: GLFW + OpenGL3.
# imgui_stdlib adds std::string support for text input widgets.
# -----------------------------------------------------------------------------
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui")

if(glfw3_FOUND AND EXISTS "${IMGUI_DIR}/imgui.h")
    set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp   # GLFW event handling
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp # OpenGL3 renderer
        ${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp       # std::string InputText
        # src/GUI.cpp                                # uncomment in GUI phase
    )

    add_library(imgui_lib STATIC ${IMGUI_SOURCES})

    target_include_directories(imgui_lib PUBLIC
        ${IMGUI_DIR}                  # imgui.h
        ${IMGUI_DIR}/backends         # imgui_impl_*.h
        ${IMGUI_DIR}/misc/cpp         # imgui_stdlib.h
    )

    # Link GLFW + OpenGL system libraries
    if(WIN32)
        target_link_libraries(imgui_lib PUBLIC glfw3 opengl32)
    else()
        find_package(OpenGL REQUIRED)
        target_link_libraries(imgui_lib PUBLIC glfw OpenGL::GL)
    endif()

    target_link_libraries(objectRecognition PRIVATE imgui_lib)
    message(STATUS "ImGui compiled and linked.")
else()
    if(NOT EXISTS "${IMGUI_DIR}/imgui.h")
        message(STATUS "ImGui sources not found at ${IMGUI_DIR} — skipping.")
        message(STATUS "  Run: git clone https://github.com/ocornut/imgui ${IMGUI_DIR}")
    endif()
endif()

# -----------------------------------------------------------------------------
# Post-build: copy data folder
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/data")
    file(COPY "${CMAKE_SOURCE_DIR}/data/"
         DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data/")
endif()

# Post-build: copy ONNX model (copy_if_different preserves existing data/db files)
if(EXISTS "${CMAKE_SOURCE_DIR}/data/models/resnet18-v2-7.onnx")
    add_custom_command(TARGET objectRecognition POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/data/models/resnet18-v2-7.onnx"
            "$<TARGET_FILE_DIR:objectRecognition>/data/models/resnet18-v2-7.onnx"
        COMMENT "Copying resnet18-v2-7.onnx"
    )
endif()

# Post-build: copy ONNX Runtime DLLs (Windows)
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

# Post-build: copy GLFW DLL if present (dynamic build)
if(WIN32 AND glfw3_FOUND)
    file(GLOB GLFW_DLLS "${GLFW_ROOT}/lib-vc2022/*.dll"
                        "${GLFW_ROOT}/lib/*.dll")
    foreach(DLL ${GLFW_DLLS})
        add_custom_command(TARGET objectRecognition POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DLL}" "$<TARGET_FILE_DIR:objectRecognition>"
            COMMENT "Copying GLFW DLL: ${DLL}"
        )
    endforeach()
endif()

# -----------------------------------------------------------------------------
# Build summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== ObjectRecognition Build Config ===")
message(STATUS "  C++ standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV       : ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime : ${ONNXRuntime_FOUND}")
message(STATUS "  GLFW / ImGui : ${glfw3_FOUND}")
message(STATUS "  Output dir   : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "======================================")