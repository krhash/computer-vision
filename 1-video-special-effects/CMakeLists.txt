cmake_minimum_required(VERSION 3.15)
project(VisionProject VERSION 1.0.0 LANGUAGES CXX)

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

# Compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find OpenCV (required)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Find ONNX Runtime (optional for Task 10)
# Set path hint for Windows installation
set(ONNXRUNTIME_ROOT "C:/lib/onnxruntime" CACHE PATH "ONNX Runtime root")
find_package(ONNXRuntime)

if(ONNXRuntime_FOUND)
    message(STATUS "ONNX Runtime found!")
    message(STATUS "  Include: ${ONNXRuntime_INCLUDE_DIRS}")
    message(STATUS "  Library: ${ONNXRuntime_LIBRARIES}")
    add_definitions(-DUSE_ONNXRUNTIME)
else()
    message(WARNING "ONNX Runtime not found. Depth estimation features will be disabled.")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files for filters library
set(FILTER_SOURCES
    src/filters.cpp
    src/faceDetect.cpp
)

# Create filters library
add_library(filters STATIC ${FILTER_SOURCES})
target_link_libraries(filters ${OpenCV_LIBS})

if(ONNXRuntime_FOUND)
    target_include_directories(filters PUBLIC ${ONNXRuntime_INCLUDE_DIRS})
    target_link_libraries(filters ${ONNXRuntime_LIBRARIES})
endif()

# Task 1: Image Display
add_executable(imgDisplay src/imgDisplay.cpp)
target_link_libraries(imgDisplay ${OpenCV_LIBS})

# Task 2+: Video Display with all filters
add_executable(vidDisplay src/vidDisplay.cpp)
target_link_libraries(vidDisplay filters ${OpenCV_LIBS})

# Copy ONNX Runtime DLLs to executable directory (Windows)
if(WIN32 AND ONNXRuntime_FOUND)
    # Find all ONNX Runtime DLLs
    file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll")
    
    foreach(DLL_FILE ${ONNXRUNTIME_DLLS})
        add_custom_command(
            TARGET vidDisplay POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL_FILE}
                $<TARGET_FILE_DIR:vidDisplay>
            COMMENT "Copying ${DLL_FILE}"
        )
    endforeach()
endif()

# Copy cuDNN DLLs to executable directory (Windows)
if(WIN32 AND EXISTS "C:/lib/cudnn/bin/x64")
    file(GLOB CUDNN_DLLS "C:/lib/cudnn/bin/x64/*.dll")
    
    foreach(DLL_FILE ${CUDNN_DLLS})
        add_custom_command(
            TARGET vidDisplay POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL_FILE}
                $<TARGET_FILE_DIR:vidDisplay>
            COMMENT "Copying cuDNN: ${DLL_FILE}"
        )
    endforeach()
endif()

# Copy data files to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data/)

# Copy Haar Cascade XML file to executable directory
if(EXISTS ${CMAKE_SOURCE_DIR}/data/haarcascade_frontalface_alt2.xml)
    if(MSVC)
        add_custom_command(
            TARGET vidDisplay POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_SOURCE_DIR}/data/haarcascade_frontalface_alt2.xml
                $<TARGET_FILE_DIR:vidDisplay>/haarcascade_frontalface_alt2.xml
            COMMENT "Copying Haar Cascade XML to executable directory"
        )
    else()
        configure_file(
            ${CMAKE_SOURCE_DIR}/data/haarcascade_frontalface_alt2.xml
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/haarcascade_frontalface_alt2.xml
            COPYONLY
        )
    endif()
    message(STATUS "Haar Cascade XML will be copied during build")
else()
    message(WARNING "haarcascade_frontalface_alt2.xml not found in data/")
endif()

# Copy DA2 ONNX model to executable directory (if exists)
if(EXISTS ${CMAKE_SOURCE_DIR}/data/model_fp16.onnx)
    if(MSVC)
        add_custom_command(
            TARGET vidDisplay POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_SOURCE_DIR}/data/model_fp16.onnx
                $<TARGET_FILE_DIR:vidDisplay>/model_fp16.onnx
            COMMENT "Copying DA2 ONNX model to executable directory"
        )
    else()
        configure_file(
            ${CMAKE_SOURCE_DIR}/data/model_fp16.onnx
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/model_fp16.onnx
            COPYONLY
        )
    endif()
    message(STATUS "DA2 ONNX model (model_fp16.onnx) will be copied during build")
else()
    message(WARNING "model_fp16.onnx not found in data/ - depth estimation will not work")
endif()

# Print build configuration
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Executables output: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "ONNX Runtime: ${ONNXRuntime_FOUND}")
message(STATUS "Image Display: imgDisplay")
message(STATUS "Video Display: vidDisplay")
message(STATUS "===========================")