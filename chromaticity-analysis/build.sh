#!/bin/bash
# Linux/Mac Build script for Chromaticity Analysis

echo "==================================="
echo "Chromaticity Analysis Build"
echo "==================================="
echo

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "CMake Configuration FAILED"
    exit 1
fi

# Build the project
echo
echo "Building project..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo
echo "==================================="
echo "Build completed successfully!"
echo "==================================="
echo
echo "Executable location: ../bin/chromaticity_analysis"
echo
echo "Usage:"
echo "   cd ../bin"
echo "   ./chromaticity_analysis ../data/shadow.jpg"
echo
