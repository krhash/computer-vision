@echo off
REM ============================================================================
REM build.bat - Windows Build Script for CBIR System
REM Author: Krushna Sanjay Sharma
REM Description: Builds the Content-Based Image Retrieval System with OpenCV
REM Date: February 2026
REM ============================================================================

echo ========================================
echo CBIR System Build
echo OpenCV Location: C:\lib\build_opencv
echo ========================================
echo.

REM ============================================================================
REM Set OpenCV Paths
REM ============================================================================

set OpenCV_DIR=C:\lib\build_opencv
set OpenCV_INSTALL_DIR=C:\lib\install

echo Checking OpenCV installation...
echo.

REM Check if build_opencv exists
if not exist "%OpenCV_DIR%" (
    echo [ERROR] C:\lib\build_opencv not found
    echo.
    echo Available directories in C:\lib:
    dir C:\lib /AD
    echo.
    echo Please verify the correct build directory.
    pause
    exit /b 1
)

REM ============================================================================
REM Locate OpenCVConfig.cmake
REM ============================================================================

if exist "%OpenCV_DIR%\OpenCVConfig.cmake" (
    echo [OK] Found OpenCVConfig.cmake in build_opencv
) else if exist "%OpenCV_INSTALL_DIR%\x64\*\lib\OpenCVConfig.cmake" (
    echo [OK] Found OpenCVConfig.cmake in install directory
    set OpenCV_DIR=%OpenCV_INSTALL_DIR%
) else if exist "%OpenCV_INSTALL_DIR%\OpenCVConfig.cmake" (
    echo [OK] Found OpenCVConfig.cmake in install directory
    set OpenCV_DIR=%OpenCV_INSTALL_DIR%
) else (
    echo [WARN] Searching for OpenCVConfig.cmake...
    dir /s /b C:\lib\OpenCVConfig.cmake
    echo.
    echo Please check the output above and update OpenCV_DIR
    pause
    exit /b 1
)

echo Using OpenCV from: %OpenCV_DIR%
echo.

REM ============================================================================
REM Create Build Directory
REM ============================================================================

echo Creating build directory...
if not exist build mkdir build
cd build

REM ============================================================================
REM Configure with CMake
REM ============================================================================

echo.
echo ========================================
echo Configuring with CMake...
echo ========================================
echo Command: cmake .. -DOpenCV_DIR=%OpenCV_DIR% -DCMAKE_BUILD_TYPE=Release
echo.

cmake .. -DOpenCV_DIR=%OpenCV_DIR% -DCMAKE_BUILD_TYPE=Release

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo [ERROR] CMake Configuration FAILED
    echo ========================================
    echo.
    echo Troubleshooting:
    echo 1. Check if OpenCVConfig.cmake exists
    echo 2. Try these directories:
    echo    - C:\lib\build_opencv
    echo    - C:\lib\install
    echo.
    echo Manual search command:
    echo    dir /s /b C:\lib\OpenCVConfig.cmake
    echo.
    pause
    exit /b 1
)

REM ============================================================================
REM Build the Project
REM ============================================================================

echo.
echo ========================================
echo Building CBIR System...
echo ========================================
echo.

cmake --build . --config Release

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

REM ============================================================================
REM Find and Report OpenCV DLLs
REM ============================================================================

echo.
echo ========================================
echo Checking for OpenCV DLLs...
echo ========================================
echo.

REM Check common locations
if exist "C:\lib\build_opencv\bin\Release\opencv_world*.dll" (
    set OPENCV_BIN=C:\lib\build_opencv\bin\Release
    echo [OK] Found DLLs in: %OPENCV_BIN%
) else if exist "C:\lib\build_opencv\bin\Debug\opencv_world*.dll" (
    set OPENCV_BIN=C:\lib\build_opencv\bin\Debug
    echo [OK] Found DLLs in: %OPENCV_BIN%
) else if exist "C:\lib\install\x64\*\bin\opencv_world*.dll" (
    set OPENCV_BIN=C:\lib\install\x64\vc16\bin
    echo [OK] Found DLLs in: %OPENCV_BIN%
) else (
    echo [WARN] Searching for DLLs...
    dir /s /b C:\lib\opencv_world*.dll 2>nul
    echo.
    echo If DLLs are found above, note their location
    set OPENCV_BIN=
)

REM ============================================================================
REM Build Success Summary
REM ============================================================================

echo.
echo ========================================
echo Build Completed Successfully!
echo ========================================
echo.
echo Output directories:
echo   Executables: ..\bin\Release\
echo   Libraries:   ..\lib\Release\
echo.

if defined OPENCV_BIN (
    echo To run the executables, ensure OpenCV DLLs are accessible:
    echo.
    echo Option 1 - Add to PATH:
    echo    set PATH=%%PATH%%;%OPENCV_BIN%
    echo.
    echo Option 2 - Copy DLLs to executable directory:
    echo    copy "%OPENCV_BIN%\*.dll" "..\bin\Release\"
    echo.
)

echo ========================================
echo Next Steps:
echo ========================================
echo 1. Add feature implementations
echo 2. Build applications
echo 3. Run CBIR queries
echo.

cd ..
pause
