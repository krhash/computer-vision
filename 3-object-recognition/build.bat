@echo off
REM =============================================================================
REM  build.bat — Windows build script for ObjectRecognition
REM  Author:  Krushna Sanjay Sharma
REM  Date:    February 2026
REM =============================================================================

echo =============================================
echo  2D Object Recognition — Build Script
echo  OpenCV : C:\lib\build_opencv
echo  ONNX   : C:\lib\onnxruntime  (optional)
echo  GLFW   : C:\lib\glfw         (optional)
echo =============================================
echo.

REM --- OpenCV ------------------------------------------------------------------
set OpenCV_DIR=C:\lib\build_opencv

if not exist "%OpenCV_DIR%\OpenCVConfig.cmake" (
    echo [ERROR] OpenCVConfig.cmake not found in %OpenCV_DIR%
    echo Run:  dir /s /b C:\lib\OpenCVConfig.cmake
    pause & exit /b 1
)
echo [OK] OpenCV: %OpenCV_DIR%

REM --- ONNX Runtime (optional) -------------------------------------------------
set ONNX_DIR=C:\lib\onnxruntime
if exist "%ONNX_DIR%" (
    echo [OK] ONNX Runtime: %ONNX_DIR%
) else (
    echo [SKIP] ONNX Runtime not found — Task 9 CNN embedding disabled.
)

REM --- GLFW (optional, for ImGui phase) ----------------------------------------
set GLFW_DIR=C:\lib\glfw
if exist "%GLFW_DIR%" (
    echo [OK] GLFW: %GLFW_DIR%
) else (
    echo [SKIP] GLFW not found — GUI phase will be skipped.
)

echo.

REM --- Create build dir --------------------------------------------------------
if not exist build mkdir build
cd build

REM --- CMake configure ---------------------------------------------------------
echo Configuring...
cmake .. ^
    -DOpenCV_DIR=%OpenCV_DIR% ^
    -DONNXRUNTIME_ROOT=%ONNX_DIR% ^
    -DGLFW_ROOT=%GLFW_DIR% ^
    -DCMAKE_BUILD_TYPE=Release

if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed.
    pause & exit /b 1
)

REM --- Build -------------------------------------------------------------------
echo.
echo Building...
cmake --build . --config Release

if %errorlevel% neq 0 (
    echo [ERROR] Build failed.
    pause & exit /b 1
)

REM --- Locate and report OpenCV DLLs ------------------------------------------
echo.
if exist "C:\lib\build_opencv\bin\Release\opencv_world*.dll" (
    set OPENCV_BIN=C:\lib\build_opencv\bin\Release
) else if exist "C:\lib\install\x64\vc16\bin\opencv_world*.dll" (
    set OPENCV_BIN=C:\lib\install\x64\vc16\bin
) else (
    set OPENCV_BIN=
    echo [WARN] OpenCV DLLs not auto-detected. Add them to PATH manually.
)

echo =============================================
echo  Build SUCCESS
echo  Executable : ..\bin\Release\objectRecognition.exe
echo =============================================
echo.

if defined OPENCV_BIN (
    echo Add OpenCV to PATH:
    echo   set PATH=%%PATH%%;%OPENCV_BIN%
    echo Or copy DLLs:
    echo   copy "%OPENCV_BIN%\*.dll" "..\bin\Release\"
    echo.
)

echo Usage:
echo   objectRecognition.exe --mode live   --camera 0
echo   objectRecognition.exe --mode image  --input ..\data\images\test.jpg
echo   objectRecognition.exe --mode train
echo   objectRecognition.exe --mode eval   --input ..\data\images\
echo.

cd ..
pause
