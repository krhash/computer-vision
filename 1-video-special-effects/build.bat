@echo off
REM Windows Build script with OpenCV in C:\lib
REM Author: Krushna Sanjay Sharma
REM Date: January 24, 2026

echo ===================================
echo Vision Project Build
echo OpenCV Location: C:\lib\build_opencv
echo ===================================
echo.

REM Set OpenCV paths
set OpenCV_DIR=C:\lib\build_opencv
set OpenCV_INSTALL_DIR=C:\lib\install

echo Checking OpenCV installation...
echo.

REM Check if build_opencv exists
if not exist "%OpenCV_DIR%" (
    echo Error: C:\lib\build_opencv not found
    echo.
    echo Available directories in C:\lib:
    dir C:\lib /AD
    echo.
    echo Please verify the correct build directory.
    pause
    exit /b 1
)

REM Look for OpenCVConfig.cmake
if exist "%OpenCV_DIR%\OpenCVConfig.cmake" (
    echo [OK] Found OpenCVConfig.cmake in build_opencv
) else if exist "%OpenCV_INSTALL_DIR%\x64\*\lib\OpenCVConfig.cmake" (
    echo [OK] Found OpenCVConfig.cmake in install directory
    set OpenCV_DIR=%OpenCV_INSTALL_DIR%
) else if exist "%OpenCV_INSTALL_DIR%\OpenCVConfig.cmake" (
    echo [OK] Found OpenCVConfig.cmake in install directory
    set OpenCV_DIR=%OpenCV_INSTALL_DIR%
) else (
    echo Searching for OpenCVConfig.cmake...
    dir /s /b C:\lib\OpenCVConfig.cmake
    echo.
    echo Please check the output above and update OpenCV_DIR
    pause
    exit /b 1
)

echo Using OpenCV from: %OpenCV_DIR%
echo.

REM Create build directory
echo Creating build directory...
if not exist build mkdir build
cd build

REM Configure with CMake
echo.
echo Configuring with CMake...
echo Command: cmake .. -DOpenCV_DIR=%OpenCV_DIR% -DCMAKE_BUILD_TYPE=Release
echo.

cmake .. -DOpenCV_DIR=%OpenCV_DIR% -DCMAKE_BUILD_TYPE=Release

if %errorlevel% neq 0 (
    echo.
    echo ===================================
    echo CMake Configuration FAILED
    echo ===================================
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

REM Build the project
echo.
echo ===================================
echo Building project...
echo ===================================
cmake --build . --config Release

if %errorlevel% neq 0 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

REM Find OpenCV DLLs
echo.
echo Checking for OpenCV DLLs...

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
    dir /s /b C:\lib\opencv_world*.dll
    echo.
    echo If DLLs are found above, note their location
    set OPENCV_BIN=
)

REM Success
echo.
echo ===================================
echo Build completed successfully!
echo ===================================
echo.
echo Executables are in: ..\bin\Release\
echo.

if defined OPENCV_BIN (
    echo To run the executables, add OpenCV to PATH:
    echo    set PATH=%%PATH%%;%OPENCV_BIN%
    echo.
    echo Or copy DLLs to executable directory:
    echo    copy "%OPENCV_BIN%\*.dll" "..\bin\Release\"
    echo.
)

echo Run the applications:
echo    cd ..\bin\Release
echo    imgDisplay.exe ..\..\data\images\test.jpg
echo    vidDisplay.exe
echo.

cd ..
pause