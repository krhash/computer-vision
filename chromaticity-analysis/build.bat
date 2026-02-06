@echo off
REM Windows Build script for Chromaticity Analysis
REM OpenCV Location: C:\lib\build_opencv

echo ===================================
echo Chromaticity Analysis Build
echo ===================================
echo.

REM Set OpenCV paths
set OpenCV_DIR=C:\lib\build_opencv

echo Checking OpenCV installation...

if not exist "%OpenCV_DIR%" (
    echo Error: C:\lib\build_opencv not found
    pause
    exit /b 1
)

echo [OK] Found OpenCV at: %OpenCV_DIR%
echo.

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -DOpenCV_DIR=%OpenCV_DIR% -DCMAKE_BUILD_TYPE=Release

if %errorlevel% neq 0 (
    echo CMake Configuration FAILED
    pause
    exit /b 1
)

REM Build the project
echo.
echo Building project...
cmake --build . --config Release

if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ===================================
echo Build completed successfully!
echo ===================================
echo.
echo Executable location: ..\bin\Release\chromaticity_analysis.exe
echo.
echo Usage:
echo    cd ..\bin\Release
echo    chromaticity_analysis.exe ..\..\data\shadow.jpg
echo.

cd ..
pause
