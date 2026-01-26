/*
  Author: Krushna Sanjay Sharma
  Date: January 24, 2026
  Purpose: Capture and display live video from camera with various filters.
           Main application for Tasks 2-12 plus Extensions.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <ctime>
#include <chrono>
#include "filters.hpp"
#include "faceDetect.h"

#ifdef USE_ONNXRUNTIME
#include "DA2Network.hpp"
#endif

using namespace cv;
using namespace std;
using namespace chrono;

// Display mode enumeration
enum DisplayMode {
    MODE_COLOR,           // Normal color video (Task 2)
    MODE_GREYSCALE_CV,    // OpenCV greyscale (Task 3)
    MODE_GREYSCALE_CUSTOM,// Custom greyscale (Task 4)
    MODE_SEPIA,           // Sepia tone (Task 5)
    MODE_BLUR,            // Blur filter (Task 6)
    MODE_SOBEL_X,         // Sobel X (Task 7)
    MODE_SOBEL_Y,         // Sobel Y (Task 7)
    MODE_MAGNITUDE,       // Gradient magnitude (Task 8)
    MODE_QUANTIZE,        // Blur and quantize (Task 9)
    MODE_FACE_DETECT,     // Face detection (Task 10)
    MODE_DEPTH,           // Depth estimation (Task 11)
    MODE_DEPTH_FOG,       // Depth Fog (Task 12)
    MODE_EMBOSS,          // Emboss (Task 12)
    MODE_NEGATIVE,        // Negative (Task 12)
    MODE_FACE_HIGHLIGHT,  // Face Highlight (Task 12)
    MODE_CARTOON,         // Cartoon (Task 12)
    MODE_DEPTH_FOCUS,     // Depth Focus (Task 12)
    MODE_BULGE,           // Bulge warp (Extension)
    MODE_WAVE,            // Wave warp (Extension)
    MODE_SWIRL,           // Swirl warp (Extension)
    MODE_FACE_BULGE,      // Face Bulge warp (Extension)
    MODE_SPARKLES         // Sparkles around face (Extension)
};

/**
 * @brief Converts display mode enum to human-readable string.
 * 
 * @param mode The DisplayMode enum value to convert
 * @return String representation of the mode for display/logging
 */
string getModeString(DisplayMode mode) {
    switch (mode) {
        case MODE_COLOR: return "Color";
        case MODE_GREYSCALE_CV: return "Greyscale (OpenCV)";
        case MODE_GREYSCALE_CUSTOM: return "Greyscale (Custom)";
        case MODE_SEPIA: return "Sepia Tone";
        case MODE_BLUR: return "Blur";
        case MODE_SOBEL_X: return "Sobel X";
        case MODE_SOBEL_Y: return "Sobel Y";
        case MODE_MAGNITUDE: return "Gradient Magnitude";
        case MODE_QUANTIZE: return "Blur & Quantize";
        case MODE_FACE_DETECT: return "Face Detection";
        case MODE_DEPTH: return "Depth Estimation";
        case MODE_DEPTH_FOG: return "Depth Fog";
        case MODE_EMBOSS: return "Emboss";
        case MODE_NEGATIVE: return "Negative";
        case MODE_FACE_HIGHLIGHT: return "Face Highlight";
        case MODE_CARTOON: return "Cartoon";
        case MODE_DEPTH_FOCUS: return "Depth Focus";
        case MODE_BULGE: return "Bulge Warp";
        case MODE_WAVE: return "Wave Warp";
        case MODE_SWIRL: return "Swirl Warp";
        case MODE_FACE_BULGE: return "Face Bulge";
        case MODE_SPARKLES: return "Sparkles";
        default: return "Unknown";
    }
}

/**
 * @brief Displays comprehensive keyboard control information to user.
 * 
 * Prints a formatted help menu showing all available keyboard commands
 * for controlling the video display application, including mode switches,
 * adjustments, and utility functions.
 */
void printControls() {
    cout << "\n=== Video Display Controls ===" << endl;
    cout << "  q/ESC : Quit application" << endl;
    cout << "  s     : Save current frame" << endl;
    cout << "  i     : Display video information" << endl;
    cout << "  p     : Show this help" << endl;
    cout << "\n--- Display Modes ---" << endl;
    cout << "  c     : Color mode (default)" << endl;
    cout << "  g     : OpenCV Greyscale (Task 3)" << endl;
    cout << "  h     : Custom Greyscale (Task 4)" << endl;
    cout << "  t     : Sepia Tone (Task 5)" << endl;
    cout << "  b     : Blur filter (Task 6)" << endl;
    cout << "  x     : Sobel X - vertical edges (Task 7)" << endl;
    cout << "  y     : Sobel Y - horizontal edges (Task 7)" << endl;
    cout << "  m     : Gradient Magnitude (Task 8)" << endl;
    cout << "  l     : Blur & Quantize (Task 9)" << endl;
    cout << "  f     : Face Detection (Task 10)" << endl;
    cout << "  d     : Depth Estimation (Task 11)" << endl;
    cout << "\n--- Custom Effects (Task 12) ---" << endl;
    cout << "  1     : Depth Fog - atmospheric depth effect" << endl;
    cout << "  2     : Emboss - 3D relief effect" << endl;
    cout << "  3     : Negative - color inversion" << endl;
    cout << "  4     : Face Highlight - color face, grey background" << endl;
    cout << "  5     : Cartoon - edge + quantization" << endl;
    cout << "  6     : Depth Focus - portrait mode blur" << endl;
    cout << "\n--- Extensions ---" << endl;
    cout << "  7     : Bulge - fisheye distortion" << endl;
    cout << "  8     : Wave - ripple effect" << endl;
    cout << "  9     : Swirl - twirl distortion" << endl;
    cout << "  0     : Face Bulge - caricature effect" << endl;
    cout << "  [     : Sparkles - magical effect around face" << endl;
    cout << "\n--- Adjustments ---" << endl;
    cout << "  +/-   : Adjust effect strength / quantize levels" << endl;
    cout << "==============================\n" << endl;
}

/**
 * @brief Displays detailed information about the video stream and statistics.
 * 
 * @param frame Current video frame for size/type information
 * @param frameCount Total number of frames captured since start
 * @param savedCount Number of frames saved to disk
 * @param fps Camera frames per second
 * @param mode Current display mode
 */
void displayVideoInfo(const Mat& frame, int frameCount, int savedCount, 
                      double fps, DisplayMode mode) {
    cout << "\n=== Video Information ===" << endl;
    cout << "Current mode: " << getModeString(mode) << endl;
    cout << "Frame size: " << frame.cols << " x " << frame.rows << " pixels" << endl;
    cout << "Channels: " << frame.channels() << endl;
    cout << "Type: " << frame.type() << endl;
    cout << "FPS (camera): " << fps << endl;
    cout << "Frames captured: " << frameCount << endl;
    cout << "Frames saved: " << savedCount << endl;
    cout << "Size per frame: " << (frame.total() * frame.elemSize() / 1024.0) << " KB" << endl;
    cout << "========================\n" << endl;
}

/**
 * @brief Processes a video frame by applying the selected filter/effect.
 * 
 * Central function that applies the appropriate image processing effect based
 * on the current display mode. Handles all filter pipelines including basic
 * filters, edge detection, face detection, depth effects, and warp effects.
 * 
 * @param frame Input video frame (3-channel BGR image)
 * @param displayFrame Output frame with effect applied
 * @param mode Current display mode determining which effect to apply
 * @param sobelX Temporary storage for Sobel X gradient
 * @param sobelY Temporary storage for Sobel Y gradient
 * @param quantizeLevels Number of quantization levels for blur-quantize effect
 * @param depthMap Pre-computed depth map (if available)
 * @param warpStrength Strength parameter for warp effects [0.1-1.0]
 * @param sparkles Persistent sparkle state data for animation
 * @param time Current animation time in seconds for time-based effects
 * @return 0 on success
 */
int processFrame(Mat &frame, Mat &displayFrame, DisplayMode mode,
                 Mat &sobelX, Mat &sobelY, int quantizeLevels, Mat &depthMap,
                 float warpStrength, vector<vector<Sparkle>> &sparkles, float time) {
    
    switch (mode) {
        case MODE_COLOR:
            displayFrame = frame.clone();
            break;
            
        case MODE_GREYSCALE_CV:
            {
                Mat greyImage;
                cvtColor(frame, greyImage, COLOR_BGR2GRAY);
                cvtColor(greyImage, displayFrame, COLOR_GRAY2BGR);
            }
            break;
            
        case MODE_GREYSCALE_CUSTOM:
            if (greyscale(frame, displayFrame) != 0) {
                displayFrame = frame.clone();
            }
            break;
            
        case MODE_SEPIA:
            if (sepiaTone(frame, displayFrame, true) != 0) {
                displayFrame = frame.clone();
            }
            break;
            
        case MODE_BLUR:
            if (blur5x5_2(frame, displayFrame) != 0) {
                displayFrame = frame.clone();
            }
            break;
            
        case MODE_SOBEL_X:
            if (sobelX3x3(frame, sobelX) != 0) {
                displayFrame = frame.clone();
            } else {
                // Scale by 2 and add 128 offset to center around gray
                convertScaleAbs(sobelX, displayFrame, 2.0, 128.0);
            }
            break;
            
        case MODE_SOBEL_Y:
            if (sobelY3x3(frame, sobelY) != 0) {
                displayFrame = frame.clone();
            } else {
                // Scale by 2 and add 128 offset to center around gray
                convertScaleAbs(sobelY, displayFrame, 2.0, 128.0);
            }
            break;
            
        case MODE_MAGNITUDE:
            if (sobelX3x3(frame, sobelX) != 0 || sobelY3x3(frame, sobelY) != 0) {
                displayFrame = frame.clone();
            } else {
                Mat magTemp;
                if (magnitude(sobelX, sobelY, magTemp) != 0) {
                    displayFrame = frame.clone();
                } else {
                    // Magnitude doesn't need offset, just scaling
                    convertScaleAbs(magTemp, displayFrame, 4.0, 0);
                }
            }
            break;
            
        case MODE_QUANTIZE:
            if (blurQuantize(frame, displayFrame, quantizeLevels) != 0) {
                displayFrame = frame.clone();
            }
            break;
            
        case MODE_FACE_DETECT:
            {
                Mat grey;
                cvtColor(frame, grey, COLOR_BGR2GRAY);
                vector<Rect> faces;
                detectFaces(grey, faces);
                displayFrame = frame.clone();
                drawBoxes(displayFrame, faces);
            }
            break;
            
        case MODE_DEPTH:
            if (!depthMap.empty()) {
                Mat inverted;
                cv::subtract(255, depthMap, inverted);
                cv::applyColorMap(inverted, displayFrame, COLORMAP_INFERNO);
            } else {
                displayFrame = frame.clone();
                putText(displayFrame, "Depth not available", 
                       Point(50, frame.rows/2),
                       FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
            }
            break;
            
        case MODE_DEPTH_FOG:
            if (!depthMap.empty()) {
                Mat invertedDepth;
                cv::subtract(255, depthMap, invertedDepth);
                if (depthFog(frame, invertedDepth, displayFrame, 3.0f) != 0) {
                    displayFrame = frame.clone();
                }
            } else {
                displayFrame = frame.clone();
                putText(displayFrame, "Depth not available", 
                       Point(50, frame.rows/2),
                       FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
            }
            break;

        case MODE_EMBOSS:
            if (embossEffect(frame, displayFrame) != 0) {
                displayFrame = frame.clone();
            }
            break;
            
        case MODE_NEGATIVE:
            if (negativeEffect(frame, displayFrame) != 0) {
                displayFrame = frame.clone();
            }
            break;

        case MODE_FACE_HIGHLIGHT:
            {
                Mat grey;
                cvtColor(frame, grey, COLOR_BGR2GRAY);
                vector<Rect> faces;
                detectFaces(grey, faces);
                
                if (faceHighlight(frame, displayFrame, faces) != 0) {
                    displayFrame = frame.clone();
                }
            }
            break;

        case MODE_CARTOON:
            if (cartoonEffect(frame, displayFrame) != 0) {
                displayFrame = frame.clone();
            }
            break;

        case MODE_DEPTH_FOCUS:
            if (!depthMap.empty()) {
                if (depthFocus(frame, depthMap, displayFrame, 200, 40) != 0) {
                    displayFrame = frame.clone();
                }
            } else {
                displayFrame = frame.clone();
                putText(displayFrame, "Depth not available", 
                       Point(50, frame.rows/2),
                       FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
            }
            break;

        // === WARP EFFECTS (Extension) ===
        
        case MODE_BULGE:
            if (bulgeEffect(frame, displayFrame, warpStrength) != 0) {
                displayFrame = frame.clone();
            }
            break;
            
        case MODE_WAVE:
            // amplitude = warpStrength * 20, frequency based on strength
            if (waveEffect(frame, displayFrame, warpStrength * 20.0f, 0.02f + warpStrength * 0.03f) != 0) {
                displayFrame = frame.clone();
            }
            break;
            
        case MODE_SWIRL:
            // angle = warpStrength * 4 radians
            if (swirlEffect(frame, displayFrame, warpStrength * 4.0f) != 0) {
                displayFrame = frame.clone();
            }
            break;
            
        case MODE_FACE_BULGE:
            {
                Mat grey;
                cvtColor(frame, grey, COLOR_BGR2GRAY);
                vector<Rect> faces;
                detectFaces(grey, faces);
                
                if (faceBulgeEffect(frame, displayFrame, faces, warpStrength) != 0) {
                    displayFrame = frame.clone();
                }
            }
            break;

        case MODE_SPARKLES:
            {
                Mat grey;
                cvtColor(frame, grey, COLOR_BGR2GRAY);
                vector<Rect> faces;
                detectFaces(grey, faces);
                
                if (sparkleEffect(frame, displayFrame, faces, sparkles, time) != 0) {
                    displayFrame = frame.clone();
                }
            }
            break;

        default:
            displayFrame = frame.clone();
    }
    
    return 0;
}

/**
 * @brief Main application entry point for video capture and display.
 * 
 * Initializes camera, creates display window, and runs the main processing loop.
 * Handles user input for mode switching, parameter adjustments, and frame saving.
 * Supports optional depth estimation via ONNX Runtime if compiled with USE_ONNXRUNTIME.
 * 
 * @param argc Number of command line arguments
 * @param argv Command line arguments (argv[1] = optional camera index)
 * @return 0 on successful execution, -1 on error
 * 
 * Main Loop:
 * 1. Capture frame from camera
 * 2. Update animation time
 * 3. Compute depth map if needed
 * 4. Process frame with current effect
 * 5. Display result
 * 6. Handle keyboard input
 */
int main(int argc, char* argv[]) {
    cout << "=== Video Display Application ===" << endl;
    cout << "Tasks 2-12 + Extensions: Live video with filters\n" << endl;
    
    int cameraIndex = 0;
    if (argc > 1) {
        cameraIndex = atoi(argv[1]);
        cout << "Using camera index: " << cameraIndex << endl;
    }
    
    VideoCapture capdev(cameraIndex);
    
    if (!capdev.isOpened()) {
        cerr << "Error: Unable to open video device" << endl;
        return -1;
    }
    
    Size refS((int)capdev.get(CAP_PROP_FRAME_WIDTH),
              (int)capdev.get(CAP_PROP_FRAME_HEIGHT));
    double fps = capdev.get(CAP_PROP_FPS);
    
    cout << "Video device opened successfully!" << endl;
    cout << "Frame size: " << refS.width << " x " << refS.height << endl;
    cout << "Camera FPS: " << fps << endl;
    
    printControls();
    
    namedWindow("Video Display", WINDOW_AUTOSIZE);
    
    // State variables
    DisplayMode currentMode = MODE_COLOR;
    Mat frame, displayFrame;
    Mat sobelX, sobelY;
    int frameCount = 0;
    int savedCount = 0;
    int quantizeLevels = 10;
    float warpStrength = 0.5f;  // Adjustable warp strength [0.1 - 1.0]
    
    // Sparkle animation variables
    vector<vector<Sparkle>> sparkles;
    auto startTime = high_resolution_clock::now();
    
    // Depth estimation variables
    #ifdef USE_ONNXRUNTIME
    DA2Network* depthNet = nullptr;
    #endif
    Mat depthMap;
    
    #ifdef USE_ONNXRUNTIME
    try {
        depthNet = new DA2Network("model_fp16.onnx");
        cout << "Depth Anything V2 network initialized with GPU!" << endl;
    } catch (const exception& e) {
        cerr << "Warning: Could not load depth network: " << e.what() << endl;
        depthNet = nullptr;
    }
    #endif
    
    cout << "Starting video capture... Current mode: " << getModeString(currentMode) << endl;
    cout << "Warp strength: " << warpStrength << " (adjust with +/-)" << endl;
    
    // Main video loop
    while (true) {
        capdev >> frame;
        
        if (frame.empty()) {
            cerr << "Error: Frame is empty" << endl;
            break;
        }
        
        frameCount++;
        
        // Calculate elapsed time for animations
        auto currentTime = high_resolution_clock::now();
        duration<float> elapsed = currentTime - startTime;
        float time = elapsed.count();
        
        // Compute depth when needed
        #ifdef USE_ONNXRUNTIME
        bool needsDepth = (currentMode == MODE_DEPTH || currentMode == MODE_DEPTH_FOG ||
                          currentMode == MODE_DEPTH_FOCUS);
        
        if (depthNet != nullptr && needsDepth) {
            depthNet->set_input(frame, 1.0f);
            depthNet->run_network(depthMap, frame.size());
        }
        #endif
        
        // Process frame based on current mode
        processFrame(frame, displayFrame, currentMode, sobelX, sobelY, 
                    quantizeLevels, depthMap, warpStrength, sparkles, time);
        
        imshow("Video Display", displayFrame);
        
        char key = (char)waitKey(1);
        
        if (key == -1) continue;
        
        // Handle keyboard input
        if (key == 'q' || key == 'Q' || key == 27) {
            cout << "\nQuitting... Frames: " << frameCount << ", Saved: " << savedCount << endl;
            break;
        }
        else if (key == 's' || key == 'S') {
            string modePrefix = getModeString(currentMode);
            modePrefix.erase(remove(modePrefix.begin(), modePrefix.end(), ' '), modePrefix.end());
            modePrefix.erase(remove(modePrefix.begin(), modePrefix.end(), '('), modePrefix.end());
            modePrefix.erase(remove(modePrefix.begin(), modePrefix.end(), ')'), modePrefix.end());
            string filename = generateTimestampFilename("frame_" + modePrefix, ".jpg");
            
            if (imwrite(filename, displayFrame)) {
                savedCount++;
                cout << "Saved: " << filename << endl;
            }
        }
        else if (key == 'i' || key == 'I') {
            displayVideoInfo(displayFrame, frameCount, savedCount, fps, currentMode);
        }
        else if (key == 'p' || key == 'P') {
            printControls();
        }
        // Basic modes
        else if (key == 'c' || key == 'C') { currentMode = MODE_COLOR; cout << "Mode: Color" << endl; }
        else if (key == 'g' || key == 'G') { currentMode = MODE_GREYSCALE_CV; cout << "Mode: Greyscale (CV)" << endl; }
        else if (key == 'h' || key == 'H') { currentMode = MODE_GREYSCALE_CUSTOM; cout << "Mode: Greyscale (Custom)" << endl; }
        else if (key == 't' || key == 'T') { currentMode = MODE_SEPIA; cout << "Mode: Sepia" << endl; }
        else if (key == 'b' || key == 'B') { currentMode = MODE_BLUR; cout << "Mode: Blur" << endl; }
        else if (key == 'x' || key == 'X') { currentMode = MODE_SOBEL_X; cout << "Mode: Sobel X" << endl; }
        else if (key == 'y' || key == 'Y') { currentMode = MODE_SOBEL_Y; cout << "Mode: Sobel Y" << endl; }
        else if (key == 'm' || key == 'M') { currentMode = MODE_MAGNITUDE; cout << "Mode: Magnitude" << endl; }
        else if (key == 'l' || key == 'L') { currentMode = MODE_QUANTIZE; cout << "Mode: Quantize" << endl; }
        else if (key == 'f' || key == 'F') { currentMode = MODE_FACE_DETECT; cout << "Mode: Face Detect" << endl; }
        else if (key == 'd' || key == 'D') { currentMode = MODE_DEPTH; cout << "Mode: Depth" << endl; }
        // Custom effects (Task 12)
        else if (key == '1') { currentMode = MODE_DEPTH_FOG; cout << "Mode: Depth Fog" << endl; }
        else if (key == '2') { currentMode = MODE_EMBOSS; cout << "Mode: Emboss" << endl; }
        else if (key == '3') { currentMode = MODE_NEGATIVE; cout << "Mode: Negative" << endl; }
        else if (key == '4') { currentMode = MODE_FACE_HIGHLIGHT; cout << "Mode: Face Highlight" << endl; }
        else if (key == '5') { currentMode = MODE_CARTOON; cout << "Mode: Cartoon" << endl; }
        else if (key == '6') { currentMode = MODE_DEPTH_FOCUS; cout << "Mode: Depth Focus" << endl; }
        // Warp effects (Extension)
        else if (key == '7') { currentMode = MODE_BULGE; cout << "Mode: Bulge Warp (strength: " << warpStrength << ")" << endl; }
        else if (key == '8') { currentMode = MODE_WAVE; cout << "Mode: Wave Warp (strength: " << warpStrength << ")" << endl; }
        else if (key == '9') { currentMode = MODE_SWIRL; cout << "Mode: Swirl Warp (strength: " << warpStrength << ")" << endl; }
        else if (key == '0') { currentMode = MODE_FACE_BULGE; cout << "Mode: Face Bulge (strength: " << warpStrength << ")" << endl; }
        else if (key == '[' || key == '{') { currentMode = MODE_SPARKLES; cout << "Mode: Sparkles" << endl; }
        // Adjustments
        else if (key == '+' || key == '=') {
            if (currentMode == MODE_QUANTIZE) {
                quantizeLevels = std::min(25, quantizeLevels + 1);
                cout << "Quantize levels: " << quantizeLevels << endl;
            } else {
                warpStrength = std::min(1.0f, warpStrength + 0.1f);
                cout << "Warp strength: " << warpStrength << endl;
            }
        }
        else if (key == '-' || key == '_') {
            if (currentMode == MODE_QUANTIZE) {
                quantizeLevels = std::max(2, quantizeLevels - 1);
                cout << "Quantize levels: " << quantizeLevels << endl;
            } else {
                warpStrength = std::max(0.1f, warpStrength - 0.1f);
                cout << "Warp strength: " << warpStrength << endl;
            }
        }
    }
    
    // Cleanup
    #ifdef USE_ONNXRUNTIME
    if (depthNet != nullptr) {
        delete depthNet;
    }
    #endif
    
    destroyAllWindows();
    cout << "Video capture closed." << endl;
    return 0;
}