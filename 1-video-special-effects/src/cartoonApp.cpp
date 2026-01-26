/*
  Author: Krushna Sanjay Sharma
  Date: January 25, 2026
  Purpose: Simple video cartoonization using bilateral filtering and DoG
           Based on Winnemöller et al. (2006) "Real-time video abstraction"
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "cartoonVideo.hpp"

using namespace cv;
using namespace std;

/**
 * @brief Main application entry point
 * 
 * Simple cartoon video stream with no user controls.
 * Press 'q' or ESC to quit.
 * 
 * @param argc Number of command line arguments
 * @param argv Command line arguments (optional camera index)
 * @return 0 on success, -1 on error
 */
int main(int argc, char* argv[]) {
    cout << "=== Cartoon Video Application ===" << endl;
    cout << "Based on Winnemoller et al. (2006)" << endl;
    cout << "Press 'q' or ESC to quit\n" << endl;
    
    // Get camera index
    int cameraIndex = 0;
    if (argc > 1) {
        cameraIndex = atoi(argv[1]);
    }
    
    // Open video capture
    VideoCapture capdev(cameraIndex);
    if (!capdev.isOpened()) {
        cerr << "Error: Unable to open video device" << endl;
        return -1;
    }
    
    cout << "Video device opened successfully!" << endl;
    
    // Create cartoon processor with default parameters
    CartoonVideo cartoon;
    
    // Create display window
    namedWindow("Cartoon Video", WINDOW_AUTOSIZE);
    
    Mat frame, displayFrame;
    
    // Main video loop
    while (true) {
        capdev >> frame;
        
        if (frame.empty()) {
            cerr << "Error: Frame is empty" << endl;
            break;
        }
        
        // Process frame
        if (cartoon.processFrame(frame, displayFrame) != 0) {
            displayFrame = frame.clone();
        }
        
        imshow("Cartoon Video", displayFrame);
        
        // Check for quit key
        char key = (char)waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }
    }
    
    // Cleanup
    destroyAllWindows();
    cout << "Video capture closed." << endl;
    return 0;
}