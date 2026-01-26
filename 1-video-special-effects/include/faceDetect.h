/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Include file for faceDetect.cpp, face detection and drawing functions
*/
#ifndef FACEDETECT_H
#define FACEDETECT_H

// put the path to the haar cascade file here
#define FACE_CASCADE_FILE "./haarcascade_frontalface_alt2.xml"

// prototypes

/*
  Function: detectFaces
  Purpose: Detect faces in a grayscale image using Haar cascade classifier
  Arguments:
    grey - input grayscale image (CV_8UC1)
    faces - output vector to store detected face rectangles
  Return value: 0 on success, -1 on error
*/
int detectFaces( cv::Mat &grey, std::vector<cv::Rect> &faces );

/*
  Function: drawBoxes
  Purpose: Draw bounding boxes around detected faces on the input frame
  Arguments:
    frame - input/output image where boxes will be drawn
    faces - vector of detected face rectangles
    minWidth - minimum width of faces to draw (default: 50)
    scale - scaling factor for the frame (default: 1.0)
  Return value: 0 on success, -1 on error
*/
int drawBoxes( cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth = 50, float scale = 1.0  );

#endif
