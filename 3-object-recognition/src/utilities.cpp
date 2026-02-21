/**
 * @file    utilities.cpp
 * @brief   Embedding utility functions implementation.
 *
 *          Modified from Prof. Bruce Maxwell's utilities code (CS 5330).
 *          Original source provided as course material.
 *          Modifications:
 *            - Removed features.h / vision.h dependencies
 *            - Parameters use float theta in radians directly
 *            - Added bounds-check comments and defensive clamping
 *
 * @author  Krushna Sanjay Sharma
 *          Modified from Prof. Bruce Maxwell's utilities code (CS 5330)
 * @date    February 2026
 */

#include "utilities.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

// MSVC does not define M_PI by default
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------------
// Modified from Prof. Bruce Maxwell's prepEmbeddingImage()
// -----------------------------------------------------------------------------
void prepEmbeddingImage(cv::Mat& frame, cv::Mat& embimage,
                        int cx, int cy, float theta,
                        float minE1, float maxE1,
                        float minE2, float maxE2,
                        int debug)
{
    // Rotate image so primary axis aligns with X-axis
    cv::Mat rotatedImage;
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(static_cast<float>(cx),
                                                    static_cast<float>(cy)),
                                        -theta * 180.0f / static_cast<float>(M_PI),
                                        1.0);

    // Use diagonal as output size so corners don't get clipped after rotation
    int largest = frame.cols > frame.rows ? frame.cols : frame.rows;
    largest = static_cast<int>(1.414 * largest);
    cv::warpAffine(frame, rotatedImage, M, cv::Size(largest, largest));

    if (debug)
        cv::imshow("rotated", rotatedImage);

    // Compute ROI from axis extents
    int left   = cx + static_cast<int>(minE1);
    int top    = cy - static_cast<int>(maxE2);
    int width  = static_cast<int>(maxE1) - static_cast<int>(minE1);
    int height = static_cast<int>(maxE2) - static_cast<int>(minE2);

    // Bounds check — clamp ROI to image dimensions
    if (left < 0)                        { width  += left; left = 0; }
    if (top  < 0)                        { height += top;  top  = 0; }
    if (left + width  >= rotatedImage.cols) width  = rotatedImage.cols - 1 - left;
    if (top  + height >= rotatedImage.rows) height = rotatedImage.rows - 1 - top;

    // Guard against degenerate ROI
    if (width <= 0 || height <= 0) {
        embimage = frame.clone();
        return;
    }

    if (debug)
        printf("ROI box: %d %d %d %d\n", left, top, width, height);

    cv::Rect objroi(left, top, width, height);
    cv::Mat  extracted(rotatedImage, objroi);

    if (debug)
        cv::imshow("extracted", extracted);

    extracted.copyTo(embimage);
}

// -----------------------------------------------------------------------------
// Modified from Prof. Bruce Maxwell's getEmbedding()
// -----------------------------------------------------------------------------
int getEmbedding(cv::Mat& src, cv::Mat& embedding,
                 cv::dnn::Net& net, int debug)
{
    const int ORNet_size = 224;
    cv::Mat   blob, resized;

    cv::resize(src, resized, cv::Size(ORNet_size, ORNet_size));

    cv::dnn::blobFromImage(resized,
                           blob,
                           (1.0 / 255.0) * (1.0 / 0.226),   // scale
                           cv::Size(ORNet_size, ORNet_size),
                           cv::Scalar(124, 116, 104),         // mean subtraction
                           true,    // swapRB
                           false,   // center crop
                           CV_32F);

    net.setInput(blob);

    // Layer name for resnet18-v2-7.onnx penultimate flatten layer
    embedding = net.forward("onnx_node!resnetv22_flatten0_reshape0");

    if (debug)
        std::cout << "Embedding size: " << embedding.size() << "\n";

    return 0;
}
