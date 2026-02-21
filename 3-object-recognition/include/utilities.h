/**
 * @file    utilities.h
 * @brief   Utility functions for computing embeddings from region images.
 *
 *          Modified from Prof. Bruce Maxwell's utilities code.
 *          Original: CS 5330 utility functions for features and embeddings.
 *          Modifications:
 *            - Removed dependency on features.h and vision.h
 *            - Parameters adapted to use float theta (radians) directly
 *            - Added include guards and Doxygen comments
 *
 * @author  Krushna Sanjay Sharma
 *          Modified from Prof. Bruce Maxwell's utilities code (CS 5330)
 * @date    February 2026
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/**
 * @brief Rotate frame and extract aligned ROI for embedding.
 *
 *        Rotates the original image so the region's primary axis aligns
 *        with the X-axis, then crops the oriented bounding box.
 *        Output image is suitable for passing to getEmbedding().
 *
 *        Modified from Prof. Bruce Maxwell's prepEmbeddingImage().
 *
 * @param frame     Original BGR camera frame.
 * @param embimage  Output ROI image (aligned and cropped).
 * @param cx        Centroid X coordinate.
 * @param cy        Centroid Y coordinate.
 * @param theta     Primary axis angle in radians.
 * @param minE1     Min projection along primary axis (negative value).
 * @param maxE1     Max projection along primary axis (positive value).
 * @param minE2     Min projection along secondary axis (negative value).
 * @param maxE2     Max projection along secondary axis (positive value).
 * @param debug     1 = show intermediate windows, 0 = silent.
 */
void prepEmbeddingImage(cv::Mat& frame, cv::Mat& embimage,
                        int cx, int cy, float theta,
                        float minE1, float maxE1,
                        float minE2, float maxE2,
                        int debug = 0);

/**
 * @brief Compute embedding vector from image using ResNet18 network.
 *
 *        Resizes input to 224x224, normalises, and runs forward pass.
 *        Output is a 512-dimensional embedding from the penultimate layer.
 *
 *        Modified from Prof. Bruce Maxwell's getEmbedding().
 *
 * @param src       Input BGR image (any size — resized internally).
 * @param embedding Output embedding vector (1x512 cv::Mat, CV_32F).
 * @param net       Loaded ResNet18 DNN network.
 * @param debug     1 = print embedding to console, 0 = silent.
 * @return          0 on success.
 */
int getEmbedding(cv::Mat& src, cv::Mat& embedding,
                 cv::dnn::Net& net, int debug = 0);
                 