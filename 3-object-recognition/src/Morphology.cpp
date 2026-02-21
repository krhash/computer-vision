/**
 * @file    Morphology.cpp
 * @brief   From-scratch morphological filtering implementation.
 *
 *          Erosion and dilation are implemented manually without using
 *          cv::erode, cv::dilate, or cv::morphologyEx.  Only basic
 *          cv::Mat pixel access is used.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#include "Morphology.h"

// -----------------------------------------------------------------------------
// Internal helper — single-pass erosion
// -----------------------------------------------------------------------------
static void erodeOnce(const cv::Mat& src, cv::Mat& dst, int kSize)
{
    CV_Assert(src.type() == CV_8UC1);

    int half = kSize / 2;

    // Pad with 255 (foreground) so border pixels are not incorrectly eroded
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, half, half, half, half,
                       cv::BORDER_CONSTANT, cv::Scalar(255));

    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            bool allFg = true;
            for (int dr = 0; dr < kSize && allFg; dr++) {
                const uchar* row = padded.ptr<uchar>(r + dr);
                for (int dc = 0; dc < kSize && allFg; dc++) {
                    if (row[c + dc] == 0) allFg = false;
                }
            }
            dst.at<uchar>(r, c) = allFg ? 255 : 0;
        }
    }
}

// -----------------------------------------------------------------------------
// Internal helper — single-pass dilation
// -----------------------------------------------------------------------------
static void dilateOnce(const cv::Mat& src, cv::Mat& dst, int kSize)
{
    CV_Assert(src.type() == CV_8UC1);

    int half = kSize / 2;

    // Pad with 0 (background) so border pixels are not incorrectly dilated
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, half, half, half, half,
                       cv::BORDER_CONSTANT, cv::Scalar(0));

    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            bool anyFg = false;
            for (int dr = 0; dr < kSize && !anyFg; dr++) {
                const uchar* row = padded.ptr<uchar>(r + dr);
                for (int dc = 0; dc < kSize && !anyFg; dc++) {
                    if (row[c + dc] == 255) anyFg = true;
                }
            }
            dst.at<uchar>(r, c) = anyFg ? 255 : 0;
        }
    }
}

// -----------------------------------------------------------------------------
void erodeCustom(const cv::Mat& src, cv::Mat& dst, int kSize, int iters)
{
    // Clamp and enforce odd kernel size
    if (kSize < 1) kSize = 1;
    if (kSize % 2 == 0) kSize++;

    cv::Mat cur = src.clone();
    cv::Mat tmp;
    for (int i = 0; i < iters; i++) {
        erodeOnce(cur, tmp, kSize);
        cur = tmp;
    }
    dst = cur;
}

// -----------------------------------------------------------------------------
void dilateCustom(const cv::Mat& src, cv::Mat& dst, int kSize, int iters)
{
    if (kSize < 1) kSize = 1;
    if (kSize % 2 == 0) kSize++;

    cv::Mat cur = src.clone();
    cv::Mat tmp;
    for (int i = 0; i < iters; i++) {
        dilateOnce(cur, tmp, kSize);
        cur = tmp;
    }
    dst = cur;
}

// -----------------------------------------------------------------------------
void applyMorphology(const cv::Mat& src, cv::Mat& dst,
                     const PipelineParams& params)
{
    int k    = params.morphKernelSize;
    int iter = params.morphIterations;

    switch (params.morphMode) {
        case 0: // Open: erode → dilate (removes noise)
        {
            cv::Mat eroded;
            erodeCustom(src, eroded, k, iter);
            dilateCustom(eroded, dst, k, iter);
            break;
        }
        case 1: // Close: dilate → erode (fills holes)
        {
            cv::Mat dilated;
            dilateCustom(src, dilated, k, iter);
            erodeCustom(dilated, dst, k, iter);
            break;
        }
        case 2: // Erode only
            erodeCustom(src, dst, k, iter);
            break;

        case 3: // Dilate only
            dilateCustom(src, dst, k, iter);
            break;

        default:
            dst = src.clone();
            break;
    }
}
