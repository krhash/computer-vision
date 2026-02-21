/**
 * @file    Threshold.cpp
 * @brief   Implementation of thresholding pipeline stage.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#include "Threshold.h"
#include <random>
#include <cmath>

// -----------------------------------------------------------------------------
void applyCustomSatIntensityThreshold(const cv::Mat& src, cv::Mat& dst,
                                      const PipelineParams& params)
{
    CV_Assert(!src.empty() && src.channels() == 3);

    // Convert BGR → HSV for saturation and intensity access
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    // Normalised threshold in [0..1] from integer param [0..255]
    float thresh = params.thresholdValue / 255.f;

    for (int r = 0; r < hsv.rows; r++) {
        const cv::Vec3b* rowPtr = hsv.ptr<cv::Vec3b>(r);
        uchar*           dstPtr = dst.ptr<uchar>(r);
        for (int c = 0; c < hsv.cols; c++) {
            // OpenCV HSV: H[0..179], S[0..255], V[0..255]
            float S = rowPtr[c][1] / 255.f;  ///< Saturation [0..1]
            float V = rowPtr[c][2] / 255.f;  ///< Value/intensity [0..1]

            // score: high for white (low S, high V), low for dark/coloured
            float score = (1.f - S) * V;

            // Foreground (object) = score below threshold
            dstPtr[c] = (score < thresh) ? 255 : 0;
        }
    }
}

// -----------------------------------------------------------------------------
void applyBlur(const cv::Mat& src, cv::Mat& dst, const PipelineParams& params)
{
    int k = params.blurKernelSize;
    // Kernel must be odd and >= 1
    if (k < 1) k = 1;
    if (k % 2 == 0) k++;
    cv::GaussianBlur(src, dst, cv::Size(k, k), 0);
}

// -----------------------------------------------------------------------------
void toGrayscale(const cv::Mat& src, cv::Mat& dst)
{
    if (src.channels() == 1) {
        dst = src.clone();
    } else {
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    }
}

// -----------------------------------------------------------------------------
int computeISODATAThreshold(const cv::Mat& gray)
{
    CV_Assert(gray.type() == CV_8UC1);

    // Collect a random 1/16 sample of pixel values
    std::vector<float> samples;
    samples.reserve(gray.total() / 16 + 1);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> distR(0, gray.rows - 1);
    std::uniform_int_distribution<int> distC(0, gray.cols - 1);

    int nSamples = static_cast<int>(gray.total() / 16);
    for (int i = 0; i < nSamples; i++) {
        samples.push_back(static_cast<float>(
            gray.at<uchar>(distR(rng), distC(rng))));
    }

    // Two-cluster k-means (ISODATA): iterate until means stabilise
    float m1 = 80.f, m2 = 180.f; // initial guesses: dark / light
    for (int iter = 0; iter < 50; iter++) {
        float sum1 = 0, sum2 = 0;
        int   cnt1 = 0, cnt2 = 0;
        for (float v : samples) {
            if (std::fabs(v - m1) < std::fabs(v - m2)) { sum1 += v; cnt1++; }
            else                                        { sum2 += v; cnt2++; }
        }
        float nm1 = cnt1 > 0 ? sum1 / cnt1 : m1;
        float nm2 = cnt2 > 0 ? sum2 / cnt2 : m2;
        if (std::fabs(nm1 - m1) < 0.5f && std::fabs(nm2 - m2) < 0.5f) break;
        m1 = nm1; m2 = nm2;
    }
    return static_cast<int>((m1 + m2) / 2.f);
}

// -----------------------------------------------------------------------------
void applyThreshold(const cv::Mat& src, cv::Mat& dst,
                    const PipelineParams& params)
{
    // 1. Blur to reduce noise
    cv::Mat blurred;
    applyBlur(src, blurred, params);

    // 2. Convert to grayscale
    cv::Mat gray;
    toGrayscale(blurred, gray);

    // 3. Threshold — select mode from params
    if (params.useSatIntensity) {
        // Custom saturation+intensity mode — works on blurred colour frame
        applyCustomSatIntensityThreshold(blurred, dst, params);

    } else if (params.useKMeans) {
        // Dynamic threshold: midpoint of two dominant pixel clusters
        int dynThresh = computeISODATAThreshold(gray);
        cv::threshold(gray, dst, dynThresh, 255, cv::THRESH_BINARY_INV);

    } else if (params.useAdaptive) {
        // Adaptive: threshold varies locally — good for uneven lighting
        cv::adaptiveThreshold(gray, dst, 255,
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY_INV,
                              11, 2);

    } else {
        // Global fixed threshold from params
        cv::threshold(gray, dst,
                      params.thresholdValue, 255,
                      cv::THRESH_BINARY_INV);
    }
}
