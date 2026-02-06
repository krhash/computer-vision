/*
  Chromaticity Analysis
  Based on Bruce A. Maxwell's histogram code (Spring 2024, CS 5330)
  
  This program:
  1. Loads an image with a cast shadow
  2. Generates an rg chromaticity 2D histogram
  3. Creates a chromaticity visualization of the original image
     (all pixels at same intensity, shadows should disappear)
*/

#include <cstdio>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>

// Function to create rg chromaticity histogram
cv::Mat createChromaticityHistogram(const cv::Mat& src, int histsize = 256) {
    cv::Mat hist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32FC1);
    float max = 0;

    // Loop over all pixels
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];

            // Compute the r,g chromaticity
            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0;
            float r = R / divisor;
            float g = G / divisor;

            // Compute indexes
            int rindex = (int)(r * (histsize - 1) + 0.5);
            int gindex = (int)(g * (histsize - 1) + 0.5);

            // Increment the histogram
            hist.at<float>(rindex, gindex)++;
            float newvalue = hist.at<float>(rindex, gindex);
            max = newvalue > max ? newvalue : max;
        }
    }

    printf("Histogram: Largest bucket has %d pixels\n", (int)max);
    
    // Normalize by number of pixels
    hist /= (src.rows * src.cols);
    
    return hist;
}

// Function to visualize the histogram
cv::Mat visualizeHistogram(const cv::Mat& hist) {
    cv::Mat dst;
    dst.create(hist.size(), CV_8UC3);
    
    for (int i = 0; i < hist.rows; i++) {
        cv::Vec3b* ptr = dst.ptr<cv::Vec3b>(i);
        const float* hptr = hist.ptr<float>(i);
        for (int j = 0; j < hist.cols; j++) {
            // Region where r + g > 1 (invalid chromaticity)
            if (i + j > hist.rows) {
                ptr[j] = cv::Vec3b(200, 120, 60);
                continue;
            }
            
            float rcolor = (float)i / hist.rows;
            float gcolor = (float)j / hist.cols;
            float bcolor = 1 - (rcolor + gcolor);

            // Color bins by their chromaticity, weighted by histogram value
            ptr[j][0] = hptr[j] > 0 ? (uchar)(hptr[j] * 128 + 128 * bcolor) : 0;
            ptr[j][1] = hptr[j] > 0 ? (uchar)(hptr[j] * 128 + 128 * gcolor) : 0;
            ptr[j][2] = hptr[j] > 0 ? (uchar)(hptr[j] * 128 + 128 * rcolor) : 0;
        }
    }
    
    return dst;
}

// Function to create chromaticity image (same intensity for all pixels)
cv::Mat createChromaticityImage(const cv::Mat& src, float intensity = 200.0f) {
    cv::Mat dst;
    dst.create(src.size(), CV_8UC3);
    
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b* srcPtr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* dstPtr = dst.ptr<cv::Vec3b>(i);
        
        for (int j = 0; j < src.cols; j++) {
            float B = srcPtr[j][0];
            float G = srcPtr[j][1];
            float R = srcPtr[j][2];
            
            // Compute chromaticity
            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0;
            
            float r = R / divisor;
            float g = G / divisor;
            float b = 1.0f - r - g;  // b = B / divisor, but also 1 - r - g
            
            // Multiply by constant intensity
            dstPtr[j][0] = cv::saturate_cast<uchar>(b * intensity);
            dstPtr[j][1] = cv::saturate_cast<uchar>(g * intensity);
            dstPtr[j][2] = cv::saturate_cast<uchar>(r * intensity);
        }
    }
    
    return dst;
}

int main(int argc, char* argv[]) {
    cv::Mat src;
    std::string filename;
    
    // Check arguments
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image filename>" << std::endl;
        return -1;
    }
    
    filename = argv[1];
    
    // Read the image
    src = cv::imread(filename);
    if (src.empty()) {
        std::cout << "Error: Unable to read file " << filename << std::endl;
        return -2;
    }
    
    std::cout << "Image loaded: " << src.cols << " x " << src.rows << " pixels" << std::endl;
    
    // 1. Create chromaticity histogram
    std::cout << "\nCreating rg chromaticity histogram..." << std::endl;
    cv::Mat hist = createChromaticityHistogram(src);
    
    // 2. Visualize the histogram
    cv::Mat histVis = visualizeHistogram(hist);
    
    // 3. Create chromaticity image (constant intensity = 200)
    std::cout << "Creating chromaticity visualization image..." << std::endl;
    cv::Mat chromaImage = createChromaticityImage(src, 200.0f);
    
    // Resize images for display if they're too large
    cv::Mat srcDisplay, chromaDisplay;
    double scale = 1.0;
    if (src.rows > 800 || src.cols > 800) {
        scale = 800.0 / std::max(src.rows, src.cols);
        cv::resize(src, srcDisplay, cv::Size(), scale, scale);
        cv::resize(chromaImage, chromaDisplay, cv::Size(), scale, scale);
    } else {
        srcDisplay = src.clone();
        chromaDisplay = chromaImage.clone();
    }
    
    // Display all images
    cv::imshow("Original Image", srcDisplay);
    cv::imshow("rg Chromaticity Histogram", histVis);
    cv::imshow("Chromaticity Image (Constant Intensity)", chromaDisplay);
    
    // Save all output images
    std::cout << "\nSaving images..." << std::endl;
    
    // Save original image (resized version for consistency)
    cv::imwrite("original_image.png", src);
    std::cout << "Saved: original_image.png" << std::endl;
    
    // Save histogram visualization
    cv::imwrite("histogram_output.png", histVis);
    std::cout << "Saved: histogram_output.png" << std::endl;
    
    // Save chromaticity image (constant intensity)
    cv::imwrite("chromaticity_image.png", chromaImage);
    std::cout << "Saved: chromaticity_image.png" << std::endl;
    
    // Save a larger version of histogram for better visibility
    cv::Mat histLarge;
    cv::resize(histVis, histLarge, cv::Size(512, 512), 0, 0, cv::INTER_NEAREST);
    cv::imwrite("histogram_large.png", histLarge);
    std::cout << "Saved: histogram_large.png (512x512)" << std::endl;
    
    std::cout << "\nPress any key to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}
