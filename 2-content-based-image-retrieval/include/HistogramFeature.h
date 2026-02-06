////////////////////////////////////////////////////////////////////////////////
// HistogramFeature.h
// Author: Krushna Sanjay Sharma
// Description: Color histogram feature extractor for CBIR Task 2. Supports
//              RGB histograms and RG chromaticity histograms for content-based
//              image retrieval with histogram intersection distance metric.
////////////////////////////////////////////////////////////////////////////////

#ifndef HISTOGRAM_FEATURE_H
#define HISTOGRAM_FEATURE_H

#include "FeatureExtractor.h"

namespace cbir {

/**
 * @class HistogramFeature
 * @brief Extracts color histograms from images
 * 
 * This feature extractor creates multi-dimensional color histograms that
 * represent the distribution of colors in an image. Two histogram types
 * are supported:
 * 
 * 1. RGB Histogram: Bins pixels based on R, G, B values
 *    - 3D histogram with configurable bins per channel
 *    - Default: 8 bins per channel (8×8×8 = 512 bins total)
 * 
 * 2. RG Chromaticity Histogram: Bins pixels based on normalized r, g values
 *    - 2D histogram, lighting-invariant
 *    - r = R/(R+G+B), g = G/(R+G+B)
 *    - Default: 16 bins per channel (16×16 = 256 bins total)
 * 
 * The histogram is flattened into a 1D feature vector for distance computation.
 * Can store raw counts or normalized values.
 * 
 * @author Krushna Sanjay Sharma
 */
class HistogramFeature : public FeatureExtractor {
public:
    enum class HistogramType {
        RGB,            ///< RGB color histogram
        RG_CHROMATICITY ///< RG chromaticity histogram (lighting-invariant)
    };
    
    /**
     * @brief Constructor with histogram type and bins
     * 
     * @param type Histogram type (RGB or RG_CHROMATICITY)
     * @param binsPerChannel Number of bins per channel
     * @param normalize If true, normalize histogram to [0,1]
     */
    HistogramFeature(HistogramType type = HistogramType::RGB,
                     int binsPerChannel = 8,
                     bool normalize = true);
    
    virtual ~HistogramFeature() = default;
    
    virtual cv::Mat extractFeatures(const cv::Mat& image) override;
    virtual std::string getFeatureName() const override;
    virtual int getFeatureDimension() const override;
    
    void setHistogramType(HistogramType type);
    void setBinsPerChannel(int bins);
    void setNormalize(bool normalize);
    
    HistogramType getHistogramType() const { return type_; }
    int getBinsPerChannel() const { return binsPerChannel_; }
    
private:
    HistogramType type_;
    int binsPerChannel_;
    bool normalize_;
    
    /**
     * @brief Compute RGB histogram
     */
    cv::Mat computeRGBHistogram(const cv::Mat& image);
    
    /**
     * @brief Compute RG chromaticity histogram
     */
    cv::Mat computeRGChromaticityHistogram(const cv::Mat& image);
    
    /**
     * @brief Flatten multi-dimensional histogram to 1D vector
     */
    cv::Mat flattenHistogram(const cv::Mat& histogram);
    
    /**
     * @brief Normalize histogram so sum = 1.0
     */
    cv::Mat normalizeHistogram(const cv::Mat& histogram);
};

} // namespace cbir

#endif // HISTOGRAM_FEATURE_H
