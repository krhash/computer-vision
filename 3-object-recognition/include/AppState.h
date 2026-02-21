/**
 * @file    AppState.h
 * @brief   Shared pipeline parameters and application state structs.
 *
 *          PipelineParams — all tunable values consumed by task functions.
 *          AppState       — pipeline outputs and display-ready data.
 *
 *          Design goal: every task function takes (params) in and writes
 *          to (state) out.  Keyboard controls modify params now; ImGui
 *          sliders will modify the same struct in the GUI phase with zero
 *          changes to task code.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// =============================================================================
// RegionInfo — computed per detected object region each frame
// =============================================================================
struct RegionInfo {
    int             id          = 0;
    cv::Rect        boundingBox;        ///< Axis-aligned bounding box
    cv::RotatedRect orientedBox;        ///< Oriented (rotated) bounding box
    cv::Point2f     centroid;           ///< Region centroid (cx, cy)
    double          angle       = 0.0;  ///< Primary axis angle (radians)
    double          area        = 0.0;  ///< Pixel area
    double          fillRatio   = 0.0;  ///< area / boundingBox.area()
    double          bboxRatio   = 0.0;  ///< boundingBox height / width

    // Axis projections — needed by Task 9 embedding (prepEmbeddingImage)
    float           minE1       = 0.f;  ///< Min projection along primary axis
    float           maxE1       = 0.f;  ///< Max projection along primary axis
    float           minE2       = 0.f;  ///< Min projection along secondary axis
    float           maxE2       = 0.f;  ///< Max projection along secondary axis
    std::vector<double> huMoments;      ///< 7 Hu moment invariants

    // Classifier output
    std::string     label       = "unknown";
    float           confidence  = 0.f;  ///< 1.0 = closest match

    // Embedding (Task 9)
    std::vector<float> embedding;

    // Display
    cv::Scalar      displayColor = {200, 200, 200};
};

// =============================================================================
// PipelineParams — all tunable values (keyboard now, ImGui sliders later)
// =============================================================================
struct PipelineParams {

    // --- Task 1: Threshold ---------------------------------------------------
    int     thresholdValue      = 127;  ///< Global threshold [0..255]
    int     blurKernelSize      = 21;    ///< Pre-blur kernel size (must be odd)
    bool    useAdaptive         = false;///< Adaptive vs global threshold
    bool    useKMeans           = false;///< ISODATA dynamic threshold
    bool    useSatIntensity     = false;

    // --- Task 2: Morphology --------------------------------------------------
    int     morphKernelSize     = 5;    ///< Structuring element size
    int     morphIterations     = 2;    ///< Number of morph iterations
    /// 0=open, 1=close, 2=erode, 3=dilate
    int     morphMode           = 0;

    // --- Task 3: Connected Components ----------------------------------------
    int     minRegionArea       = 500;  ///< Ignore regions smaller than this
    int     maxRegions          = 5;    ///< Keep top N largest regions

    // --- Task 4: Features ----------------------------------------------------
    bool    showAxes            = true;
    bool    showOrientedBBox    = true;
    bool    showFeatureText     = true;

    // --- Task 6: Classifier --------------------------------------------------
    int     kNeighbors          = 1;
    float   confidenceThresh    = 0.50f;
    /// 0=scaled Euclidean, 1=cosine
    int     distanceMetric      = 0;

    // --- Task 9: Embedding ---------------------------------------------------
    /// 0=hand-features, 1=CNN ResNet18, 2=eigenspace PCA
    int     embeddingMode       = 0;
    int     roiSize             = 224;  ///< ROI resize before embedding
};

// =============================================================================
// AppState — runtime state passed through the main loop each frame
// =============================================================================
struct AppState {

    // --- Mode ----------------------------------------------------------------
    enum class Mode { Live, Train, Eval, Embed } mode = Mode::Live;

    // --- Pipeline intermediate images ----------------------------------------
    cv::Mat frameOriginal;      ///< Raw camera / loaded frame
    cv::Mat frameThresholded;   ///< After Task 1
    cv::Mat frameCleaned;       ///< After Task 2
    cv::Mat frameRegions;       ///< Color-coded region map (Task 3)
    cv::Mat frameDisplay;       ///< Final annotated output frame

    // --- Detected regions this frame -----------------------------------------
    std::vector<RegionInfo> regions;

    // --- Training (Task 5) ---------------------------------------------------
    std::string currentTrainLabel;
    int         samplesThisLabel    = 0;
    bool        captureRequested    = false; ///< Set true on keypress/button

    // --- Evaluation (Task 7) -------------------------------------------------
    std::vector<std::string>        evalLabels;
    std::vector<std::vector<int>>   confusionMatrix;
    float                           accuracy    = 0.f;

    // --- Embedding (Task 9) --------------------------------------------------
    std::vector<float> lastEmbedding;

    // --- UI / diagnostics ----------------------------------------------------
    std::string statusMessage   = "Ready";
    float       fps             = 0.f;
    bool        running         = true;
    bool        embeddingMode_  = false; ///< Task 9: use CNN embedding classifier
};
