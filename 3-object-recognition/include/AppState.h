/**
 * @file    AppState.h
 * @brief   Shared pipeline parameters and application state structs.
 *
 *          PipelineParams — all tunable values consumed by pipeline functions.
 *          AppState       — pipeline outputs and display-ready data.
 *
 *          Design goal: every pipeline function takes (params) in and writes
 *          to (state) out.  Both keyboard shortcuts and ImGui sliders write
 *          to the same struct, keeping pipeline code unchanged.
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
    double          area        = 0.0;  ///< Pixel area (normalised by image area)
    double          fillRatio   = 0.0;  ///< area / orientedBBox.area
    double          bboxRatio   = 0.0;  ///< long side / short side of oriented bbox

    // Axis projections — used by the embedding pipeline (prepEmbeddingImage)
    float           minE1       = 0.f;  ///< Min projection along primary axis
    float           maxE1       = 0.f;  ///< Max projection along primary axis
    float           minE2       = 0.f;  ///< Min projection along secondary axis
    float           maxE2       = 0.f;  ///< Max projection along secondary axis

    std::vector<double> huMoments;      ///< 7 Hu moment invariants (log-scaled)

    // Classifier output
    std::string     label           = "unknown";
    float           confidence      = 0.f;
    int             unknownFrames   = 0;    ///< Consecutive frames classified as unknown

    // CNN embedding vector (512-dimensional ResNet18 output)
    std::vector<float> embedding;

    // Display
    cv::Scalar      displayColor = {200, 200, 200};
};

// =============================================================================
// PipelineParams — all tunable values, written by both keyboard and ImGui controls
// =============================================================================
struct PipelineParams {

    // --- Task 1: Threshold ---------------------------------------------------
    int     thresholdValue      = 127;  ///< Global threshold [0..255]
    int     blurKernelSize      = 21;   ///< Pre-blur kernel size (must be odd)
    bool    useAdaptive         = false;///< Adaptive vs global threshold
    bool    useKMeans           = false;///< ISODATA dynamic threshold
    bool    useSatIntensity     = false;///< Custom sat+intensity threshold (bonus)

    // --- Task 2: Morphology --------------------------------------------------
    int     morphKernelSize     = 5;    ///< Structuring element size
    int     morphIterations     = 3;    ///< Number of morph iterations
    int     morphMode           = 0;    ///< 0=open, 1=close, 2=erode, 3=dilate

    // --- Task 3: Connected Components ----------------------------------------
    int     minRegionArea       = 500;  ///< Ignore regions smaller than this
    int     maxRegions          = 5;    ///< Keep top N largest regions

    // --- Task 4: Features ----------------------------------------------------
    bool    showAxes            = true;
    bool    showOrientedBBox    = true;
    bool    showFeatureText     = true;

    // --- Classifier ----------------------------------------------------------
    int     kNeighbors          = 1;
    float   confidenceThresh    = 0.60f;
    int     distanceMetric      = 0;    ///< 0=scaled Euclidean, 1=cosine

    // --- CNN Embedding -------------------------------------------------------
    int     embeddingMode       = 0;    ///< 0=hand-crafted features, 1=CNN ResNet18
    int     roiSize             = 224;  ///< ROI resize dimension before embedding (pixels)
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

    // --- Training ------------------------------------------------------------
    std::string currentTrainLabel;
    int         samplesThisLabel    = 0;
    bool        captureRequested    = false; ///< Set true by keypress or GUI button to trigger capture

    // --- Evaluation ----------------------------------------------------------
    std::vector<std::string>        evalLabels;
    std::vector<std::vector<int>>   confusionMatrix;
    float                           accuracy    = 0.f;

    // --- CNN Embedding -------------------------------------------------------
    std::vector<float>   lastEmbedding;
    cv::Mat              lastCroppedROI;  ///< Most recent aligned ROI sent to the network
    std::vector<cv::Mat> croppedROIs;     ///< One aligned ROI crop per detected region

    // --- UI / diagnostics ----------------------------------------------------
    std::string statusMessage   = "Ready";
    float       fps             = 0.f;
    bool        running         = true;
    bool        embeddingMode_  = false; ///< Task 9: use CNN embedding classifier
    bool        showOverlay     = true;  ///< Show config overlay text on main window
    bool        showPlot        = false; ///< Show embedding PCA scatter plot window

    // --- Auto-learn ----------------------------------------------------------
    bool        autoLearnPending = false; ///< True when an unknown region has persisted long enough to prompt the user
    RegionInfo  autoLearnRegion;          ///< Copy of the region that triggered the auto-learn prompt
};
