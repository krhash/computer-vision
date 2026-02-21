/**
 * @file    Classifier.h
 * @brief   K-Nearest Neighbour classifier using scaled Euclidean distance.
 *
 *          Implements nearest-neighbour recognition as required by Task 6:
 *            distance = sqrt( sum( ((x1-x2) / stdev_x)^2 ) )
 *
 *          Each feature dimension is scaled by its standard deviation across
 *          all DB entries so no single feature dominates by magnitude.
 *
 *          Supports:
 *            - K-NN with configurable K (majority vote for K > 1)
 *            - Scaled Euclidean distance (required)
 *            - Cosine distance (extension — for comparison)
 *            - Unknown object detection via confidence threshold
 *              If best distance > threshold → label = "unknown"
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#pragma once

#include <string>
#include <vector>
#include "AppState.h"
#include "ObjectDB.h"

// =============================================================================
// ClassifyResult — result for one region
// =============================================================================
struct ClassifyResult {
    std::string label       = "unknown";
    float       distance    = 1e9f;     ///< Best match distance (lower = better)
    float       confidence  = 0.f;      ///< 1 / (1 + distance), [0..1]
    bool        isUnknown   = true;
};

// =============================================================================
// Classifier
// =============================================================================
class Classifier {
public:
    /**
     * @brief Construct classifier and pre-compute feature stdevs from DB.
     *        Call refit() whenever the DB changes.
     *
     * @param db      Object database to classify against.
     * @param params  Pipeline params (kNeighbors, confidenceThresh,
     *                distanceMetric).
     */
    explicit Classifier(const ObjectDB& db, const PipelineParams& params);

    /**
     * @brief Recompute per-feature standard deviations from current DB.
     *        Must be called after DB entries are added or removed.
     */
    void refit(const ObjectDB& db);

    /**
     * @brief Classify a single feature vector.
     *
     * @param fv      Feature vector (fillRatio, bboxRatio, hu0..hu6).
     * @param params  Runtime params (k, threshold, metric).
     * @return        ClassifyResult with label, distance, confidence.
     */
    ClassifyResult classify(const std::vector<double>& fv,
                            const PipelineParams& params) const;

    /**
     * @brief Classify all regions in AppState in place.
     *        Writes label and confidence into each RegionInfo.
     *
     * @param state   AppState — reads and writes state.regions.
     * @param params  Pipeline params.
     */
    void classifyAll(AppState& state, const PipelineParams& params) const;

private:
    const ObjectDB*     db_;
    std::vector<double> stdevs_;    ///< Per-feature standard deviations
    std::vector<double> means_;     ///< Per-feature means (for reference)

    static constexpr int kFeatureDim = 9; // fillRatio, bboxRatio, hu0..hu6

    /** Scaled Euclidean distance between two feature vectors. */
    float scaledEuclidean(const std::vector<double>& a,
                          const std::vector<double>& b) const;

    /** Cosine distance between two feature vectors. */
    float cosineDistance(const std::vector<double>& a,
                         const std::vector<double>& b) const;
};
