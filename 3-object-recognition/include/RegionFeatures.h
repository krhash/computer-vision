/**
 * @file    RegionFeatures.h
 * @brief   Region-based feature extraction for 2D object recognition.
 *
 *          Computes shape features from region pixel statistics (not boundary).
 *          All features are translation, scale, and rotation invariant.
 *
 *          Features computed per region:
 *            1. angle        — primary axis of least central moment (radians)
 *            2. fillRatio    — area / orientedBBox.area  [0..1]
 *            3. bboxRatio    — oriented bbox height/width ratio
 *            4. huMoments[7] — Hu moment invariants (log-scaled)
 *
 *          Visualisation drawn on output frame:
 *            - Primary axis line through centroid
 *            - Oriented bounding box (rotates with object)
 *            - Feature values as text overlay
 *
 *          Uses cv::moments() for raw moment calculation.
 *          Hu moments are computed via cv::HuMoments().
 *          Axis angle and oriented bbox derived from central moments mu20, mu02, mu11.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "AppState.h"

/**
 * @brief Compute features for a single region.
 *
 *        Extracts a binary mask for the given region from the label map,
 *        computes cv::moments on the mask, derives axis angle, oriented
 *        bounding box, fill ratio, bbox ratio, and Hu moments.
 *        Results are written directly into the RegionInfo struct.
 *
 * @param labelMap   Label map from connected components (CV_32SC1).
 * @param reg        RegionInfo to update — reads id, writes all features.
 */
void computeRegionFeatures(const cv::Mat& labelMap, RegionInfo& reg);

/**
 * @brief Compute features for all regions in AppState.
 *
 * @param labelMap   Label map (CV_32SC1).
 * @param state      AppState — iterates state.regions and updates each.
 */
void computeAllFeatures(const cv::Mat& labelMap, AppState& state);

/**
 * @brief Draw feature visualisations on the display frame.
 *
 *        For each region draws:
 *          - Primary axis line (length proportional to region extent)
 *          - Oriented bounding box (rotated rectangle)
 *          - Feature text overlay (fillRatio, bboxRatio, hu[0])
 *
 * @param frame    BGR image to draw on (modified in place).
 * @param state    AppState containing computed regions.
 * @param params   Pipeline params (showAxes, showOrientedBBox, showFeatureText).
 */
void drawFeatures(cv::Mat& frame, const AppState& state,
                  const PipelineParams& params);