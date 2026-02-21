/**
 * @file    ConnectedComponents.h
 * @brief   Connected components labeling — written from scratch.
 *
 *          Implements the two-pass algorithm using a union-find (disjoint set)
 *          data structure to label connected foreground regions in a binary
 *          image without using cv::connectedComponents or related OpenCV calls.
 *
 *          Why from scratch:
 *            Satisfies the solo-developer requirement alongside Morphology.cpp.
 *
 *          Algorithm overview:
 *            Pass 1 — scan left-to-right, top-to-bottom; assign provisional
 *                     labels and record equivalences when regions merge.
 *            Pass 2 — replace every provisional label with its canonical
 *                     (root) label from the union-find structure.
 *
 *          After labeling, regions are filtered by minimum area and ranked
 *          by size.  Each surviving region is assigned a display color from
 *          a fixed palette so colors are stable frame-to-frame.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "AppState.h"

// =============================================================================
// Union-Find (Disjoint Set) — used internally by the two-pass algorithm
// =============================================================================

/**
 * @brief Simple union-find structure for label equivalence tracking.
 */
struct UnionFind {
    std::vector<int> parent; ///< parent[i] = parent label of label i

    explicit UnionFind(int n) : parent(n) {
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    /** Find root label with path compression. */
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    /** Union two labels — smaller root becomes child. */
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a != b) parent[b] = a;
    }
};

// =============================================================================
// Public API
// =============================================================================

/**
 * @brief Run two-pass connected components on a binary image from scratch.
 *
 * @param binary     Input binary image (CV_8UC1, 0=background, 255=foreground).
 * @param labelMap   Output label map (CV_32SC1); 0=background, 1..N=regions.
 * @return           Number of foreground labels found (before size filtering).
 */
int twoPassLabel(const cv::Mat& binary, cv::Mat& labelMap);

/**
 * @brief Compute per-region stats (area, centroid, bounding box) from labelMap.
 *
 * @param labelMap   Label map from twoPassLabel (CV_32SC1).
 * @param numLabels  Total number of labels including background (label 0).
 * @param areas      Output: pixel area per label index.
 * @param centroids  Output: centroid (cx,cy) per label index.
 * @param bboxes     Output: axis-aligned bounding box per label index.
 */
void computeRegionStats(const cv::Mat& labelMap, int numLabels,
                        std::vector<int>&         areas,
                        std::vector<cv::Point2f>& centroids,
                        std::vector<cv::Rect>&    bboxes);

/**
 * @brief Build color-coded region display image from label map.
 *
 *        Each surviving region is drawn in a unique color from a fixed
 *        palette.  Background is black.
 *
 * @param labelMap      Label map (CV_32SC1).
 * @param regions       Filtered region list (provides ids and display colors).
 * @param dst           Output BGR color image (same size as labelMap).
 */
void buildRegionDisplay(const cv::Mat& labelMap,
                        const std::vector<RegionInfo>& regions,
                        cv::Mat& dst);

/**
 * @brief Full connected components pipeline stage.
 *
 * @param cleaned    Binary image after morphology (CV_8UC1).
 * @param state      AppState — regions and frameRegions written here.
 * @param params     Pipeline parameters (minRegionArea, maxRegions).
 * @param labelMap   Output label map (CV_32SC1) — needed by Task 4 features.
 */
void findRegions(const cv::Mat& cleaned, AppState& state,
                 const PipelineParams& params, cv::Mat& labelMap);