/**
 * @file    ConnectedComponents.cpp
 * @brief   From-scratch two-pass connected components implementation.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#include "ConnectedComponents.h"
#include <algorithm>
#include <numeric>

// -----------------------------------------------------------------------------
// Fixed color palette for region display — 16 visually distinct BGR colors
// -----------------------------------------------------------------------------
static const cv::Scalar kPalette[] = {
    {  0, 114, 189}, {217,  83,  25}, {237, 177,  32}, {126,  47, 142},
    { 77, 190,  238}, { 162, 20,  47}, {118, 171,  47}, { 76,  76,  76},
    {153, 153, 255}, {255, 128,   0}, {  0, 204, 153}, {204,   0, 102},
    {102, 204,   0}, {  0, 102, 204}, {255,  51, 153}, { 51, 255, 153}
};
static constexpr int kPaletteSize = 16;

// -----------------------------------------------------------------------------
int twoPassLabel(const cv::Mat& binary, cv::Mat& labelMap)
{
    CV_Assert(binary.type() == CV_8UC1);

    labelMap = cv::Mat::zeros(binary.size(), CV_32SC1);

    // Union-Find — pre-allocate for worst case (half of all pixels)
    int maxLabels = (binary.rows * binary.cols) / 2 + 2;
    UnionFind uf(maxLabels);

    int nextLabel = 1; // 0 is background

    // -------------------------------------------------------------------------
    // Pass 1 — assign provisional labels, record equivalences
    // -------------------------------------------------------------------------
    for (int r = 0; r < binary.rows; r++) {
        const uchar*  binRow = binary.ptr<uchar>(r);
        int*          lblRow = labelMap.ptr<int>(r);

        for (int c = 0; c < binary.cols; c++) {
            if (binRow[c] == 0) continue; // background pixel — skip

            // Collect labels from already-visited 4-connected neighbours
            // (left and top only — we scan left-to-right, top-to-bottom)
            int left = (c > 0)             ? labelMap.at<int>(r,   c-1) : 0;
            int top  = (r > 0)             ? labelMap.at<int>(r-1, c  ) : 0;

            if (left == 0 && top == 0) {
                // No labelled neighbours — assign new label
                lblRow[c] = nextLabel++;
            } else if (left != 0 && top == 0) {
                lblRow[c] = left;
            } else if (left == 0 && top != 0) {
                lblRow[c] = top;
            } else {
                // Both neighbours labelled — use smaller root, record equivalence
                int rootL = uf.find(left);
                int rootT = uf.find(top);
                lblRow[c] = std::min(rootL, rootT);
                if (rootL != rootT) uf.unite(rootL, rootT);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Pass 2 — replace every label with its canonical root
    // -------------------------------------------------------------------------
    // Build compact re-numbering: root → sequential id
    std::vector<int> labelRemap(nextLabel, 0);
    int compactId = 0;
    for (int lbl = 1; lbl < nextLabel; lbl++) {
        if (uf.find(lbl) == lbl) labelRemap[lbl] = ++compactId;
    }
    // Propagate remap through non-root labels
    for (int lbl = 1; lbl < nextLabel; lbl++) {
        labelRemap[lbl] = labelRemap[uf.find(lbl)];
    }

    // Apply remap to labelMap
    for (int r = 0; r < labelMap.rows; r++) {
        int* row = labelMap.ptr<int>(r);
        for (int c = 0; c < labelMap.cols; c++) {
            if (row[c] > 0) row[c] = labelRemap[row[c]];
        }
    }

    return compactId; // number of foreground components
}

// -----------------------------------------------------------------------------
void computeRegionStats(const cv::Mat& labelMap, int numLabels,
                        std::vector<int>&         areas,
                        std::vector<cv::Point2f>& centroids,
                        std::vector<cv::Rect>&    bboxes)
{
    // Index 0 = background, indices 1..numLabels = regions
    areas    .assign(numLabels + 1, 0);
    centroids.assign(numLabels + 1, {0.f, 0.f});
    bboxes   .assign(numLabels + 1, {0, 0, 0, 0});

    // Accumulate sum of coordinates for centroid, track bbox extents
    std::vector<int> minR(numLabels+1, INT_MAX), maxR(numLabels+1, INT_MIN);
    std::vector<int> minC(numLabels+1, INT_MAX), maxC(numLabels+1, INT_MIN);
    std::vector<double> sumR(numLabels+1, 0.0), sumC(numLabels+1, 0.0);

    for (int r = 0; r < labelMap.rows; r++) {
        const int* row = labelMap.ptr<int>(r);
        for (int c = 0; c < labelMap.cols; c++) {
            int lbl = row[c];
            if (lbl == 0) continue;
            areas[lbl]++;
            sumR[lbl] += r;
            sumC[lbl] += c;
            minR[lbl] = std::min(minR[lbl], r);
            maxR[lbl] = std::max(maxR[lbl], r);
            minC[lbl] = std::min(minC[lbl], c);
            maxC[lbl] = std::max(maxC[lbl], c);
        }
    }

    for (int lbl = 1; lbl <= numLabels; lbl++) {
        if (areas[lbl] == 0) continue;
        centroids[lbl] = { static_cast<float>(sumC[lbl] / areas[lbl]),
                           static_cast<float>(sumR[lbl] / areas[lbl]) };
        bboxes[lbl] = { minC[lbl], minR[lbl],
                        maxC[lbl] - minC[lbl] + 1,
                        maxR[lbl] - minR[lbl] + 1 };
    }
}

// -----------------------------------------------------------------------------
void buildRegionDisplay(const cv::Mat& labelMap,
                        const std::vector<RegionInfo>& regions,
                        cv::Mat& dst)
{
    dst = cv::Mat::zeros(labelMap.size(), CV_8UC3);

    // Build a fast id→color lookup
    // Max label id won't exceed labelMap pixel max
    double maxVal;
    cv::minMaxLoc(labelMap, nullptr, &maxVal);
    int maxLbl = static_cast<int>(maxVal);
    std::vector<cv::Scalar> colorMap(maxLbl + 1, cv::Scalar(0,0,0));

    for (const auto& reg : regions)
        if (reg.id <= maxLbl) colorMap[reg.id] = reg.displayColor;

    for (int r = 0; r < labelMap.rows; r++) {
        const int* lblRow = labelMap.ptr<int>(r);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(r);
        for (int c = 0; c < labelMap.cols; c++) {
            int lbl = lblRow[c];
            if (lbl == 0 || lbl > maxLbl) continue;
            const cv::Scalar& col = colorMap[lbl];
            dstRow[c] = { static_cast<uchar>(col[0]),
                          static_cast<uchar>(col[1]),
                          static_cast<uchar>(col[2]) };
        }
    }
}

// -----------------------------------------------------------------------------
void findRegions(const cv::Mat& cleaned, AppState& state,
                 const PipelineParams& params, cv::Mat& labelMap)
{
    state.regions.clear();

    // --- Two-pass labeling ---------------------------------------------------
    int numLabels = twoPassLabel(cleaned, labelMap);
    if (numLabels == 0) {
        state.frameRegions = cv::Mat::zeros(cleaned.size(), CV_8UC3);
        return;
    }

    // --- Compute stats -------------------------------------------------------
    std::vector<int>         areas;
    std::vector<cv::Point2f> centroids;
    std::vector<cv::Rect>    bboxes;
    computeRegionStats(labelMap, numLabels, areas, centroids, bboxes);

    // --- Filter by min area, sort by area descending -------------------------
    std::vector<int> validIds;
    for (int lbl = 1; lbl <= numLabels; lbl++) {
        if (areas[lbl] >= params.minRegionArea)
            validIds.push_back(lbl);
    }
    std::sort(validIds.begin(), validIds.end(),
              [&](int a, int b){ return areas[a] > areas[b]; });

    // Keep top N
    int keep = std::min(static_cast<int>(validIds.size()), params.maxRegions);

    // --- Populate RegionInfo -------------------------------------------------
    for (int i = 0; i < keep; i++) {
        int lbl = validIds[i];
        RegionInfo reg;
        reg.id           = lbl;
        reg.boundingBox  = bboxes[lbl];
        reg.centroid     = centroids[lbl];
        reg.area         = static_cast<double>(areas[lbl]);
        reg.displayColor = kPalette[i % kPaletteSize];
        state.regions.push_back(reg);
    }

    // --- Build color display image -------------------------------------------
    buildRegionDisplay(labelMap, state.regions, state.frameRegions);
}