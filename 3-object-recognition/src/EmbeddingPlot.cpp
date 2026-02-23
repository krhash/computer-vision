/**
 * @file    EmbeddingPlot.cpp
 * @brief   2D PCA scatter plot of CNN embeddings.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#include "EmbeddingPlot.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// Fixed color palette per label — BGR
static const cv::Scalar kColors[] = {
    {  0, 200, 255}, {  0, 255, 100}, {255, 100,   0},
    {200,   0, 255}, {255, 255,   0}, {  0, 150, 255},
    {100, 255, 100}, {255,   0, 150}, {150, 150, 255},
    {255, 200, 100}
};
static constexpr int kNumColors = 10;

// -----------------------------------------------------------------------------
void renderEmbeddingPlot(const EmbeddingDB& db, int plotSize, cv::Mat& dst)
{
    const auto& entries = db.entries();

    if (entries.size() < 2) {
        dst = cv::Mat(plotSize, plotSize, CV_8UC3, cv::Scalar(30,30,30));
        cv::putText(dst, "Need >= 2 embedding samples",
                    {20, plotSize/2}, cv::FONT_HERSHEY_SIMPLEX,
                    0.5, {180,180,180}, 1);
        return;
    }

    int n   = static_cast<int>(entries.size());
    int dim = static_cast<int>(entries[0].embedding.size());

    // --- Build data matrix (n x dim) ----------------------------------------
    cv::Mat data(n, dim, CV_32F);
    for (int i = 0; i < n; i++) {
        const auto& emb = entries[i].embedding;
        for (int d = 0; d < dim && d < static_cast<int>(emb.size()); d++)
            data.at<float>(i, d) = emb[d];
    }

    // --- PCA — project onto top 2 eigenvectors via cv::PCACompute ------------
    cv::Mat mean, eigenvectors;
    cv::PCACompute(data, mean, eigenvectors, 2);

    // Project all points: projected = (data - mean) * eigenvectors^T
    cv::Mat projected;
    cv::PCAProject(data, mean, eigenvectors, projected);
    // projected is (n x 2)

    // --- Find range for normalisation ----------------------------------------
    float minX =  1e9f, maxX = -1e9f;
    float minY =  1e9f, maxY = -1e9f;
    for (int i = 0; i < n; i++) {
        float x = projected.at<float>(i, 0);
        float y = projected.at<float>(i, 1);
        minX = std::min(minX, x); maxX = std::max(maxX, x);
        minY = std::min(minY, y); maxY = std::max(maxY, y);
    }
    float rangeX = (maxX - minX) < 1e-6f ? 1.f : maxX - minX;
    float rangeY = (maxY - minY) < 1e-6f ? 1.f : maxY - minY;

    // --- Build label → color map ---------------------------------------------
    std::map<std::string, cv::Scalar> colorMap;
    std::map<std::string, int>        labelIdx;
    int idx = 0;
    for (const auto& e : entries) {
        if (colorMap.find(e.label) == colorMap.end()) {
            colorMap[e.label] = kColors[idx % kNumColors];
            labelIdx[e.label] = idx;
            idx++;
        }
    }

    // --- Draw plot -----------------------------------------------------------
    int margin = 40;
    int inner  = plotSize - 2 * margin;

    dst = cv::Mat(plotSize, plotSize, CV_8UC3, cv::Scalar(25, 25, 25));

    // Grid lines
    for (int g = 0; g <= 4; g++) {
        int gx = margin + g * inner / 4;
        int gy = margin + g * inner / 4;
        cv::line(dst, {gx, margin}, {gx, plotSize-margin}, {50,50,50}, 1);
        cv::line(dst, {margin, gy}, {plotSize-margin, gy}, {50,50,50}, 1);
    }

    // Axes labels
    cv::putText(dst, "PC1", {plotSize/2 - 10, plotSize - 8},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, {120,120,120}, 1);
    cv::putText(dst, "PC2", {4, plotSize/2},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, {120,120,120}, 1);

    // Plot points
    for (int i = 0; i < n; i++) {
        float nx = (projected.at<float>(i, 0) - minX) / rangeX;
        float ny = (projected.at<float>(i, 1) - minY) / rangeY;

        int px = margin + static_cast<int>(nx * inner);
        int py = margin + static_cast<int>((1.f - ny) * inner); // flip Y

        const cv::Scalar& col = colorMap[entries[i].label];
        cv::circle(dst, {px, py}, 6, col, -1);
        cv::circle(dst, {px, py}, 6, {255,255,255}, 1); // white outline
    }

    // Legend — bottom left
    int ly = plotSize - margin - static_cast<int>(colorMap.size()) * 18;
    for (const auto& kv : colorMap) {
        cv::circle(dst, {margin + 6, ly}, 5, kv.second, -1);
        cv::putText(dst, kv.first, {margin + 16, ly + 4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, kv.second, 1);
        ly += 18;
    }

    // Title
    cv::putText(dst, "Embedding Space (PCA 2D)",
                {margin, 18}, cv::FONT_HERSHEY_SIMPLEX,
                0.45, {200,200,200}, 1);
}
