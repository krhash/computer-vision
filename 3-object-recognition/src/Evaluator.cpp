/**
 * @file    Evaluator.cpp
 * @brief   Confusion matrix evaluation implementation.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#include "Evaluator.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <filesystem>

// -----------------------------------------------------------------------------
void Evaluator::record(const std::string& trueLabel,
                        const std::string& predictedLabel,
                        float confidence)
{
    records_.push_back({trueLabel, predictedLabel, confidence});
}

// -----------------------------------------------------------------------------
std::vector<std::string> Evaluator::uniqueLabels() const
{
    std::vector<std::string> lbls;
    for (const auto& r : records_) {
        if (std::find(lbls.begin(), lbls.end(), r.trueLabel) == lbls.end())
            lbls.push_back(r.trueLabel);
        if (std::find(lbls.begin(), lbls.end(), r.predictedLabel) == lbls.end())
            lbls.push_back(r.predictedLabel);
    }
    std::sort(lbls.begin(), lbls.end());
    return lbls;
}

// -----------------------------------------------------------------------------
void Evaluator::buildMatrix(std::vector<std::string>& labels,
                             std::vector<std::vector<int>>& matrix) const
{
    labels = uniqueLabels();
    int n  = static_cast<int>(labels.size());

    matrix.assign(n, std::vector<int>(n, 0));

    // Build label → index lookup
    std::map<std::string, int> idx;
    for (int i = 0; i < n; i++) idx[labels[i]] = i;

    for (const auto& r : records_) {
        int row = idx[r.trueLabel];
        int col = idx[r.predictedLabel];
        matrix[row][col]++;
    }
}

// -----------------------------------------------------------------------------
float Evaluator::accuracy() const
{
    if (records_.empty()) return 0.f;
    int correct = 0;
    for (const auto& r : records_)
        if (r.trueLabel == r.predictedLabel) correct++;
    return static_cast<float>(correct) / static_cast<float>(records_.size());
}

// -----------------------------------------------------------------------------
void Evaluator::printMatrix() const
{
    std::vector<std::string>        labels;
    std::vector<std::vector<int>>   matrix;
    buildMatrix(labels, matrix);

    int n     = static_cast<int>(labels.size());
    int width = 12;

    std::cout << "\n=== Confusion Matrix ===\n";
    std::cout << "Rows = True label, Cols = Predicted label\n\n";

    // Header
    std::cout << std::setw(width) << "";
    for (const auto& l : labels)
        std::cout << std::setw(width) << l.substr(0, width-1);
    std::cout << "\n";

    // Divider
    std::cout << std::string(width * (n + 1), '-') << "\n";

    // Rows
    for (int i = 0; i < n; i++) {
        std::cout << std::setw(width) << labels[i].substr(0, width-1);
        for (int j = 0; j < n; j++) {
            // Mark diagonal (correct) with brackets
            if (i == j)
                std::cout << std::setw(width-2) << "[" + std::to_string(matrix[i][j]) + "]";
            else
                std::cout << std::setw(width) << matrix[i][j];
        }
        // Per-class accuracy
        int rowTotal = std::accumulate(matrix[i].begin(), matrix[i].end(), 0);
        float classAcc = rowTotal > 0
            ? static_cast<float>(matrix[i][i]) / rowTotal : 0.f;
        std::cout << "  " << std::fixed << std::setprecision(0)
                  << classAcc * 100.f << "%\n";
    }

    std::cout << "\nOverall accuracy: "
              << std::fixed << std::setprecision(1)
              << accuracy() * 100.f << "%"
              << "  (" << records_.size() << " samples)\n";
    std::cout << "========================\n\n";
}

// -----------------------------------------------------------------------------
void Evaluator::saveMatrix(const std::string& filepath) const
{
    std::vector<std::string>        labels;
    std::vector<std::vector<int>>   matrix;
    buildMatrix(labels, matrix);

    std::filesystem::path p(filepath);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());

    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "[Evaluator] Cannot write: " << filepath << "\n";
        return;
    }

    // Header row
    f << "true/predicted";
    for (const auto& l : labels) f << "," << l;
    f << ",accuracy\n";

    // Matrix rows
    int n = static_cast<int>(labels.size());
    for (int i = 0; i < n; i++) {
        f << labels[i];
        for (int j = 0; j < n; j++) f << "," << matrix[i][j];
        int rowTotal = std::accumulate(matrix[i].begin(), matrix[i].end(), 0);
        float acc = rowTotal > 0
            ? static_cast<float>(matrix[i][i]) / rowTotal : 0.f;
        f << "," << std::fixed << std::setprecision(2) << acc << "\n";
    }

    f << "overall," << std::string(n, ',')
      << std::fixed << std::setprecision(2) << accuracy() << "\n";

    std::cout << "[Evaluator] Saved to " << filepath << "\n";
}

// -----------------------------------------------------------------------------
void Evaluator::drawMatrix(cv::Mat& dst) const
{
    std::vector<std::string>        labels;
    std::vector<std::vector<int>>   matrix;
    buildMatrix(labels, matrix);

    int n        = static_cast<int>(labels.size());
    if (n == 0) {
        dst = cv::Mat(200, 400, CV_8UC3, cv::Scalar(40,40,40));
        cv::putText(dst, "No eval data yet. Press 'e' to record.",
                    {10, 100}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    {200,200,200}, 1);
        return;
    }

    int cellSize = 70;
    int margin   = 120; // left margin for labels
    int topMargin= 80;
    int imgW     = margin + n * cellSize + 10;
    int imgH     = topMargin + n * cellSize + 40;

    dst = cv::Mat(imgH, imgW, CV_8UC3, cv::Scalar(30, 30, 30));

    // Column headers (predicted)
    for (int j = 0; j < n; j++) {
        std::string lbl = labels[j].substr(0, 8);
        cv::putText(dst, lbl,
                    {margin + j * cellSize + 4, topMargin - 10},
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, {200,200,200}, 1);
    }

    // Row headers (true) + cells
    for (int i = 0; i < n; i++) {
        // Row label
        std::string lbl = labels[i].substr(0, 8);
        cv::putText(dst, lbl,
                    {4, topMargin + i * cellSize + cellSize/2},
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, {200,200,200}, 1);

        int rowTotal = std::accumulate(matrix[i].begin(), matrix[i].end(), 0);

        for (int j = 0; j < n; j++) {
            cv::Rect cell(margin + j * cellSize,
                          topMargin + i * cellSize,
                          cellSize - 2, cellSize - 2);

            // Color: diagonal = green intensity, off-diagonal = red intensity
            cv::Scalar cellColor;
            if (matrix[i][j] == 0) {
                cellColor = {50.0, 50.0, 50.0};
            } else if (i == j) {
                double intensity = rowTotal > 0
                    ? 200.0 * matrix[i][j] / rowTotal + 55.0
                    : 55.0;
                cellColor = {30.0, intensity, 30.0};
            } else {
                double intensity = rowTotal > 0
                    ? 200.0 * matrix[i][j] / rowTotal + 55.0
                    : 55.0;
                cellColor = {30.0, 30.0, intensity};
            }

            cv::rectangle(dst, cell, cellColor, -1);
            cv::rectangle(dst, cell, {80,80,80}, 1);

            // Value text
            std::string val = std::to_string(matrix[i][j]);
            cv::putText(dst, val,
                        {cell.x + cellSize/2 - 8, cell.y + cellSize/2 + 6},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, {255,255,255}, 1);
        }
    }

    // Accuracy footer
    std::string accTxt = "Accuracy: " +
        std::to_string(static_cast<int>(accuracy() * 100.f)) + "%" +
        "  (" + std::to_string(records_.size()) + " samples)";
    cv::putText(dst, accTxt, {4, imgH - 10},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 220, 255}, 1);
}
