/**
 * @file    Evaluator.h
 * @brief   Confusion matrix and accuracy evaluation for Task 7.
 *
 *          Records true label vs predicted label for each classification,
 *          builds a confusion matrix, and computes per-class and overall
 *          accuracy.
 *
 *          Evaluation workflow:
 *            1. User places known object in frame
 *            2. Presses 'e' and types true label
 *            3. System records predicted label from classifier
 *            4. Repeat for all objects / positions
 *            5. Press 'p' to print confusion matrix to console
 *            6. Matrix saved to data/db/confusion_matrix.csv
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include "AppState.h"

// =============================================================================
// EvalRecord — one ground truth vs prediction pair
// =============================================================================
struct EvalRecord {
    std::string trueLabel;
    std::string predictedLabel;
    float       confidence = 0.f;
};

// =============================================================================
// Evaluator
// =============================================================================
class Evaluator {
public:
    /**
     * @brief Record one classification result.
     *
     * @param trueLabel       Ground truth label (entered by user).
     * @param predictedLabel  Label assigned by classifier.
     * @param confidence      Classifier confidence score.
     */
    void record(const std::string& trueLabel,
                const std::string& predictedLabel,
                float confidence);

    /**
     * @brief Build and return confusion matrix from recorded results.
     *
     *        Rows = true labels, Cols = predicted labels.
     *        matrix[i][j] = count of true=i predicted=j.
     *
     * @param labels  Output: ordered list of unique labels.
     * @param matrix  Output: NxN confusion matrix.
     */
    void buildMatrix(std::vector<std::string>& labels,
                     std::vector<std::vector<int>>& matrix) const;

    /**
     * @brief Compute overall accuracy from recorded results.
     * @return Accuracy in [0..1].
     */
    float accuracy() const;

    /**
     * @brief Print confusion matrix and accuracy to console.
     */
    void printMatrix() const;

    /**
     * @brief Save confusion matrix to CSV file.
     * @param filepath  Output CSV path.
     */
    void saveMatrix(const std::string& filepath =
                    "data/db/confusion_matrix.csv") const;

    /**
     * @brief Draw confusion matrix on an OpenCV image for display.
     *
     * @param dst   Output BGR image (created internally).
     */
    void drawMatrix(cv::Mat& dst) const;

    /** Return number of recorded evaluations. */
    int count() const { return static_cast<int>(records_.size()); }

    /** Clear all records. */
    void clear() { records_.clear(); }

private:
    std::vector<EvalRecord> records_;

    /** Get sorted unique labels from records. */
    std::vector<std::string> uniqueLabels() const;
};
