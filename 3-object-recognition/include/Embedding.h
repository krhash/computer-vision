/**
 * @file    Embedding.h
 * @brief   CNN-based one-shot classification using ResNet18 embeddings.
 *
 *          Implements Task 9 — one-shot classification:
 *            1. Extract aligned ROI from region (via utilities.cpp)
 *            2. Run ROI through ResNet18 to get 512-dim embedding
 *            3. Compare embeddings using sum-squared difference
 *            4. Classify by nearest embedding in training set
 *
 *          Training DB stored separately from hand-feature DB:
 *            data/db/embeddings.csv  — label + 512 float values
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include "AppState.h"

// =============================================================================
// EmbeddingEntry — one labeled embedding vector
// =============================================================================
struct EmbeddingEntry {
    std::string        label;
    std::vector<float> embedding; ///< 512-dimensional ResNet18 output
};

// =============================================================================
// EmbeddingDB — stores labeled embeddings for one-shot classification
// =============================================================================
class EmbeddingDB {
public:
    /**
     * @brief Construct and optionally auto-load the embedding database.
     * @param filepath  Path to the CSV file (created on first save if absent).
     */
    explicit EmbeddingDB(const std::string& filepath =
                         "data/db/embeddings.csv");

    /**
     * @brief Load all entries from the CSV file, replacing in-memory state.
     * @return True on success, false if the file could not be opened.
     */
    bool load();

    /**
     * @brief Write all in-memory entries to the CSV file.
     * @return True on success, false if the file could not be written.
     */
    bool save() const;

    /**
     * @brief Append a single entry to the CSV file and to the in-memory list.
     * @param entry  Labeled embedding to store.
     * @return       True on success.
     */
    bool append(const EmbeddingEntry& entry);

    /** Clear all in-memory entries (does not modify the CSV file). */
    void clear();

    /** Return a read-only reference to all stored entries. */
    const std::vector<EmbeddingEntry>& entries() const { return entries_; }

    /** Return true if the database contains no entries. */
    bool  empty() const { return entries_.empty(); }

    /** Return the total number of stored entries. */
    int   size()  const { return static_cast<int>(entries_.size()); }

private:
    std::string                filepath_;
    std::vector<EmbeddingEntry> entries_;
};

// =============================================================================
// EmbeddingClassifier — loads ResNet18 and classifies via embedding distance
// =============================================================================
class EmbeddingClassifier {
public:
    /**
     * @brief Load ResNet18 ONNX model.
     * @param modelPath  Path to resnet18-v2-7.onnx
     * @return           True if model loaded successfully.
     */
    bool loadModel(const std::string& modelPath =
                   "data/models/resnet18-v2-7.onnx");

    /** Return true if model is loaded and ready. */
    bool isReady() const { return modelLoaded_; }

    /**
     * @brief Compute embedding for a region in the frame.
     *
     *        Uses prepEmbeddingImage() then getEmbedding() internally.
     *
     * @param frame   Original BGR frame.
     * @param reg     RegionInfo with centroid, angle, axis extents.
     * @param emb     Output embedding vector (512 floats).
     * @param debug   Show intermediate images if true.
     * @return        True on success.
     */
    bool computeEmbedding(cv::Mat& frame, const RegionInfo& reg,
                          std::vector<float>& emb, bool debug = false,
                          cv::Mat* cropOut = nullptr);

    /**
     * @brief Classify a region using sum-squared distance to DB entries.
     *
     * @param emb           Query embedding.
     * @param db            Embedding database.
     * @param threshold     Max SSD distance to accept as known (0 = no limit).
     * @param outLabel      Output: best matching label or "unknown".
     * @param outDist       Output: best matching distance.
     */
    void classify(const std::vector<float>& emb,
                  const EmbeddingDB& db,
                  float threshold,
                  std::string& outLabel,
                  float& outDist) const;

    /**
     * @brief Classify all regions in AppState using embeddings.
     *
     * @param frame   Original BGR frame.
     * @param state   AppState — reads regions, writes embedding labels.
     * @param db      Embedding DB.
     * @param thresh  Distance threshold for unknown detection.
     */
    void classifyAll(cv::Mat& frame, AppState& state,
                     const EmbeddingDB& db, float thresh);

private:
    cv::dnn::Net net_;
    bool         modelLoaded_ = false;

    /** Sum-squared distance between two embedding vectors. */
    float ssdDistance(const std::vector<float>& a,
                      const std::vector<float>& b) const;
};
