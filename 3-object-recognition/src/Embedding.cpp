/**
 * @file    Embedding.cpp
 * @brief   CNN embedding classification implementation.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#include "Embedding.h"
#include "utilities.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <filesystem>

// =============================================================================
// EmbeddingDB
// =============================================================================
EmbeddingDB::EmbeddingDB(const std::string& filepath)
    : filepath_(filepath)
{
    std::ifstream f(filepath_);
    if (f.good()) load();
}

// -----------------------------------------------------------------------------
bool EmbeddingDB::load()
{
    std::ifstream file(filepath_);
    if (!file.is_open()) return false;

    entries_.clear();
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line.find("label") == 0) continue;

        std::istringstream ss(line);
        std::string token;
        EmbeddingEntry e;

        if (!std::getline(ss, token, ',')) continue;
        e.label = token;

        while (std::getline(ss, token, ',')) {
            try { e.embedding.push_back(std::stof(token)); }
            catch (...) {}
        }

        if (!e.embedding.empty())
            entries_.push_back(e);
    }

    std::cout << "[EmbeddingDB] Loaded " << entries_.size()
              << " entries from " << filepath_ << "\n";
    return true;
}

// -----------------------------------------------------------------------------
bool EmbeddingDB::save() const
{
    std::filesystem::path p(filepath_);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());

    std::ofstream file(filepath_);
    if (!file.is_open()) return false;

    file << "label";
    if (!entries_.empty())
        for (size_t i = 0; i < entries_[0].embedding.size(); i++)
            file << ",e" << i;
    file << "\n";

    for (const auto& e : entries_) {
        file << e.label;
        for (float v : e.embedding)
            file << "," << v;
        file << "\n";
    }
    return true;
}

// -----------------------------------------------------------------------------
bool EmbeddingDB::append(const EmbeddingEntry& entry)
{
    std::filesystem::path p(filepath_);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());

    bool needsHeader = true;
    { std::ifstream f(filepath_); needsHeader = !f.good(); }

    std::ofstream file(filepath_, std::ios::app);
    if (!file.is_open()) return false;

    if (needsHeader) {
        file << "label";
        for (size_t i = 0; i < entry.embedding.size(); i++)
            file << ",e" << i;
        file << "\n";
    }

    file << entry.label;
    for (float v : entry.embedding)
        file << "," << v;
    file << "\n";

    entries_.push_back(entry);
    return true;
}

// -----------------------------------------------------------------------------
void EmbeddingDB::clear() { entries_.clear(); }

// =============================================================================
// EmbeddingClassifier
// =============================================================================
bool EmbeddingClassifier::loadModel(const std::string& modelPath)
{
    try {
        net_ = cv::dnn::readNetFromONNX(modelPath);
        if (net_.empty()) {
            std::cerr << "[Embedding] Failed to load model: " << modelPath << "\n";
            return false;
        }
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        modelLoaded_ = true;
        std::cout << "[Embedding] ResNet18 loaded from " << modelPath << "\n";
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[Embedding] OpenCV error: " << e.what() << "\n";
        return false;
    }
}

// -----------------------------------------------------------------------------
bool EmbeddingClassifier::computeEmbedding(cv::Mat& frame,
                                            const RegionInfo& reg,
                                            std::vector<float>& emb,
                                            bool debug)
{
    if (!modelLoaded_) return false;

    // Step A-C: extract aligned ROI using Prof. Maxwell's utility
    cv::Mat embImage;
    prepEmbeddingImage(frame, embImage,
                       static_cast<int>(reg.centroid.x),
                       static_cast<int>(reg.centroid.y),
                       static_cast<float>(reg.angle),
                       reg.minE1, reg.maxE1,
                       reg.minE2, reg.maxE2,
                       debug ? 1 : 0);

    if (embImage.empty()) return false;

    // Step D: get embedding from ResNet18
    cv::Mat embMat;
    getEmbedding(embImage, embMat, net_, debug ? 1 : 0);

    // Convert cv::Mat row to std::vector<float>
    emb.assign(embMat.ptr<float>(0),
               embMat.ptr<float>(0) + static_cast<int>(embMat.total()));
    return true;
}

// -----------------------------------------------------------------------------
float EmbeddingClassifier::ssdDistance(const std::vector<float>& a,
                                        const std::vector<float>& b) const
{
    float ssd = 0.f;
    int   dim = std::min(a.size(), b.size());
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        ssd += d * d;
    }
    return ssd;
}

// -----------------------------------------------------------------------------
void EmbeddingClassifier::classify(const std::vector<float>& emb,
                                    const EmbeddingDB& db,
                                    float threshold,
                                    std::string& outLabel,
                                    float& outDist) const
{
    outLabel = "unknown";
    outDist  = 1e9f;

    for (const auto& e : db.entries()) {
        float d = ssdDistance(emb, e.embedding);
        if (d < outDist) {
            outDist  = d;
            outLabel = e.label;
        }
    }

    // Unknown detection — if best distance exceeds threshold
    if (threshold > 0.f && outDist > threshold)
        outLabel = "unknown";
}

// -----------------------------------------------------------------------------
void EmbeddingClassifier::classifyAll(cv::Mat& frame, AppState& state,
                                       const EmbeddingDB& db, float thresh)
{
    for (auto& reg : state.regions) {
        std::vector<float> emb;
        if (!computeEmbedding(frame, reg, emb)) continue;

        std::string label;
        float       dist;
        classify(emb, db, thresh, label, dist);

        reg.label      = label;
        reg.embedding  = emb;

        // Normalise confidence — SSD over 512 dims gives large numbers
        float normDist = dist / 512.f;
        reg.confidence = 1.f / (1.f + normDist);
    }
}