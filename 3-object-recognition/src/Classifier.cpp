/**
 * @file    Classifier.cpp
 * @brief   K-NN classifier implementation with scaled Euclidean distance.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#include "Classifier.h"
#include <cmath>
#include <algorithm>
#include <map>
#include <numeric>
#include <iostream>

// -----------------------------------------------------------------------------
Classifier::Classifier(const ObjectDB& db, const PipelineParams& params)
    : db_(&db)
{
    refit(db);
    (void)params;
}

// -----------------------------------------------------------------------------
void Classifier::refit(const ObjectDB& db)
{
    db_ = &db;
    const auto& entries = db.entries();

    if (entries.empty()) {
        stdevs_.assign(kFeatureDim, 1.0);
        means_ .assign(kFeatureDim, 0.0);
        return;
    }

    // Accumulate sum and sum-of-squares per feature dimension
    std::vector<double> sum(kFeatureDim, 0.0);
    std::vector<double> sumSq(kFeatureDim, 0.0);
    int n = 0;

    for (const auto& e : entries) {
        auto fv = e.toFeatureVector();
        for (int d = 0; d < kFeatureDim && d < static_cast<int>(fv.size()); d++) {
            sum[d]   += fv[d];
            sumSq[d] += fv[d] * fv[d];
        }
        n++;
    }

    means_ .resize(kFeatureDim);
    stdevs_.resize(kFeatureDim);

    for (int d = 0; d < kFeatureDim; d++) {
        means_[d] = sum[d] / n;
        double var = (sumSq[d] / n) - (means_[d] * means_[d]);
        // Guard: if stdev ~ 0 all entries have same value → no scaling needed
        stdevs_[d] = (var > 1e-10) ? std::sqrt(var) : 1.0;
    }
}

// -----------------------------------------------------------------------------
float Classifier::scaledEuclidean(const std::vector<double>& a,
                                   const std::vector<double>& b) const
{
    double sum = 0.0;
    int dim = std::min({static_cast<int>(a.size()),
                        static_cast<int>(b.size()),
                        kFeatureDim});
    for (int d = 0; d < dim; d++) {
        double diff = (a[d] - b[d]) / stdevs_[d];
        sum += diff * diff;
    }
    return static_cast<float>(std::sqrt(sum));
}

// -----------------------------------------------------------------------------
float Classifier::cosineDistance(const std::vector<double>& a,
                                  const std::vector<double>& b) const
{
    double dot = 0.0, normA = 0.0, normB = 0.0;
    int dim = std::min({static_cast<int>(a.size()),
                        static_cast<int>(b.size()),
                        kFeatureDim});
    for (int d = 0; d < dim; d++) {
        dot   += a[d] * b[d];
        normA += a[d] * a[d];
        normB += b[d] * b[d];
    }
    if (normA < 1e-10 || normB < 1e-10) return 1.f;
    // Cosine distance = 1 - cosine similarity
    return static_cast<float>(1.0 - dot / (std::sqrt(normA) * std::sqrt(normB)));
}

// -----------------------------------------------------------------------------
ClassifyResult Classifier::classify(const std::vector<double>& fv,
                                     const PipelineParams& params) const
{
    ClassifyResult result;
    const auto& entries = db_->entries();

    if (entries.empty()) {
        result.label     = "no DB";
        result.isUnknown = true;
        return result;
    }

    // --- Compute distance to every DB entry ----------------------------------
    struct Match {
        std::string label;
        float       dist;
    };
    std::vector<Match> matches;
    matches.reserve(entries.size());

    for (const auto& e : entries) {
        auto dbFv = e.toFeatureVector();
        float dist = (params.distanceMetric == 1)
                     ? cosineDistance(fv, dbFv)
                     : scaledEuclidean(fv, dbFv);
        matches.push_back({e.label, dist});
    }

    // Sort by distance ascending
    std::sort(matches.begin(), matches.end(),
              [](const Match& a, const Match& b){ return a.dist < b.dist; });

    // --- K-NN majority vote --------------------------------------------------
    int k = std::min(params.kNeighbors, static_cast<int>(matches.size()));
    std::map<std::string, int> votes;
    for (int i = 0; i < k; i++)
        votes[matches[i].label]++;

    // Find label with most votes (tie → closest distance wins)
    std::string bestLabel = matches[0].label;
    int bestVotes = 0;
    for (const auto& kv : votes) {
        if (kv.second > bestVotes) {
            bestVotes = kv.second;
            bestLabel = kv.first;
        }
    }

    float bestDist = matches[0].dist;

    // --- Confidence = 1 / (1 + distance) ------------------------------------
    float confidence = 1.f / (1.f + bestDist);

    // --- Unknown detection ---------------------------------------------------
    // If best distance exceeds threshold → object not in DB
    bool isUnknown = (bestDist > params.confidenceThresh);

    result.label      = isUnknown ? "unknown" : bestLabel;
    result.distance   = bestDist;
    result.confidence = confidence;
    result.isUnknown  = isUnknown;

    return result;
}

// -----------------------------------------------------------------------------
void Classifier::classifyAll(AppState& state,
                              const PipelineParams& params) const
{
    for (auto& reg : state.regions) {
        if (reg.huMoments.empty()) continue;

        DBEntry tmp     = ObjectDB::entryFromRegion(reg, "");
        auto fv         = tmp.toFeatureVector();
        auto result     = classify(fv, params);

        reg.label      = result.label;
        reg.confidence = result.confidence;
    }
}
