/**
 * @file    ObjectDB.h
 * @brief   Object database for storing and querying feature vectors.
 *
 *          Manages a CSV-based database of labeled feature vectors collected
 *          during training mode. Each entry stores:
 *            label, fillRatio, bboxRatio, hu[0..6]  (9 values total)
 *
 *          CSV format (data/db/objects.csv):
 *            label,fillRatio,bboxRatio,hu0,hu1,hu2,hu3,hu4,hu5,hu6
 *            scissors,0.823,3.241,-1.452,-4.123,-6.234,-7.891,...
 *
 *          Supports:
 *            - Load from / save to CSV file
 *            - Append single entry
 *            - Query all entries for a given label
 *            - List all unique labels and sample counts
 *            - Delete all entries for a label
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
// DBEntry — one labeled feature vector
// =============================================================================
struct DBEntry {
    std::string         label;
    double              fillRatio  = 0.0;
    double              bboxRatio  = 0.0;
    std::vector<double> huMoments;          ///< 7 Hu moment invariants

    /** Build feature vector as a flat double array for distance computation. */
    std::vector<double> toFeatureVector() const;
};

// =============================================================================
// ObjectDB
// =============================================================================
class ObjectDB {
public:
    /**
     * @brief Construct and optionally load existing DB from file.
     * @param filepath  Path to CSV file (created on first save if missing).
     */
    explicit ObjectDB(const std::string& filepath = "data/db/objects.csv");

    // --- Persistence ---------------------------------------------------------

    /** Load all entries from CSV. Returns true on success. */
    bool load();

    /** Save all entries to CSV. Returns true on success. */
    bool save() const;

    /** Append a single entry to CSV without rewriting the whole file. */
    bool append(const DBEntry& entry);

    // --- Data management -----------------------------------------------------

    /** Add entry to in-memory DB (does not save to disk). */
    void addEntry(const DBEntry& entry);

    /** Build a DBEntry from a RegionInfo and a label string. */
    static DBEntry entryFromRegion(const RegionInfo& reg,
                                   const std::string& label);

    /** Delete all entries with the given label. */
    void deleteLabel(const std::string& label);

    /** Clear all entries. */
    void clear();

    // --- Queries -------------------------------------------------------------

    /** Return all entries. */
    const std::vector<DBEntry>& entries() const { return entries_; }

    /** Return entries for a specific label. */
    std::vector<DBEntry> entriesForLabel(const std::string& label) const;

    /** Return map of label → sample count. */
    std::map<std::string, int> labelCounts() const;

    /** Return list of unique labels. */
    std::vector<std::string> labels() const;

    /** Return true if DB has at least one entry. */
    bool empty() const { return entries_.empty(); }

    /** Return total number of entries. */
    int size() const { return static_cast<int>(entries_.size()); }

    /** Return file path. */
    const std::string& filepath() const { return filepath_; }

private:
    std::string         filepath_;
    std::vector<DBEntry> entries_;

    static constexpr const char* kHeader =
        "label,fillRatio,bboxRatio,hu0,hu1,hu2,hu3,hu4,hu5,hu6";
};
