/**
 * @file    ObjectDB.cpp
 * @brief   Object database implementation — CSV-based feature vector store.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#include "ObjectDB.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

// -----------------------------------------------------------------------------
// DBEntry
// -----------------------------------------------------------------------------
std::vector<double> DBEntry::toFeatureVector() const
{
    // Feature vector layout: [fillRatio, bboxRatio, hu0..hu6] = 9 values
    std::vector<double> fv;
    fv.reserve(9);
    fv.push_back(fillRatio);
    fv.push_back(bboxRatio);
    for (int i = 0; i < 7 && i < static_cast<int>(huMoments.size()); i++)
        fv.push_back(huMoments[i]);
    // Pad with zeros if hu moments are missing
    while (static_cast<int>(fv.size()) < 9) fv.push_back(0.0);
    return fv;
}

// -----------------------------------------------------------------------------
// ObjectDB
// -----------------------------------------------------------------------------
ObjectDB::ObjectDB(const std::string& filepath)
    : filepath_(filepath)
{
    // Auto-load if file exists
    std::ifstream f(filepath_);
    if (f.good()) load();
}

// -----------------------------------------------------------------------------
bool ObjectDB::load()
{
    std::ifstream file(filepath_);
    if (!file.is_open()) {
        std::cerr << "[ObjectDB] Cannot open: " << filepath_ << "\n";
        return false;
    }

    entries_.clear();
    std::string line;
    int lineNum = 0;

    while (std::getline(file, line)) {
        lineNum++;
        if (line.empty()) continue;
        // Skip header row
        if (line.find("label") == 0) continue;

        std::istringstream ss(line);
        std::string token;
        DBEntry entry;

        // Parse label
        if (!std::getline(ss, token, ',')) continue;
        entry.label = token;

        // Parse fillRatio
        if (!std::getline(ss, token, ',')) continue;
        try { entry.fillRatio = std::stod(token); } catch (...) { continue; }

        // Parse bboxRatio
        if (!std::getline(ss, token, ',')) continue;
        try { entry.bboxRatio = std::stod(token); } catch (...) { continue; }

        // Parse 7 Hu moments
        entry.huMoments.reserve(7);
        for (int i = 0; i < 7; i++) {
            if (!std::getline(ss, token, ',')) break;
            try { entry.huMoments.push_back(std::stod(token)); }
            catch (...) { entry.huMoments.push_back(0.0); }
        }

        entries_.push_back(entry);
    }

    std::cout << "[ObjectDB] Loaded " << entries_.size()
              << " entries from " << filepath_ << "\n";
    return true;
}

// -----------------------------------------------------------------------------
bool ObjectDB::save() const
{
    // Ensure directory exists
    std::filesystem::path p(filepath_);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());

    std::ofstream file(filepath_);
    if (!file.is_open()) {
        std::cerr << "[ObjectDB] Cannot write: " << filepath_ << "\n";
        return false;
    }

    // Write header
    file << kHeader << "\n";

    // Write entries
    for (const auto& e : entries_) {
        file << e.label << ","
             << e.fillRatio << ","
             << e.bboxRatio;
        for (int i = 0; i < 7; i++) {
            file << ",";
            if (i < static_cast<int>(e.huMoments.size()))
                file << e.huMoments[i];
            else
                file << "0.0";
        }
        file << "\n";
    }

    std::cout << "[ObjectDB] Saved " << entries_.size()
              << " entries to " << filepath_ << "\n";
    return true;
}

// -----------------------------------------------------------------------------
bool ObjectDB::append(const DBEntry& entry)
{
    // Ensure directory exists
    std::filesystem::path p(filepath_);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());

    // Check if file needs a header
    bool needsHeader = true;
    {
        std::ifstream f(filepath_);
        needsHeader = !f.good();
    }

    std::ofstream file(filepath_, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "[ObjectDB] Cannot append to: " << filepath_ << "\n";
        return false;
    }

    if (needsHeader) file << kHeader << "\n";

    file << entry.label << ","
         << entry.fillRatio << ","
         << entry.bboxRatio;
    for (int i = 0; i < 7; i++) {
        file << ",";
        if (i < static_cast<int>(entry.huMoments.size()))
            file << entry.huMoments[i];
        else
            file << "0.0";
    }
    file << "\n";

    // Also keep in memory
    entries_.push_back(entry);
    return true;
}

// -----------------------------------------------------------------------------
void ObjectDB::addEntry(const DBEntry& entry)
{
    entries_.push_back(entry);
}

// -----------------------------------------------------------------------------
DBEntry ObjectDB::entryFromRegion(const RegionInfo& reg,
                                   const std::string& label)
{
    DBEntry e;
    e.label      = label;
    e.fillRatio  = reg.fillRatio;
    e.bboxRatio  = reg.bboxRatio;
    e.huMoments  = reg.huMoments;
    return e;
}

// -----------------------------------------------------------------------------
void ObjectDB::deleteLabel(const std::string& label)
{
    entries_.erase(
        std::remove_if(entries_.begin(), entries_.end(),
                       [&](const DBEntry& e){ return e.label == label; }),
        entries_.end());
    save(); // persist deletion
}

// -----------------------------------------------------------------------------
void ObjectDB::clear()
{
    entries_.clear();
}

// -----------------------------------------------------------------------------
std::vector<DBEntry> ObjectDB::entriesForLabel(const std::string& label) const
{
    std::vector<DBEntry> result;
    for (const auto& e : entries_)
        if (e.label == label) result.push_back(e);
    return result;
}

// -----------------------------------------------------------------------------
std::map<std::string, int> ObjectDB::labelCounts() const
{
    std::map<std::string, int> counts;
    for (const auto& e : entries_)
        counts[e.label]++;
    return counts;
}

// -----------------------------------------------------------------------------
std::vector<std::string> ObjectDB::labels() const
{
    std::vector<std::string> lbls;
    for (const auto& kv : labelCounts())
        lbls.push_back(kv.first);
    return lbls;
}
