/**
 * @file    main.cpp
 * @brief   Entry point for the 2D Object Recognition system.
 *
 *          Usage:
 *            objectRecognition.exe --mode live  [--camera 0]
 *            objectRecognition.exe --mode image --input <path>
 *            objectRecognition.exe --mode train [--camera 0]
 *
 *          Keyboard controls:
 *            t/T     threshold -/+5
 *            b/B     blur kernel -/+2
 *            a       toggle adaptive threshold
 *            k       toggle ISODATA threshold
 *            s       toggle sat+intensity threshold
 *            m/M     morph kernel -/+2
 *            i/I     morph iterations -/+1
 *            o       cycle morph mode
 *            r/R     min region area -/+100
 *            n       enter train mode — prompts for label
 *            c       capture hand-feature sample → objects.csv
 *            C       capture CNN embedding sample → embeddings.csv
 *            x/X     confidence threshold -/+0.1
 *            d       toggle Euclidean / Cosine metric
 *            j/J     K neighbours -/+1
 *            e       record eval sample (prompts for true label)
 *            p       print + save confusion matrix
 *            1-3     toggle Threshold/Cleaned/Regions windows
 *            4-6     toggle Axes/BBox/FeatureText overlays
 *            7       toggle confusion matrix window
 *            8       toggle embedding crop windows
 *            9       toggle CNN embedding / hand feature mode
 *            q/ESC   quit
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>

#include "AppState.h"
#include "Threshold.h"
#include "Morphology.h"
#include "ConnectedComponents.h"
#include "RegionFeatures.h"
#include "ObjectDB.h"
#include "Classifier.h"
#include "Evaluator.h"
#include "Embedding.h"

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
static std::string getArg(int argc, char* argv[], const std::string& key)
{
    for (int i = 1; i < argc - 1; i++)
        if (std::string(argv[i]) == key) return argv[i + 1];
    return "";
}

static void overlayParams(cv::Mat& frame, const PipelineParams& p,
                           const AppState& state)
{
    static const char* morphNames[] = {"Open","Close","Erode","Dilate"};
    double     fontScale = 0.38;
    int        thickness = 1;
    cv::Scalar col       = {180, 255, 180};

    auto put = [&](const std::string& txt, int row) {
        int x = frame.cols - 220;
        int y = 15 + row * 16;
        cv::putText(frame, txt, {x+1, y+1},
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, {0,0,0}, thickness+1);
        cv::putText(frame, txt, {x, y},
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, col, thickness);
    };

    put("T:" + std::to_string(p.thresholdValue) +
        (p.useSatIntensity ? "[SI]" :
         p.useKMeans       ? "[ISO]" :
         p.useAdaptive     ? "[Adp]" : "[G]"), 0);
    put("Blur:" + std::to_string(p.blurKernelSize), 1);
    put(std::string("Morph:") + morphNames[p.morphMode] +
        " k=" + std::to_string(p.morphKernelSize) +
        " i=" + std::to_string(p.morphIterations), 2);
    put("MinA:" + std::to_string(p.minRegionArea), 3);
    put("Rgns:" + std::to_string(state.regions.size()), 4);
    put(std::string("Mode:") + (state.embeddingMode_ ? "CNN" : "HF"), 5);
    put("FPS:"  + std::to_string(static_cast<int>(state.fps)), 6);

    if (state.mode == AppState::Mode::Train) {
        std::string txt = "TRAIN: " +
            (state.currentTrainLabel.empty() ? "press N"
                                             : state.currentTrainLabel) +
            " [" + std::to_string(state.samplesThisLabel) + "]";
        cv::putText(frame, txt, {10, frame.rows - 15},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 200, 255}, 1);
    }
}

static void printParams(const PipelineParams& p)
{
    static const char* morphNames[] = {"Open","Close","Erode","Dilate"};
    std::cout << "\r"
              << "T:" << p.thresholdValue
              << " B:" << p.blurKernelSize
              << " Morph:" << morphNames[p.morphMode]
              << " MinArea:" << p.minRegionArea
              << "   " << std::flush;
}

static void captureTrainingSample(AppState& state, ObjectDB& db,
                                   Classifier& classifier)
{
    if (state.currentTrainLabel.empty()) {
        std::cout << "\n[Train] No label set. Press 'n' first.\n"; return;
    }
    if (state.regions.empty()) {
        std::cout << "\n[Train] No region detected.\n"; return;
    }
    const RegionInfo& reg = state.regions[0];
    if (reg.huMoments.empty()) {
        std::cout << "\n[Train] Features not computed yet.\n"; return;
    }
    DBEntry entry = ObjectDB::entryFromRegion(reg, state.currentTrainLabel);
    db.append(entry);
    state.samplesThisLabel++;
    classifier.refit(db);
    std::cout << "\n[Train] Captured sample " << state.samplesThisLabel
              << " for '" << state.currentTrainLabel << "'\n";
}

// -----------------------------------------------------------------------------
// Keyboard handler
// -----------------------------------------------------------------------------
static bool handleKeyboard(int key, PipelineParams& params, AppState& state,
                            ObjectDB& db, Classifier& classifier,
                            Evaluator& evaluator,
                            EmbeddingDB& embDB,
                            EmbeddingClassifier& embClassifier,
                            bool& showThresh, bool& showCleaned,
                            bool& showRegions, bool& showMatrix,
                            bool& showCrop)
{
    switch (key) {
        case 't': params.thresholdValue = std::max(0,   params.thresholdValue - 5); break;
        case 'T': params.thresholdValue = std::min(255, params.thresholdValue + 5); break;
        case 'b': params.blurKernelSize = std::max(1,  params.blurKernelSize - 2);  break;
        case 'B': params.blurKernelSize = std::min(21, params.blurKernelSize + 2);  break;
        case 'a':
            params.useAdaptive = !params.useAdaptive;
            if (params.useAdaptive) { params.useKMeans = false; params.useSatIntensity = false; }
            break;
        case 'k':
            params.useKMeans = !params.useKMeans;
            if (params.useKMeans) { params.useAdaptive = false; params.useSatIntensity = false; }
            break;
        case 's':
            params.useSatIntensity = !params.useSatIntensity;
            if (params.useSatIntensity) { params.useAdaptive = false; params.useKMeans = false; }
            break;
        case 'm': params.morphKernelSize = std::max(1,  params.morphKernelSize - 2); break;
        case 'M': params.morphKernelSize = std::min(21, params.morphKernelSize + 2); break;
        case 'i': params.morphIterations = std::max(1,  params.morphIterations - 1); break;
        case 'I': params.morphIterations = std::min(10, params.morphIterations + 1); break;
        case 'o': params.morphMode = (params.morphMode + 1) % 4; break;
        case 'r': params.minRegionArea = std::max(100,   params.minRegionArea - 100); break;
        case 'R': params.minRegionArea = std::min(50000, params.minRegionArea + 100); break;

        case 'n': {
            std::cout << "\nEnter label name: ";
            std::string lbl;
            std::cin >> lbl;
            if (!lbl.empty()) {
                state.currentTrainLabel = lbl;
                state.samplesThisLabel  = 0;
                state.mode              = AppState::Mode::Train;
                std::cout << "[Train] Label='" << lbl
                          << "'. Press 'c' for hand features, 'C' for embedding.\n";
            }
            break;
        }
        case 'c':
            captureTrainingSample(state, db, classifier);
            break;
        case 'C': {
            if (state.regions.empty()) {
                std::cout << "\n[EmbTrain] No region detected.\n"; break;
            }
            if (state.currentTrainLabel.empty()) {
                std::cout << "\n[EmbTrain] Press 'n' to set label first.\n"; break;
            }
            if (!embClassifier.isReady()) {
                std::cout << "\n[EmbTrain] Model not loaded.\n"; break;
            }
            std::vector<float> emb;
            if (embClassifier.computeEmbedding(
                    const_cast<cv::Mat&>(state.frameOriginal),
                    state.regions[0], emb)) {
                EmbeddingEntry e{state.currentTrainLabel, emb};
                embDB.append(e);
                std::cout << "\n[EmbTrain] Captured embedding for '"
                          << state.currentTrainLabel << "'\n";
            }
            break;
        }
        case 'x': params.confidenceThresh = std::max(0.1f, params.confidenceThresh - 0.1f);
            std::cout << "\nConf threshold: " << params.confidenceThresh << "\n"; break;
        case 'X': params.confidenceThresh = std::min(5.0f, params.confidenceThresh + 0.1f);
            std::cout << "\nConf threshold: " << params.confidenceThresh << "\n"; break;
        case 'd': params.distanceMetric = (params.distanceMetric + 1) % 2;
            std::cout << "\nMetric: " << (params.distanceMetric == 0 ?
                "Scaled Euclidean" : "Cosine") << "\n"; break;
        case 'j': params.kNeighbors = std::max(1, params.kNeighbors - 1);
            std::cout << "\nK: " << params.kNeighbors << "\n"; break;
        case 'J': params.kNeighbors = std::min(9, params.kNeighbors + 1);
            std::cout << "\nK: " << params.kNeighbors << "\n"; break;
        case '9':
            state.embeddingMode_ = !state.embeddingMode_;
            std::cout << "\n[Mode] "
                      << (state.embeddingMode_ ? "CNN Embedding" : "Hand Features")
                      << "\n";
            break;
        case 'e': {
            if (state.regions.empty()) {
                std::cout << "\n[Eval] No region detected.\n"; break;
            }
            std::cout << "\nEnter TRUE label: ";
            std::string trueLabel;
            std::cin >> trueLabel;
            const RegionInfo& reg = state.regions[0];
            evaluator.record(trueLabel, reg.label, reg.confidence);
            std::cout << "[Eval] true=" << trueLabel
                      << " predicted=" << reg.label
                      << " (" << evaluator.count() << " total)\n";
            break;
        }
        case 'p':
            evaluator.printMatrix();
            evaluator.saveMatrix();
            break;

        // Window toggles — flip bool, main loop handles show/hide
        case '1': showThresh  = !showThresh;  break;
        case '2': showCleaned = !showCleaned; break;
        case '3': showRegions = !showRegions; break;
        case '4': params.showAxes         = !params.showAxes;         break;
        case '5': params.showOrientedBBox = !params.showOrientedBBox; break;
        case '6': params.showFeatureText  = !params.showFeatureText;  break;
        case '7': showMatrix = !showMatrix; break;
        case '8': showCrop   = !showCrop;
            std::cout << "\n[Crop] " << (showCrop ? "ON" : "OFF") << "\n"; break;

        case 'q': case 27: state.running = false; return false;
        default: break;
    }
    printParams(params);
    return true;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    std::string modeStr  = getArg(argc, argv, "--mode");
    std::string inputStr = getArg(argc, argv, "--input");
    std::string camStr   = getArg(argc, argv, "--camera");
    std::string dbPath   = getArg(argc, argv, "--db");
    if (modeStr.empty()) modeStr = "live";
    if (dbPath.empty())  dbPath  = "data/db/objects.csv";

    AppState       state;
    PipelineParams params;

    if      (modeStr == "train") state.mode = AppState::Mode::Train;
    else if (modeStr == "eval")  state.mode = AppState::Mode::Eval;
    else if (modeStr == "embed") state.mode = AppState::Mode::Embed;
    else                         state.mode = AppState::Mode::Live;

    ObjectDB db(dbPath);
    auto counts = db.labelCounts();
    if (!counts.empty()) {
        std::cout << "[DB] Loaded labels: ";
        for (const auto& kv : counts)
            std::cout << kv.first << "(" << kv.second << ") ";
        std::cout << "\n";
    }

    Classifier          classifier(db, params);
    Evaluator           evaluator;
    EmbeddingDB         embDB("data/db/embeddings.csv");
    EmbeddingClassifier embClassifier;
    embClassifier.loadModel("data/models/resnet18-v2-7.onnx");

    cv::VideoCapture cap;
    bool imageMode = false;

    if (modeStr == "image" && !inputStr.empty()) {
        state.frameOriginal = cv::imread(inputStr);
        if (state.frameOriginal.empty()) {
            std::cerr << "Error: cannot open image: " << inputStr << "\n";
            return 1;
        }
        imageMode = true;
    } else {
        int camIdx = camStr.empty() ? 0 : std::stoi(camStr);
        if (!inputStr.empty()) cap.open(inputStr);
        else                   cap.open(camIdx);
        if (!cap.isOpened()) {
            std::cerr << "Error: cannot open capture source.\n";
            return 1;
        }
    }

    // Window names
    const std::string winMain    = "ObjectRecognition";
    const std::string winThresh  = "Threshold";
    const std::string winCleaned = "Cleaned";
    const std::string winRegions = "Regions";
    const std::string winMatrix  = "Confusion Matrix";

    // Toggle state
    bool showThresh  = true;
    bool showCleaned = true;
    bool showRegions = true;
    bool showMatrix  = false;
    bool showCrop    = false;

    // Crop windows — fixed names, max 5
    const int maxCropWins = 5;
    std::vector<std::string> cropWins;
    for (int i = 0; i < maxCropWins; i++)
        cropWins.push_back("Crop[" + std::to_string(i) + "]");
    bool prevShowCrop = false;

    // Previous toggle states to detect changes
    bool prevThresh  = true;
    bool prevCleaned = true;
    bool prevRegions = true;

    // Create all windows upfront
    cv::namedWindow(winMain,    cv::WINDOW_AUTOSIZE);
    cv::namedWindow(winThresh,  cv::WINDOW_AUTOSIZE);
    cv::namedWindow(winCleaned, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(winRegions, cv::WINDOW_AUTOSIZE);

    printParams(params);
    auto tPrev = std::chrono::steady_clock::now();

    // =========================================================================
    // MAIN LOOP
    // =========================================================================
    while (state.running) {

        if (!imageMode) {
            cap >> state.frameOriginal;
            if (state.frameOriginal.empty()) break;
        }

        // Pipeline
        applyThreshold(state.frameOriginal,     state.frameThresholded, params);
        applyMorphology(state.frameThresholded, state.frameCleaned,     params);

        cv::Mat labelMap;
        findRegions(state.frameCleaned, state, params, labelMap);
        computeAllFeatures(labelMap, state);

        if (!state.embeddingMode_)
            classifier.classifyAll(state, params);
        else if (embClassifier.isReady() && !embDB.empty())
            embClassifier.classifyAll(state.frameOriginal, state, embDB,
                                      params.confidenceThresh * 10000.f);

        // Main window — always shown
        state.frameDisplay = state.frameOriginal.clone();
        drawFeatures(state.frameDisplay, state, params);
        overlayParams(state.frameDisplay, params, state);
        cv::imshow(winMain, state.frameDisplay);

        // Secondary windows — destroy on toggle off, recreate on toggle on
        if (showThresh && !state.frameThresholded.empty()) {
            if (!prevThresh) cv::namedWindow(winThresh, cv::WINDOW_AUTOSIZE);
            cv::imshow(winThresh, state.frameThresholded);
        } else if (!showThresh && prevThresh)
            try { cv::destroyWindow(winThresh); } catch(...) {}

        if (showCleaned && !state.frameCleaned.empty()) {
            if (!prevCleaned) cv::namedWindow(winCleaned, cv::WINDOW_AUTOSIZE);
            cv::imshow(winCleaned, state.frameCleaned);
        } else if (!showCleaned && prevCleaned)
            try { cv::destroyWindow(winCleaned); } catch(...) {}

        if (showRegions && !state.frameRegions.empty()) {
            if (!prevRegions) cv::namedWindow(winRegions, cv::WINDOW_AUTOSIZE);
            cv::imshow(winRegions, state.frameRegions);
        } else if (!showRegions && prevRegions)
            try { cv::destroyWindow(winRegions); } catch(...) {}

        prevThresh  = showThresh;
        prevCleaned = showCleaned;
        prevRegions = showRegions;

        if (showMatrix) {
            cv::Mat matImg;
            evaluator.drawMatrix(matImg);
            cv::namedWindow(winMatrix, cv::WINDOW_AUTOSIZE);
            cv::imshow(winMatrix, matImg);
        } else {
            try { cv::destroyWindow(winMatrix); } catch (...) {}
        }

        // Crop windows — create/destroy only on toggle change
        if (showCrop && !prevShowCrop) {
            int n = std::min(static_cast<int>(state.croppedROIs.size()), maxCropWins);
            n = std::max(n, 1);
            for (int i = 0; i < n; i++)
                cv::namedWindow(cropWins[i], cv::WINDOW_AUTOSIZE);
        }
        if (!showCrop && prevShowCrop) {
            for (const auto& wn : cropWins)
                try { cv::destroyWindow(wn); } catch (...) {}
        }
        prevShowCrop = showCrop;

        if (showCrop) {
            int nCrops = std::min(static_cast<int>(state.croppedROIs.size()), maxCropWins);
            for (int i = 0; i < nCrops; i++) {
                if (state.croppedROIs[i].empty()) continue;
                cv::Mat disp = state.croppedROIs[i].clone();
                std::string lbl = i < static_cast<int>(state.regions.size())
                                  ? state.regions[i].label : "?";
                cv::putText(disp, lbl, {5,20},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);
                cv::imshow(cropWins[i], disp);
            }
        }

        // FPS
        auto  tNow = std::chrono::steady_clock::now();
        float dt   = std::chrono::duration<float>(tNow - tPrev).count();
        state.fps  = dt > 0.f ? 1.f / dt : 0.f;
        tPrev      = tNow;

        // Keyboard
        int key = cv::waitKey(30) & 0xFF;
        if (key != 255)
            handleKeyboard(key, params, state, db, classifier, evaluator,
                           embDB, embClassifier,
                           showThresh, showCleaned, showRegions, showMatrix,
                           showCrop);
    }

    for (const auto& wn : cropWins)
        try { cv::destroyWindow(wn); } catch (...) {}

    std::cout << "\nExiting. DB has " << db.size() << " entries.\n";
    cv::destroyAllWindows();
    return 0;
}
