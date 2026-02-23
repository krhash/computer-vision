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
 *            n       enter train mode -- prompts for label
 *            c       capture shape feature sample
 *            C       capture CNN embedding sample
 *            x/X     confidence threshold -/+0.1
 *            d       toggle Euclidean / Cosine metric
 *            j/J     K neighbours -/+1
 *            e       record eval sample
 *            p       print + save confusion matrix
 *            1-3     toggle Threshold/Cleaned/Regions windows
 *            4-6     toggle Axes/BBox/FeatureText overlays
 *            7       toggle confusion matrix window
 *            8       toggle embedding crop windows
 *            9       toggle CNN embedding / shape feature mode
 *            0       toggle embedding scatter plot
 *            q/ESC   quit
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <cmath>

#include "AppState.h"
#include "Threshold.h"
#include "Morphology.h"
#include "ConnectedComponents.h"
#include "RegionFeatures.h"
#include "ObjectDB.h"
#include "Classifier.h"
#include "Evaluator.h"
#include "Embedding.h"
#include "EmbeddingPlot.h"
#include "GUI.h"

// Unknown auto-prompt threshold -- frames before triggering popup
static const int kUnknownFrameThresh = 60;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
/**
 * @brief Retrieve a command-line argument value by key.
 *
 * @param argc  Argument count from main.
 * @param argv  Argument vector from main.
 * @param key   The flag to search for (e.g. "--mode").
 * @return      The token immediately following key, or an empty string if
 *              key is not found or has no following token.
 */
static std::string getArg(int argc, char* argv[], const std::string& key)
{
    for (int i = 1; i < argc - 1; i++)
        if (std::string(argv[i]) == key) return argv[i + 1];
    return "";
}

/**
 * @brief Draw a compact parameter overlay in the top-right corner of the frame.
 *
 * @param frame  BGR display frame to draw on (modified in place).
 * @param p      Current pipeline parameters to display.
 * @param state  App state (mode, region count, FPS, embedding flag).
 */
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
    put(std::string("Mode:") +
        (state.embeddingMode_ ? "CNN" : "Shape Feature"), 5);
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

/**
 * @brief Print current pipeline parameter values to stdout on a single line.
 * @param p  Current pipeline parameters.
 */
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

/**
 * @brief Capture a shape-feature training sample for the first detected region.
 *
 *        Builds a DBEntry from the top-ranked region's features, appends it
 *        to the object database, increments the per-label sample counter, and
 *        refits the classifier.
 *
 * @param state       App state — reads current label and top-ranked region.
 * @param db          Object database — entry is appended here.
 * @param classifier  Classifier — refitted after the append.
 */
static void captureTrainingSample(AppState& state, ObjectDB& db,
                                   Classifier& classifier)
{
    if (state.currentTrainLabel.empty()) {
        std::cout << "\n[Train] No label set.\n"; return;
    }
    if (state.regions.empty()) {
        std::cout << "\n[Train] No region detected.\n"; return;
    }
    const RegionInfo& reg = state.regions[0];
    if (reg.huMoments.empty()) {
        std::cout << "\n[Train] Features not computed.\n"; return;
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
/**
 * @brief Dispatch a single keypress to the appropriate pipeline action.
 *
 * @param key            ASCII key code from cv::waitKey.
 * @param params         Pipeline parameters — most keys modify these.
 * @param state          App state — mode, label, and run flag modified here.
 * @param db             Object database — modified on capture/delete.
 * @param classifier     Classifier — refitted when DB changes.
 * @param evaluator      Evaluator — records evaluation samples.
 * @param embDB          Embedding database — capture appended here.
 * @param embClassifier  CNN embedding classifier — used for embedding capture.
 * @param showThresh     Toggle for the Threshold debug window.
 * @param showCleaned    Toggle for the Cleaned debug window.
 * @param showRegions    Toggle for the Regions debug window.
 * @param showMatrix     Toggle for the Confusion Matrix window.
 * @param showCrop       Toggle for the per-region crop windows.
 * @return               False if the key was quit (q/ESC), true otherwise.
 */
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
                std::cout << "[Train] Label='" << lbl << "'.\n";
            }
            break;
        }
        case 'c':
            captureTrainingSample(state, db, classifier);
            break;
        case 'C': {
            if (state.regions.empty()) {
                std::cout << "\n[EmbTrain] No region.\n"; break;
            }
            if (state.currentTrainLabel.empty()) {
                std::cout << "\n[EmbTrain] Set label first.\n"; break;
            }
            if (!embClassifier.isReady()) {
                std::cout << "\n[EmbTrain] Model not loaded.\n"; break;
            }
            std::vector<float> emb;
            if (embClassifier.computeEmbedding(
                    const_cast<cv::Mat&>(state.frameOriginal),
                    state.regions[0], emb)) {
                embDB.append({state.currentTrainLabel, emb});
                std::cout << "\n[EmbTrain] Captured for '"
                          << state.currentTrainLabel << "'\n";
            }
            break;
        }
        case 'x': params.confidenceThresh = std::max(0.1f, params.confidenceThresh - 0.1f);
            std::cout << "\nConf: " << params.confidenceThresh << "\n"; break;
        case 'X': params.confidenceThresh = std::min(5.0f, params.confidenceThresh + 0.1f);
            std::cout << "\nConf: " << params.confidenceThresh << "\n"; break;
        case 'd': params.distanceMetric = (params.distanceMetric + 1) % 2;
            std::cout << "\nMetric: " << (params.distanceMetric == 0 ?
                "Euclidean" : "Cosine") << "\n"; break;
        case 'j': params.kNeighbors = std::max(1, params.kNeighbors - 1);
            std::cout << "\nK: " << params.kNeighbors << "\n"; break;
        case 'J': params.kNeighbors = std::min(9, params.kNeighbors + 1);
            std::cout << "\nK: " << params.kNeighbors << "\n"; break;
        case '9':
            state.embeddingMode_ = !state.embeddingMode_;
            std::cout << "\n[Mode] "
                      << (state.embeddingMode_ ? "CNN" : "Shape Feature") << "\n";
            break;
        case 'e': {
            if (state.regions.empty()) {
                std::cout << "\n[Eval] No region.\n"; break;
            }
            std::cout << "\nEnter TRUE label: ";
            std::string lbl;
            std::cin >> lbl;
            const RegionInfo& reg = state.regions[0];
            evaluator.record(lbl, reg.label, reg.confidence);
            std::cout << "[Eval] true=" << lbl
                      << " predicted=" << reg.label
                      << " (" << evaluator.count() << " total)\n";
            break;
        }
        case 'p':
            evaluator.printMatrix();
            evaluator.saveMatrix();
            break;
        case '1': showThresh  = !showThresh;  break;
        case '2': showCleaned = !showCleaned; break;
        case '3': showRegions = !showRegions; break;
        case '4': params.showAxes         = !params.showAxes;         break;
        case '5': params.showOrientedBBox = !params.showOrientedBBox; break;
        case '6': params.showFeatureText  = !params.showFeatureText;  break;
        case '7': showMatrix  = !showMatrix;  break;
        case '8': showCrop    = !showCrop;    break;
        case '0': state.showPlot = !state.showPlot;
            std::cout << "\n[Plot] " << (state.showPlot ? "ON" : "OFF") << "\n"; break;
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

    // Load DBs
    ObjectDB db(dbPath);
    auto counts = db.labelCounts();
    if (!counts.empty()) {
        std::cout << "[DB] Labels: ";
        for (const auto& kv : counts)
            std::cout << kv.first << "(" << kv.second << ") ";
        std::cout << "\n";
    }

    Classifier          classifier(db, params);
    Evaluator           evaluator;
    EmbeddingDB         embDB("data/db/embeddings.csv");
    EmbeddingClassifier embClassifier;
    embClassifier.loadModel("data/models/resnet18-v2-7.onnx");

    // Capture source
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

    // GUI
    GUI  gui;
    bool guiEnabled = gui.init(480, 720, "ObjectRecognition - Controls");

    // Window names
    const std::string winMain    = "ObjectRecognition";
    const std::string winThresh  = "Threshold";
    const std::string winCleaned = "Cleaned";
    const std::string winRegions = "Regions";
    const std::string winMatrix  = "Confusion Matrix";
    const std::string winPlot    = "Embedding Space (PCA 2D)";

    bool showThresh  = true;
    bool showCleaned = true;
    bool showRegions = true;
    bool showMatrix  = false;
    bool showCrop    = false;

    // Crop windows
    const int maxCropWins = 5;
    std::vector<std::string> cropWins;
    for (int i = 0; i < maxCropWins; i++)
        cropWins.push_back("Crop[" + std::to_string(i) + "]");
    bool prevShowCrop = false;

    // Window toggle previous states
    bool prevThresh  = true;
    bool prevCleaned = true;
    bool prevShowRegions = true;

    cv::namedWindow(winMain, cv::WINDOW_AUTOSIZE);

    printParams(params);
    auto tPrev = std::chrono::steady_clock::now();

    // Region persistence for unknownFrames
    std::vector<RegionInfo> lastRegions;

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

        // Persist unknownFrames across frames by centroid matching
        for (auto& reg : state.regions) {
            for (const auto& last : lastRegions) {
                float dx = reg.centroid.x - last.centroid.x;
                float dy = reg.centroid.y - last.centroid.y;
                if (std::sqrt(dx*dx + dy*dy) < 150.f) { // wider tolerance
                    reg.unknownFrames = last.unknownFrames;
                    break;
                }
            }
        }

        // Classify
        if (!state.embeddingMode_)
            classifier.classifyAll(state, params);
        else if (embClassifier.isReady() && !embDB.empty())
            embClassifier.classifyAll(state.frameOriginal, state, embDB,
                                      params.confidenceThresh * 10000.f);

        // Extension C: auto-prompt for unknown objects
        if (!state.autoLearnPending) {
            for (auto& reg : state.regions) {
                if (reg.label == "unknown") {
                    reg.unknownFrames++;
                    std::cout << "\r[AutoLearn] " << reg.unknownFrames
                              << "/" << kUnknownFrameThresh << "   " << std::flush;
                    if (reg.unknownFrames >= kUnknownFrameThresh &&
                        !reg.huMoments.empty()) {  // only trigger on valid region
                        state.autoLearnPending = true;
                        state.autoLearnRegion  = reg; // copy not pointer
                        reg.unknownFrames      = 0;
                        std::cout << "\n[AutoLearn] Pending!\n";
                        break;
                    }
                } else {
                    reg.unknownFrames = 0;
                }
            }
        }

        // Save regions for next frame
        lastRegions = state.regions;

        // Handle embedding capture request from GUI
        if (state.captureRequested) {
            state.captureRequested = false;
            if (!state.regions.empty() && !state.currentTrainLabel.empty()
                && embClassifier.isReady()) {
                std::vector<float> emb;
                if (embClassifier.computeEmbedding(
                        const_cast<cv::Mat&>(state.frameOriginal),
                        state.regions[0], emb)) {
                    embDB.append({state.currentTrainLabel, emb});
                    std::cout << "\n[EmbTrain] Captured for '"
                              << state.currentTrainLabel << "'\n";
                }
            }
        }

        // Display
        state.frameDisplay = state.frameOriginal.clone();
        drawFeatures(state.frameDisplay, state, params);

        // Auto-learn progress bar
        for (const auto& reg : state.regions) {
            if (reg.label == "unknown" && reg.unknownFrames > 0) {
                float pct  = static_cast<float>(reg.unknownFrames) / kUnknownFrameThresh;
                int   barW = static_cast<int>(pct * 200);
                cv::rectangle(state.frameDisplay,
                              {10, state.frameDisplay.rows-35},
                              {210, state.frameDisplay.rows-25},
                              {60,60,60}, -1);
                cv::rectangle(state.frameDisplay,
                              {10, state.frameDisplay.rows-35},
                              {10+barW, state.frameDisplay.rows-25},
                              {0,140,255}, -1);
                cv::putText(state.frameDisplay, "Unknown...",
                            {10, state.frameDisplay.rows-38},
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, {0,140,255}, 1);
                break;
            }
        }

        if (state.showOverlay)
            overlayParams(state.frameDisplay, params, state);
        cv::imshow(winMain, state.frameDisplay);

        // Secondary windows
        if (showThresh && !state.frameThresholded.empty()) {
            if (!prevThresh) cv::namedWindow(winThresh, cv::WINDOW_AUTOSIZE);
            cv::imshow(winThresh, state.frameThresholded);
        } else if (!showThresh && prevThresh)
            try { cv::destroyWindow(winThresh); } catch (...) {}

        if (showCleaned && !state.frameCleaned.empty()) {
            if (!prevCleaned) cv::namedWindow(winCleaned, cv::WINDOW_AUTOSIZE);
            cv::imshow(winCleaned, state.frameCleaned);
        } else if (!showCleaned && prevCleaned)
            try { cv::destroyWindow(winCleaned); } catch (...) {}

        if (showRegions && !state.frameRegions.empty()) {
            if (!prevShowRegions) cv::namedWindow(winRegions, cv::WINDOW_AUTOSIZE);
            cv::imshow(winRegions, state.frameRegions);
        } else if (!showRegions && prevShowRegions)
            try { cv::destroyWindow(winRegions); } catch (...) {}

        prevThresh      = showThresh;
        prevCleaned     = showCleaned;
        prevShowRegions = showRegions;

        // Confusion matrix window
        if (showMatrix) {
            cv::Mat matImg;
            evaluator.drawMatrix(matImg);
            if (cv::getWindowProperty(winMatrix, cv::WND_PROP_VISIBLE) < 1)
                cv::namedWindow(winMatrix, cv::WINDOW_AUTOSIZE);
            cv::imshow(winMatrix, matImg);
        } else {
            try { cv::destroyWindow(winMatrix); } catch (...) {}
        }

        // Crop windows
        if (showCrop && !prevShowCrop) {
            int n = std::min(static_cast<int>(state.croppedROIs.size()), maxCropWins);
            n = std::max(n, 1);
            for (int i = 0; i < n; i++)
                cv::namedWindow(cropWins[i], cv::WINDOW_AUTOSIZE);
        }
        if (!showCrop && prevShowCrop)
            for (const auto& wn : cropWins)
                try { cv::destroyWindow(wn); } catch (...) {}
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

        // Extension A: embedding scatter plot
        if (state.showPlot) {
            cv::Mat plotImg;
            renderEmbeddingPlot(embDB, 500, plotImg);
            if (cv::getWindowProperty(winPlot, cv::WND_PROP_VISIBLE) < 1)
                cv::namedWindow(winPlot, cv::WINDOW_AUTOSIZE);
            cv::imshow(winPlot, plotImg);
        } else {
            try { cv::destroyWindow(winPlot); } catch (...) {}
        }

        // GUI
        if (guiEnabled) {
            gui.pollEvents();
            gui.render(params, state, db, classifier, evaluator, embDB,
                       showThresh, showCleaned, showRegions, showMatrix,
                       showCrop);
            if (!gui.isOpen()) state.running = false;
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

    if (guiEnabled) gui.shutdown();
    std::cout << "\nExiting. DB has " << db.size() << " entries.\n";
    cv::destroyAllWindows();
    return 0;
}
