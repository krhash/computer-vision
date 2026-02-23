/**
 * @file    GUI.h
 * @brief   Dear ImGui GUI manager for the Object Recognition system.
 *
 *          Renders four dockable panels in a GLFW/OpenGL3 window:
 *            1. Pipeline Controls  — threshold, blur, morph sliders
 *            2. Training Panel     — label input, capture buttons
 *            3. DB Manager         — label list, delete, sample counts
 *            4. Confusion Matrix   — color-coded evaluation table
 *
 *          All panels read/write PipelineParams and AppState directly.
 *          OpenCV windows remain unchanged alongside the ImGui window.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#pragma once

#include <string>
#include "AppState.h"
#include "ObjectDB.h"
#include "Classifier.h"
#include "Evaluator.h"
#include "Embedding.h"

// Forward declarations
struct GLFWwindow;

// =============================================================================
// GUI — manages GLFW window + ImGui lifecycle and all panels
// =============================================================================
class GUI {
public:
    /**
     * @brief Initialise GLFW window and ImGui context.
     * @param width   Window width in pixels.
     * @param height  Window height in pixels.
     * @param title   Window title bar text.
     * @return        True on success.
     */
    bool init(int width = 480, int height = 700,
              const std::string& title = "ObjectRecognition — Controls");

    /**
     * @brief Render one frame of ImGui panels.
     *        Call once per main loop iteration after the pipeline runs.
     *
     * @param params      Pipeline parameters — sliders write here directly.
     * @param state       App state — training status and regions are read here.
     * @param db          Shape feature object DB — managed by the DB panel.
     * @param classifier  K-NN classifier — refitted when the DB changes.
     * @param evaluator   Evaluator — confusion matrix data read for display.
     * @param embDB       CNN embedding DB — sample counts shown in DB panel.
     * @param showThresh  Toggle for the Threshold debug window.
     * @param showCleaned Toggle for the post-morphology Cleaned debug window.
     * @param showRegions Toggle for the color-coded Regions debug window.
     * @param showMatrix  Toggle for the Confusion Matrix OpenCV window.
     * @param showCrop    Toggle for the per-region aligned ROI crop windows.
     */
    void render(PipelineParams& params, AppState& state,
                ObjectDB& db, Classifier& classifier,
                Evaluator& evaluator, EmbeddingDB& embDB,
                bool& showThresh, bool& showCleaned,
                bool& showRegions, bool& showMatrix,
                bool& showCrop);

    /** Poll GLFW events. Call every frame. */
    void pollEvents();

    /** Return true while GLFW window is open. */
    bool isOpen() const;

    /** Shutdown ImGui and GLFW. */
    void shutdown();

private:
    GLFWwindow* window_    = nullptr;
    bool        initDone_  = false;
    char        labelBuf_[128]     = {};
    char        autoLearnBuf_[128] = {};

    /** Render threshold, morphology, region, and classifier controls. */
    void renderPipelinePanel(PipelineParams& params,
                              AppState& state,
                              bool& showThresh, bool& showCleaned,
                              bool& showRegions, bool& showCrop);

    /** Render label input and sample capture buttons. */
    void renderTrainingPanel(AppState& state,
                              ObjectDB& db, Classifier& classifier,
                              EmbeddingDB& embDB);

    /** Render the collapsible DB manager showing entry counts and delete buttons. */
    void renderDBPanel(ObjectDB& db, EmbeddingDB& embDB,
                       Classifier& classifier);

    /** Render the collapsible inline confusion matrix table. */
    void renderConfusionMatrix(Evaluator& evaluator);
};
