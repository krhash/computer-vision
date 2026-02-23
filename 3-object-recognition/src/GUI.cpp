/**
 * @file    GUI.cpp
 * @brief   Dear ImGui GUI implementation -- single scrollable window.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#include "GUI.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <map>
#include <cstring>

// -----------------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------------
bool GUI::init(int width, int height, const std::string& title)
{
    if (!glfwInit()) {
        std::cerr << "[GUI] GLFW init failed.\n"; return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        std::cerr << "[GUI] Window creation failed.\n";
        glfwTerminate(); return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding  = 0.f;
    s.FrameRounding   = 4.f;
    s.GrabRounding    = 4.f;
    s.ItemSpacing     = ImVec2(8, 5);
    s.WindowPadding   = ImVec2(10, 10);
    s.IndentSpacing   = 14.f;

    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    initDone_ = true;
    std::cout << "[GUI] Initialised.\n";
    return true;
}

void GUI::pollEvents() { glfwPollEvents(); }
bool GUI::isOpen() const { return window_ && !glfwWindowShouldClose(window_); }

void GUI::shutdown()
{
    if (!initDone_) return;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
    initDone_ = false;
}

// -----------------------------------------------------------------------------
// Main render
// -----------------------------------------------------------------------------
void GUI::render(PipelineParams& params, AppState& state,
                  ObjectDB& db, Classifier& classifier,
                  Evaluator& evaluator, EmbeddingDB& embDB,
                  bool& showThresh, bool& showCleaned,
                  bool& showRegions, bool& showMatrix,
                  bool& showCrop)
{
    if (!initDone_) return;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    int winW, winH;
    glfwGetWindowSize(window_, &winW, &winH);

    ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({static_cast<float>(winW),
                              static_cast<float>(winH)}, ImGuiCond_Always);

    ImGui::Begin("Controls", nullptr,
                 ImGuiWindowFlags_NoMove         |
                 ImGuiWindowFlags_NoResize        |
                 ImGuiWindowFlags_NoCollapse      |
                 ImGuiWindowFlags_NoTitleBar      |
                 ImGuiWindowFlags_NoSavedSettings);

    // FPS + mode
    ImGui::TextColored({0.5f,0.5f,0.5f,1.f}, "FPS: %d", (int)state.fps);
    ImGui::SameLine();
    if (state.embeddingMode_)
        ImGui::TextColored({0.f,1.f,0.8f,1.f}, "  [CNN Mode]");
    else
        ImGui::TextColored({0.8f,0.8f,0.f,1.f}, "  [Shape Feature Mode]");

    ImGui::Spacing();

    renderPipelinePanel(params, state, showThresh, showCleaned, showRegions, showCrop);
    ImGui::Spacing();
    renderTrainingPanel(state, db, classifier, embDB);
    ImGui::Spacing();
    renderDBPanel(db, embDB, classifier);
    ImGui::Spacing();
    renderConfusionMatrix(evaluator);

    // -----------------------------------------------------------------
    // Auto-learn notification -- inline, no child window
    // -----------------------------------------------------------------
    if (state.autoLearnPending) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(ImVec4{1.f, 0.8f, 0.f, 1.f},
                           "! Unknown object detected -- Add to DB?");
        ImGui::SetNextItemWidth(-1);
        bool entered = ImGui::InputText("##autolbl", autoLearnBuf_,
                                        sizeof(autoLearnBuf_),
                                        ImGuiInputTextFlags_EnterReturnsTrue);
        float bw = (ImGui::GetContentRegionAvail().x - 8) * 0.5f;
        bool confirm = ImGui::Button("Add to DB##al", {bw, 26}) || entered;
        ImGui::SameLine();
        bool skip = ImGui::Button("Skip##al", {bw, 26});

        if (confirm && strlen(autoLearnBuf_) > 0) {
            std::string lbl(autoLearnBuf_);
            if (!state.autoLearnRegion.huMoments.empty()) {
                DBEntry e = ObjectDB::entryFromRegion(state.autoLearnRegion, lbl);
                db.append(e);
                classifier.refit(db);
            }
            state.currentTrainLabel = lbl;
            state.captureRequested  = true;
            state.autoLearnPending  = false;
            memset(autoLearnBuf_, 0, sizeof(autoLearnBuf_));
        }
        if (skip) {
            state.autoLearnPending = false;
            memset(autoLearnBuf_, 0, sizeof(autoLearnBuf_));
        }
        ImGui::Separator();
    }

    ImGui::End();

    // Draw
    ImGui::Render();
    int dw, dh;
    glfwGetFramebufferSize(window_, &dw, &dh);
    glViewport(0, 0, dw, dh);
    glClearColor(0.13f, 0.13f, 0.13f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window_);
}

// =============================================================================
// Pipeline Controls
// =============================================================================
void GUI::renderPipelinePanel(PipelineParams& params,
                               AppState& state,
                               bool& showThresh, bool& showCleaned,
                               bool& showRegions, bool& showCrop)
{
    if (!ImGui::CollapsingHeader("Pipeline", ImGuiTreeNodeFlags_DefaultOpen))
        return;

    ImGui::PushItemWidth(-1);

    ImGui::Text("Threshold"); ImGui::SameLine(110);
    ImGui::SliderInt("##thresh", &params.thresholdValue, 0, 255);

    ImGui::Text("Blur Kernel"); ImGui::SameLine(110);
    if (ImGui::SliderInt("##blur", &params.blurKernelSize, 1, 21))
        if (params.blurKernelSize % 2 == 0) params.blurKernelSize++;

    ImGui::Text("Thresh Mode");
    ImGui::SameLine();
    if (ImGui::RadioButton("Global##t",
        !params.useAdaptive && !params.useKMeans && !params.useSatIntensity))
    { params.useAdaptive = params.useKMeans = params.useSatIntensity = false; }
    ImGui::SameLine();
    if (ImGui::RadioButton("ISO##t", params.useKMeans))
    { params.useKMeans = true; params.useAdaptive = params.useSatIntensity = false; }
    ImGui::SameLine();
    if (ImGui::RadioButton("S+I##t", params.useSatIntensity))
    { params.useSatIntensity = true; params.useAdaptive = params.useKMeans = false; }

    ImGui::Separator();

    const char* morphModes[] = {"Open","Close","Erode","Dilate"};
    ImGui::Text("Morph Mode"); ImGui::SameLine(110);
    ImGui::Combo("##morphmode", &params.morphMode, morphModes, 4);

    ImGui::Text("Kernel"); ImGui::SameLine(110);
    if (ImGui::SliderInt("##morphk", &params.morphKernelSize, 1, 21))
        if (params.morphKernelSize % 2 == 0) params.morphKernelSize++;

    ImGui::Text("Iterations"); ImGui::SameLine(110);
    ImGui::SliderInt("##morphi", &params.morphIterations, 1, 10);

    ImGui::Separator();

    ImGui::Text("Min Area"); ImGui::SameLine(110);
    ImGui::SliderInt("##minarea", &params.minRegionArea, 100, 20000);

    ImGui::Text("Max Regions"); ImGui::SameLine(110);
    ImGui::SliderInt("##maxrgn", &params.maxRegions, 1, 5);

    ImGui::Separator();

    ImGui::Text("Confidence"); ImGui::SameLine(110);
    ImGui::SliderFloat("##conf", &params.confidenceThresh, 0.1f, 3.0f);

    ImGui::Text("K Neighbours"); ImGui::SameLine(110);
    ImGui::SliderInt("##knn", &params.kNeighbors, 1, 9);

    const char* metrics[] = {"Euclidean","Cosine"};
    ImGui::Text("Metric"); ImGui::SameLine(110);
    ImGui::Combo("##metric", &params.distanceMetric, metrics, 2);

    ImGui::PopItemWidth();
    ImGui::Separator();

    ImGui::Text("Show:");
    ImGui::SameLine();
    ImGui::Checkbox("Thresh##w",  &showThresh);
    ImGui::SameLine();
    ImGui::Checkbox("Clean##w",   &showCleaned);
    ImGui::SameLine();
    ImGui::Checkbox("Rgns##w",    &showRegions);
    ImGui::SameLine();
    ImGui::Checkbox("Crop##w",    &showCrop);

    ImGui::Text("     ");
    ImGui::SameLine();
    ImGui::Checkbox("Axes##w",     &params.showAxes);
    ImGui::SameLine();
    ImGui::Checkbox("BBox##w",     &params.showOrientedBBox);
    ImGui::SameLine();
    ImGui::Checkbox("Feat##w",     &params.showFeatureText);
    ImGui::SameLine();
    ImGui::Checkbox("Overlay##w",  &state.showOverlay);
    ImGui::SameLine();
    ImGui::Checkbox("EmbPlot##w",  &state.showPlot);
}

// =============================================================================
// Training Panel
// =============================================================================
void GUI::renderTrainingPanel(AppState& state,
                               ObjectDB& db, Classifier& classifier,
                               EmbeddingDB& /*embDB*/)
{
    if (!ImGui::CollapsingHeader("Training", ImGuiTreeNodeFlags_DefaultOpen))
        return;

    ImGui::Checkbox("CNN Mode##train", &state.embeddingMode_);
    ImGui::Separator();

    ImGui::Text("Label:"); ImGui::SameLine();
    ImGui::SetNextItemWidth(160);
    bool entered = ImGui::InputText("##trainlabel", labelBuf_, sizeof(labelBuf_),
                                    ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if (ImGui::Button("Set##train") || entered) {
        if (strlen(labelBuf_) > 0) {
            state.currentTrainLabel = labelBuf_;
            state.samplesThisLabel  = 0;
            state.mode              = AppState::Mode::Train;
        }
    }

    if (!state.currentTrainLabel.empty())
        ImGui::TextColored({0.f,1.f,1.f,1.f}, "Active: %s",
                           state.currentTrainLabel.c_str());
    else
        ImGui::TextDisabled("No label set");

    ImGui::Spacing();

    bool hasRegion  = !state.regions.empty();
    bool hasLabel   = !state.currentTrainLabel.empty();
    bool canCapture = hasRegion && hasLabel;

    if (!canCapture) ImGui::BeginDisabled();

    float btnW = (ImGui::GetContentRegionAvail().x - 8) * 0.5f;

    if (ImGui::Button("+ Shape Sample##train", {btnW, 28})) {
        if (canCapture && !state.regions[0].huMoments.empty()) {
            DBEntry e = ObjectDB::entryFromRegion(state.regions[0],
                                                   state.currentTrainLabel);
            db.append(e);
            state.samplesThisLabel++;
            classifier.refit(db);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("+ Embedding Sample##train", {btnW, 28}))
        if (canCapture) state.captureRequested = true;

    if (!canCapture) ImGui::EndDisabled();

    if (!hasLabel)
        ImGui::TextColored({1.f,0.6f,0.f,1.f}, "  Set a label first");
    else if (!hasRegion)
        ImGui::TextColored({1.f,0.6f,0.f,1.f}, "  No region in frame");
    else
        ImGui::TextColored({0.4f,1.f,0.4f,1.f}, "  Ready to capture");
}

// =============================================================================
// DB Panel
// =============================================================================
void GUI::renderDBPanel(ObjectDB& db, EmbeddingDB& embDB, Classifier& classifier)
{
    if (!ImGui::CollapsingHeader("Databases"))
        return;

    if (ImGui::TreeNode("Shape Feature DB")) {
        auto counts = db.labelCounts();
        if (counts.empty()) {
            ImGui::TextDisabled("Empty -- capture shape samples first.");
        } else {
            ImGui::Text("Total: %d entries", db.size());
            if (ImGui::BeginTable("hfTable", 3,
                                   ImGuiTableFlags_Borders |
                                   ImGuiTableFlags_RowBg   |
                                   ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Label",   ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Samples", ImGuiTableColumnFlags_WidthFixed, 55.f);
                ImGui::TableSetupColumn("##del",   ImGuiTableColumnFlags_WidthFixed, 30.f);
                ImGui::TableHeadersRow();
                std::string toDelete;
                for (const auto& kv : counts) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::Text("%s", kv.first.c_str());
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%d", kv.second);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::PushStyleColor(ImGuiCol_Button, {0.5f,0.1f,0.1f,1.f});
                    if (ImGui::SmallButton(("X##hf" + kv.first).c_str()))
                        toDelete = kv.first;
                    ImGui::PopStyleColor();
                }
                ImGui::EndTable();
                if (!toDelete.empty()) { db.deleteLabel(toDelete); classifier.refit(db); }
            }
        }
        ImGui::TreePop();
    }

    ImGui::Spacing();

    if (ImGui::TreeNode("Embedding DB (CNN)")) {
        if (embDB.empty()) {
            ImGui::TextDisabled("Empty -- capture embedding samples first.");
        } else {
            std::map<std::string,int> embCounts;
            for (const auto& e : embDB.entries()) embCounts[e.label]++;
            ImGui::Text("Total: %d entries", embDB.size());
            if (ImGui::BeginTable("embTable", 2,
                                   ImGuiTableFlags_Borders |
                                   ImGuiTableFlags_RowBg   |
                                   ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Label",   ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Samples", ImGuiTableColumnFlags_WidthFixed, 55.f);
                ImGui::TableHeadersRow();
                for (const auto& kv : embCounts) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::Text("%s", kv.first.c_str());
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%d", kv.second);
                }
                ImGui::EndTable();
            }
        }
        ImGui::TreePop();
    }
}

// =============================================================================
// Confusion Matrix
// =============================================================================
void GUI::renderConfusionMatrix(Evaluator& evaluator)
{
    if (!ImGui::CollapsingHeader("Confusion Matrix"))
        return;

    ImGui::Text("Samples: %d", evaluator.count());
    ImGui::SameLine();
    if (ImGui::Button("Save##cm"))  evaluator.saveMatrix();
    ImGui::SameLine();
    if (ImGui::Button("Clear##cm")) evaluator.clear();

    if (evaluator.count() == 0) {
        ImGui::TextDisabled("Press 'e' in main window to record.");
        return;
    }

    ImGui::TextColored({0.2f,1.f,0.5f,1.f},
                       "Accuracy: %.1f%%", evaluator.accuracy() * 100.f);

    std::vector<std::string>      labels;
    std::vector<std::vector<int>> matrix;
    evaluator.buildMatrix(labels, matrix);
    int n = static_cast<int>(labels.size());
    if (n == 0) return;

    if (ImGui::BeginTable("cmTable", n + 1,
                           ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("T\\P", ImGuiTableColumnFlags_WidthFixed, 70.f);
        for (const auto& l : labels)
            ImGui::TableSetupColumn(l.substr(0,5).c_str(),
                                    ImGuiTableColumnFlags_WidthFixed, 45.f);
        ImGui::TableHeadersRow();

        for (int i = 0; i < n; i++) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", labels[i].substr(0,7).c_str());
            int rowTotal = 0;
            for (int v : matrix[i]) rowTotal += v;
            for (int j = 0; j < n; j++) {
                ImGui::TableSetColumnIndex(j + 1);
                int val = matrix[i][j];
                if (i == j && val > 0) {
                    float t = rowTotal > 0 ? (float)val/rowTotal : 0.f;
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg,
                        ImGui::ColorConvertFloat4ToU32(
                            {0.f, 0.25f+0.5f*t, 0.f, 1.f}));
                } else if (val > 0) {
                    float t = rowTotal > 0 ? (float)val/rowTotal : 0.f;
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg,
                        ImGui::ColorConvertFloat4ToU32(
                            {0.25f+0.5f*t, 0.f, 0.f, 1.f}));
                }
                ImGui::Text("%d", val);
            }
        }
        ImGui::EndTable();
    }
}
