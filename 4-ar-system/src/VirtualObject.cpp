////////////////////////////////////////////////////////////////////////////////
// VirtualObject.cpp - Virtual Object Class Implementation
// Author:      Krushna Sanjay Sharma
// Description: Implements the rocket 3D virtual object. Geometry is defined
//              as line segments in world space. cv::projectPoints() is used
//              each frame to project the 3D endpoints onto the 2D image plane,
//              then cv::line() draws each segment in its assigned color.
//
// Task coverage:
//   Task 6 - Virtual 3D rocket floating above the chessboard target
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#include "VirtualObject.h"
#include <iostream>

// ── Draw colors (BGR) ─────────────────────────────────────────────────────────
static const cv::Scalar COLOR_BODY    (255, 100,   0);   // blue   — body box
static const cv::Scalar COLOR_NOSE    (255, 255,   0);   // cyan   — nose cone
static const cv::Scalar COLOR_FINS    (  0, 180, 255);   // amber  — fins
static const cv::Scalar COLOR_EXHAUST (  0,  50, 255);   // red    — exhaust
static const cv::Scalar COLOR_WINDOW  (  0, 220, 100);   // green  — window

// ----------------------------------------------------------------------------
// addLine() - internal helper
// ----------------------------------------------------------------------------
void VirtualObject::addLine(const cv::Vec3f& start,
                             const cv::Vec3f& end,
                             const cv::Scalar& color,
                             int thickness)
{
    m_lines.push_back({ start, end, color, thickness });
}

// ----------------------------------------------------------------------------
// buildRocket()
// Calls each sub-builder in order. Call once after construction.
// ----------------------------------------------------------------------------
void VirtualObject::buildRocket()
{
    m_lines.clear();
    buildBody();
    buildNose();
    buildFins();
    buildExhaust();
    buildWindow();

    std::cout << "[VirtualObject] Rocket built with "
              << m_lines.size() << " line segments.\n";
}

// ----------------------------------------------------------------------------
// buildBody()
// A rectangular box: 4 vertical edges + top face + bottom face.
// 8 corners: (X_LEFT/RIGHT, Y_FRONT/BACK, Z_BASE/Z_TOP)
//
//  Bottom face (Z=0, on board surface):
//    BL_F=(3,1.5,0), BR_F=(5,1.5,0), BL_B=(3,3.5,0), BR_B=(5,3.5,0)
//  Top face (Z=-3, above board):
//    TL_F=(3,1.5,-3), TR_F=(5,1.5,-3), TL_B=(3,3.5,-3), TR_B=(5,3.5,-3)
// ----------------------------------------------------------------------------
void VirtualObject::buildBody()
{
    // Convenience: define the 8 corners
    cv::Vec3f BL_F(X_LEFT,  Y_FRONT, Z_BASE);   // bottom-left-front
    cv::Vec3f BR_F(X_RIGHT, Y_FRONT, Z_BASE);   // bottom-right-front
    cv::Vec3f BL_B(X_LEFT,  Y_BACK,  Z_BASE);   // bottom-left-back
    cv::Vec3f BR_B(X_RIGHT, Y_BACK,  Z_BASE);   // bottom-right-back

    cv::Vec3f TL_F(X_LEFT,  Y_FRONT, Z_TOP);    // top-left-front
    cv::Vec3f TR_F(X_RIGHT, Y_FRONT, Z_TOP);    // top-right-front
    cv::Vec3f TL_B(X_LEFT,  Y_BACK,  Z_TOP);    // top-left-back
    cv::Vec3f TR_B(X_RIGHT, Y_BACK,  Z_TOP);    // top-right-back

    // 4 vertical edges
    addLine(BL_F, TL_F, COLOR_BODY, 2);
    addLine(BR_F, TR_F, COLOR_BODY, 2);
    addLine(BL_B, TL_B, COLOR_BODY, 2);
    addLine(BR_B, TR_B, COLOR_BODY, 2);

    // Bottom face (4 edges)
    addLine(BL_F, BR_F, COLOR_BODY, 2);
    addLine(BR_F, BR_B, COLOR_BODY, 2);
    addLine(BR_B, BL_B, COLOR_BODY, 2);
    addLine(BL_B, BL_F, COLOR_BODY, 2);

    // Top face (4 edges)
    addLine(TL_F, TR_F, COLOR_BODY, 2);
    addLine(TR_F, TR_B, COLOR_BODY, 2);
    addLine(TR_B, TL_B, COLOR_BODY, 2);
    addLine(TL_B, TL_F, COLOR_BODY, 2);
}

// ----------------------------------------------------------------------------
// buildNose()
// A 4-sided pyramid from the body's top face to a single apex point.
// Apex: (X_CENTER, Y_CENTER, Z_APEX) = (4, 2.5, -5)
// Base: the 4 top-face corners of the body at Z_TOP = -3
// ----------------------------------------------------------------------------
void VirtualObject::buildNose()
{
    cv::Vec3f apex(X_CENTER, Y_CENTER, Z_APEX);

    cv::Vec3f TL_F(X_LEFT,  Y_FRONT, Z_TOP);
    cv::Vec3f TR_F(X_RIGHT, Y_FRONT, Z_TOP);
    cv::Vec3f TL_B(X_LEFT,  Y_BACK,  Z_TOP);
    cv::Vec3f TR_B(X_RIGHT, Y_BACK,  Z_TOP);

    // 4 edges from each top corner up to the apex
    addLine(TL_F, apex, COLOR_NOSE, 2);
    addLine(TR_F, apex, COLOR_NOSE, 2);
    addLine(TL_B, apex, COLOR_NOSE, 2);
    addLine(TR_B, apex, COLOR_NOSE, 2);
}

// ----------------------------------------------------------------------------
// buildFins()
// Four triangular fins — one at each bottom corner of the body.
// Each fin is a triangle with:
//   - Base edge on the body corner vertical edge
//   - Tip pointing outward from the rocket center
//
// Fin layout (viewed from below):
//   Front-left  fin: tip at (X_LEFT-0.5,  Y_FRONT-0.5, Z_BASE)
//   Front-right fin: tip at (X_RIGHT+0.5, Y_FRONT-0.5, Z_BASE)
//   Back-left   fin: tip at (X_LEFT-0.5,  Y_BACK+0.5,  Z_BASE)
//   Back-right  fin: tip at (X_RIGHT+0.5, Y_BACK+0.5,  Z_BASE)
// Each fin connects: corner_bottom → fin_tip → corner_at_Z_FIN → corner_bottom
// ----------------------------------------------------------------------------
void VirtualObject::buildFins()
{
    // Helper lambda — draws one triangular fin
    auto addFin = [&](cv::Vec3f cornerBase,    // body corner at Z_BASE
                      cv::Vec3f cornerMid,     // body corner at Z_FIN
                      cv::Vec3f tip)           // outer fin tip
    {
        addLine(cornerBase, tip,       COLOR_FINS, 2);
        addLine(tip,        cornerMid, COLOR_FINS, 2);
        addLine(cornerMid,  cornerBase, COLOR_FINS, 2);
    };

    // Front-left fin  (Y_FRONT is -1.5, tip goes more negative = further front)
    addFin(cv::Vec3f(X_LEFT,  Y_FRONT, Z_BASE),
           cv::Vec3f(X_LEFT,  Y_FRONT, Z_FIN),
           cv::Vec3f(X_LEFT  - 0.8f, Y_FRONT + 0.8f, Z_BASE));

    // Front-right fin
    addFin(cv::Vec3f(X_RIGHT, Y_FRONT, Z_BASE),
           cv::Vec3f(X_RIGHT, Y_FRONT, Z_FIN),
           cv::Vec3f(X_RIGHT + 0.8f, Y_FRONT + 0.8f, Z_BASE));

    // Back-left fin  (Y_BACK is -3.5, tip goes more negative = further back)
    addFin(cv::Vec3f(X_LEFT,  Y_BACK,  Z_BASE),
           cv::Vec3f(X_LEFT,  Y_BACK,  Z_FIN),
           cv::Vec3f(X_LEFT  - 0.8f, Y_BACK - 0.8f, Z_BASE));

    // Back-right fin
    addFin(cv::Vec3f(X_RIGHT, Y_BACK,  Z_BASE),
           cv::Vec3f(X_RIGHT, Y_BACK,  Z_FIN),
           cv::Vec3f(X_RIGHT + 0.8f, Y_BACK - 0.8f, Z_BASE));
}

// ----------------------------------------------------------------------------
// buildExhaust()
// A downward-pointing triangle centered below the rocket base.
// Base edge spans the bottom face center, tip points below the board (Z > 0).
// This gives the rocket a visible exhaust nozzle effect.
// ----------------------------------------------------------------------------
void VirtualObject::buildExhaust()
{
    cv::Vec3f left (X_LEFT  + 0.3f, Y_CENTER, Z_BASE);
    cv::Vec3f right(X_RIGHT - 0.3f, Y_CENTER, Z_BASE);
    cv::Vec3f tip  (X_CENTER,        Y_CENTER, Z_EXHST);  // below board

    addLine(left,  right, COLOR_EXHAUST, 2);
    addLine(left,  tip,   COLOR_EXHAUST, 2);
    addLine(right, tip,   COLOR_EXHAUST, 2);
}

// ----------------------------------------------------------------------------
// buildWindow()
// A small square porthole on the front face of the body.
// Positioned at mid-height on the front face (Y = Y_FRONT, Z midpoint).
// ----------------------------------------------------------------------------
void VirtualObject::buildWindow()
{
    // Window centered at (X_CENTER, Y_FRONT, midZ)
    // Half-size = 0.3 squares in X and Z
    float midZ = (Z_BASE + Z_TOP) / 2.0f;       // -1.5
    float hw   = 0.35f;                           // half-width in X
    float hh   = 0.35f;                           // half-height in Z

    cv::Vec3f BL(X_CENTER - hw, Y_FRONT, midZ - hh);
    cv::Vec3f BR(X_CENTER + hw, Y_FRONT, midZ - hh);
    cv::Vec3f TL(X_CENTER - hw, Y_FRONT, midZ + hh);
    cv::Vec3f TR(X_CENTER + hw, Y_FRONT, midZ + hh);

    addLine(BL, BR, COLOR_WINDOW, 2);
    addLine(BR, TR, COLOR_WINDOW, 2);
    addLine(TR, TL, COLOR_WINDOW, 2);
    addLine(TL, BL, COLOR_WINDOW, 2);

    // Cross lines inside the window (porthole style)
    addLine(BL, TR, COLOR_WINDOW, 1);
    addLine(BR, TL, COLOR_WINDOW, 1);
}

// ----------------------------------------------------------------------------
// draw() - Core of Task 6
//
// Strategy: collect ALL unique 3D endpoints from every line segment into one
// vector, call cv::projectPoints() ONCE, then draw each line using the
// pre-projected 2D indices. This is more efficient than calling projectPoints
// per line and is the correct pattern for multi-line virtual objects.
//
// cv::projectPoints() transformation chain:
//   3D world point → camera space (rvec/tvec) → normalize → distort
//   → pixel coords (cameraMatrix)
// ----------------------------------------------------------------------------
void VirtualObject::draw(cv::Mat&        frame,
                          const cv::Mat& rvec,
                          const cv::Mat& tvec,
                          const cv::Mat& cameraMatrix,
                          const cv::Mat& distCoeffs) const
{
    if (m_lines.empty()) return;

    // ── Step 1: Collect all 3D endpoints into one flat vector ─────────────────
    // Each line contributes 2 points → index i*2 = start, i*2+1 = end
    std::vector<cv::Vec3f> points3D;
    points3D.reserve(m_lines.size() * 2);

    for (const auto& seg : m_lines)
    {
        points3D.push_back(seg.start);
        points3D.push_back(seg.end);
    }

    // ── Step 2: Project ALL points in one cv::projectPoints call ──────────────
    // Full signature:
    //   void cv::projectPoints(
    //       InputArray  objectPoints,       // 3D world points (Vec3f, Nx3)
    //       InputArray  rvec,               // rotation vector from solvePnP
    //       InputArray  tvec,               // translation vector from solvePnP
    //       InputArray  cameraMatrix,       // 3x3 intrinsic matrix
    //       InputArray  distCoeffs,         // distortion coefficients
    //       OutputArray imagePoints,        // OUTPUT: projected 2D pixel coords
    //       OutputArray jacobian=noArray(), // not needed for rendering
    //       double      aspectRatio=0       // 0 = unconstrained
    //   )
    std::vector<cv::Point2f> points2D;
    cv::projectPoints(
        points3D,       // all 3D line endpoints in one batch
        rvec,           // rotation from solvePnP
        tvec,           // translation from solvePnP
        cameraMatrix,   // intrinsic matrix from calibration
        distCoeffs,     // distortion coefficients from calibration
        points2D,       // OUTPUT: 2D projected pixel positions
        cv::noArray(),  // jacobian: not needed for drawing
        0               // aspectRatio: 0 = unconstrained
    );

    // ── Step 3: Draw each line segment using its projected 2D endpoints ───────
    cv::Rect imgBounds(0, 0, frame.cols, frame.rows);

    for (size_t i = 0; i < m_lines.size(); ++i)
    {
        cv::Point p1 = static_cast<cv::Point>(points2D[i * 2    ]);
        cv::Point p2 = static_cast<cv::Point>(points2D[i * 2 + 1]);

        // Skip segments where either endpoint projects outside the image.
        // This prevents drawing artifacts when the board is at a steep angle.
        if (!imgBounds.contains(p1) || !imgBounds.contains(p2)) continue;

        cv::line(frame, p1, p2, m_lines[i].color, m_lines[i].thickness);
    }
}
