////////////////////////////////////////////////////////////////////////////////
// VirtualEagleObject.cpp - Virtual Eagle Object Implementation
// Author:      Krushna Sanjay Sharma
// Description: Implements a 3D wireframe eagle floating above the dollar bill.
//              Geometry defined in local space, transformed at build time.
//              cv::projectPoints() projects all endpoints in one batch call.
//
// Extension:   Multiple targets in the scene
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#include "VirtualEagleObject.h"
#include <iostream>
#include <cmath>

// ── Draw colors (BGR) ─────────────────────────────────────────────────────────
static const cv::Scalar COLOR_BODY (255,  80,   0);   // blue   — torso
static const cv::Scalar COLOR_HEAD (200,   0,   0);   // dark blue — head
static const cv::Scalar COLOR_BEAK (  0, 165, 255);   // amber  — beak
static const cv::Scalar COLOR_WING (200, 180, 100);   // light blue — wings
static const cv::Scalar COLOR_TAIL (  0, 165, 255);   // amber  — tail

// ----------------------------------------------------------------------------
// addLine()
// Applies axis flips, scale, then offset before storing the segment.
// ----------------------------------------------------------------------------
void VirtualEagleObject::addLine(const cv::Vec3f& start,
                                  const cv::Vec3f& end,
                                  const cv::Scalar& color,
                                  int thickness)
{
    cv::Vec3f s(start[0], start[1] * m_ySign, start[2] * m_zSign);
    cv::Vec3f e(end[0],   end[1]   * m_ySign, end[2]   * m_zSign);
    s = s * m_scale + m_offset;
    e = e * m_scale + m_offset;
    m_lines.push_back({ s, e, color, thickness });
}

// ----------------------------------------------------------------------------
// build()
// ----------------------------------------------------------------------------
void VirtualEagleObject::build(cv::Vec3f offset, float scale,
                                 bool flipY, bool flipZ)
{
    m_lines.clear();
    m_offset = offset;
    m_scale  = scale;
    m_ySign  = flipY ? -1.0f : 1.0f;
    m_zSign  = flipZ ? -1.0f : 1.0f;

    buildBody();
    buildHead();
    buildWings();
    buildTail();

    std::cout << "[VirtualEagleObject] Eagle built: "
              << m_lines.size() << " segments"
              << "  scale=" << scale
              << "  flipY=" << (flipY ? "yes" : "no")
              << "  flipZ=" << (flipZ ? "yes" : "no") << "\n";
}

// ----------------------------------------------------------------------------
// buildBody()
// Diamond-shaped torso — front and back faces connected by edges.
// ----------------------------------------------------------------------------
void VirtualEagleObject::buildBody()
{
    float w = E_BODY_W;
    float d = E_BODY_W * 0.6f;
    float midZ = (E_BODY_Z0 + E_BODY_Z1) * 0.5f;

    // Front face diamond (Y = -d)
    cv::Vec3f fT( 0, -d, E_BODY_Z1);   // top
    cv::Vec3f fR( w, -d, midZ);         // right
    cv::Vec3f fB( 0, -d, E_BODY_Z0);   // bottom
    cv::Vec3f fL(-w, -d, midZ);         // left

    // Back face diamond (Y = +d)
    cv::Vec3f bT( 0,  d, E_BODY_Z1);
    cv::Vec3f bR( w,  d, midZ);
    cv::Vec3f bB( 0,  d, E_BODY_Z0);
    cv::Vec3f bL(-w,  d, midZ);

    // Front face
    addLine(fT, fR, COLOR_BODY, 2);
    addLine(fR, fB, COLOR_BODY, 2);
    addLine(fB, fL, COLOR_BODY, 2);
    addLine(fL, fT, COLOR_BODY, 2);

    // Back face
    addLine(bT, bR, COLOR_BODY, 2);
    addLine(bR, bB, COLOR_BODY, 2);
    addLine(bB, bL, COLOR_BODY, 2);
    addLine(bL, bT, COLOR_BODY, 2);

    // Connecting edges
    addLine(fT, bT, COLOR_BODY, 1);
    addLine(fR, bR, COLOR_BODY, 1);
    addLine(fB, bB, COLOR_BODY, 1);
    addLine(fL, bL, COLOR_BODY, 1);
}

// ----------------------------------------------------------------------------
// buildHead()
// Hexagonal cross-section head with a beak pointing in +X direction.
// ----------------------------------------------------------------------------
void VirtualEagleObject::buildHead()
{
    float hw  = E_HEAD_W;
    float hd  = E_HEAD_W * 0.6f;
    float zm  = (E_HEAD_Z0 + E_HEAD_Z1) * 0.5f;
    float hr  = (E_HEAD_Z1 - E_HEAD_Z0) * 0.5f;

    // 6-point hexagon in the XZ plane at Y = -hd
    std::vector<cv::Vec3f> hex;
    for (int i = 0; i < 6; ++i)
    {
        float a = static_cast<float>(i) * static_cast<float>(CV_PI) / 3.0f;
        hex.emplace_back(hw * std::cos(a), -hd, zm + hr * std::sin(a));
    }

    // Front hexagon ring
    for (int i = 0; i < 6; ++i)
        addLine(hex[i], hex[(i + 1) % 6], COLOR_HEAD, 2);

    // Beak — V shape pointing right
    cv::Vec3f bkBase1( hw, -hd, zm + 0.1f);
    cv::Vec3f bkBase2( hw, -hd, zm - 0.1f);
    cv::Vec3f bkTip  ( hw + 0.7f, -hd, zm);
    addLine(bkBase1, bkTip,   COLOR_BEAK, 2);
    addLine(bkBase2, bkTip,   COLOR_BEAK, 2);
    addLine(bkBase1, bkBase2, COLOR_BEAK, 1);

    // Neck connector from body top to head bottom
    addLine(cv::Vec3f(0, -E_BODY_W * 0.6f, E_BODY_Z1),
            cv::Vec3f(0, -hd, E_HEAD_Z0),
            COLOR_HEAD, 1);
}

// ----------------------------------------------------------------------------
// buildWings()
// Two spread wings — each is a quad with a rib, attached at body sides.
// ----------------------------------------------------------------------------
void VirtualEagleObject::buildWings()
{
    float bodyD  = E_BODY_W * 0.6f;
    float midZ   = (E_BODY_Z0 + E_BODY_Z1) * 0.5f;

    // ── Left wing ────────────────────────────────────────────────────────────
    cv::Vec3f lR1(-E_BODY_W, -bodyD, midZ + 0.3f);  // root top
    cv::Vec3f lR2(-E_BODY_W, -bodyD, midZ - 0.3f);  // root bottom
    cv::Vec3f lT1(-E_WING_W,  0.0f,  E_WING_ZT);    // tip top
    cv::Vec3f lT2(-E_WING_W,  0.0f,  E_WING_ZB);    // tip bottom
    cv::Vec3f lM (-E_WING_W * 0.55f, -bodyD * 0.5f, E_WING_ZT + 0.2f); // mid

    addLine(lR1, lM,  COLOR_WING, 2);
    addLine(lM,  lT1, COLOR_WING, 2);
    addLine(lT1, lT2, COLOR_WING, 2);
    addLine(lT2, lR2, COLOR_WING, 2);
    addLine(lR2, lR1, COLOR_WING, 1);
    addLine(lR1, lT2, COLOR_WING, 1);  // rib

    // ── Right wing ───────────────────────────────────────────────────────────
    cv::Vec3f rR1( E_BODY_W, -bodyD, midZ + 0.3f);
    cv::Vec3f rR2( E_BODY_W, -bodyD, midZ - 0.3f);
    cv::Vec3f rT1( E_WING_W,  0.0f,  E_WING_ZT);
    cv::Vec3f rT2( E_WING_W,  0.0f,  E_WING_ZB);
    cv::Vec3f rM ( E_WING_W * 0.55f, -bodyD * 0.5f, E_WING_ZT + 0.2f);

    addLine(rR1, rM,  COLOR_WING, 2);
    addLine(rM,  rT1, COLOR_WING, 2);
    addLine(rT1, rT2, COLOR_WING, 2);
    addLine(rT2, rR2, COLOR_WING, 2);
    addLine(rR2, rR1, COLOR_WING, 1);
    addLine(rR1, rT2, COLOR_WING, 1);  // rib
}

// ----------------------------------------------------------------------------
// buildTail()
// Three feather lines fanning downward from the body base.
// ----------------------------------------------------------------------------
void VirtualEagleObject::buildTail()
{
    cv::Vec3f bL(-0.3f, 0.0f, E_BODY_Z0);
    cv::Vec3f bM( 0.0f, 0.0f, E_BODY_Z0);
    cv::Vec3f bR( 0.3f, 0.0f, E_BODY_Z0);

    cv::Vec3f tL(-0.5f, 0.0f, E_TAIL_Z);
    cv::Vec3f tM( 0.0f, 0.0f, E_TAIL_Z - 0.2f);
    cv::Vec3f tR( 0.5f, 0.0f, E_TAIL_Z);

    addLine(bL, tL, COLOR_TAIL, 2);
    addLine(bM, tM, COLOR_TAIL, 2);
    addLine(bR, tR, COLOR_TAIL, 2);
    addLine(tL, tM, COLOR_TAIL, 1);
    addLine(tM, tR, COLOR_TAIL, 1);
}

// ----------------------------------------------------------------------------
// draw() - Core rendering
// Collects all 3D endpoints, calls cv::projectPoints() once, draws lines.
// ----------------------------------------------------------------------------
void VirtualEagleObject::draw(cv::Mat&        frame,
                                const cv::Mat& rvec,
                                const cv::Mat& tvec,
                                const cv::Mat& cameraMatrix,
                                const cv::Mat& distCoeffs) const
{
    if (m_lines.empty()) return;

    // Collect all 3D endpoints — index i*2 = start, i*2+1 = end
    std::vector<cv::Vec3f> pts3D;
    pts3D.reserve(m_lines.size() * 2);
    for (const auto& seg : m_lines)
    {
        pts3D.push_back(seg.start);
        pts3D.push_back(seg.end);
    }

    // Project all points in one cv::projectPoints call
    std::vector<cv::Point2f> pts2D;
    cv::projectPoints(
        pts3D, rvec, tvec,
        cameraMatrix, distCoeffs,
        pts2D, cv::noArray(), 0
    );

    // Draw each segment
    cv::Rect bounds(0, 0, frame.cols, frame.rows);
    for (size_t i = 0; i < m_lines.size(); ++i)
    {
        cv::Point p1 = static_cast<cv::Point>(pts2D[i * 2    ]);
        cv::Point p2 = static_cast<cv::Point>(pts2D[i * 2 + 1]);
        if (!bounds.contains(p1) || !bounds.contains(p2)) continue;
        cv::line(frame, p1, p2, m_lines[i].color, m_lines[i].thickness);
    }
}
