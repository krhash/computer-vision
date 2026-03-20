/*
 * VirtualEagleObject.h - Virtual Eagle Object Class Header
 * Author:      Krushna Sanjay Sharma
 * Description: Declares the VirtualEagleObject class responsible for defining
 *              a 3D line-based eagle model in world space and projecting it
 *              onto the image plane using cv::projectPoints.
 *              Designed for the dollar bill target in the multi-target AR app.
 *
 * Extension:   Multiple targets in the scene
 *
 * Object design:
 *   The eagle is defined as line segments between 3D world-space points.
 *   Built in local space centered at (0,0,0), then transformed via
 *   offset/scale/flipY/flipZ at build time.
 *
 *   Four parts, four colors (BGR):
 *     Body  : diamond-shaped torso     → Blue       (255,  80,   0)
 *     Head  : hexagonal head + beak    → Dark blue  (200,   0,   0)
 *     Wings : spread wings left+right  → Light blue (200, 180, 100)
 *     Tail  : fan of tail feathers     → Amber      (  0, 165, 255)
 *
 * Coordinate convention (dollar bill):
 *   X → rightward along bill width
 *   Y → downward (positive Y = down on bill surface)
 *   Z → positive = toward the camera (above the bill)
 *   flipY=true, flipZ=true for bill target (standard OpenCV solvePnP)
 *
 * Key OpenCV function:
 *   cv::projectPoints() — projects 3D line endpoints onto the 2D image
 *
 * Date: March 2026
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/*
 * EagleLineSegment
 * Represents one 3D line — start point, end point, color, thickness.
 */
struct EagleLineSegment
{
    cv::Vec3f  start;
    cv::Vec3f  end;
    cv::Scalar color;
    int        thickness;
};

/*
 * VirtualEagleObject
 *
 * Stores the eagle geometry as a list of EagleLineSegments in 3D world space.
 * On each frame, draw() collects all endpoints, calls cv::projectPoints() once,
 * then draws each line segment between its projected 2D positions.
 *
 * Usage:
 *   VirtualEagleObject eagle;
 *   eagle.build(offset, scale, flipY, flipZ);   // once
 *   eagle.draw(frame, rvec, tvec, cam, dist);    // each frame
 */
class VirtualEagleObject
{
public:
    VirtualEagleObject() : m_scale(1.0f), m_offset(0,0,0),
                            m_ySign(1.0f), m_zSign(1.0f) {}

    /*
     * build()
     * Constructs all eagle line segments in world space.
     * Call once after construction.
     *
     * offset : world position of eagle base center
     * scale  : uniform scale (1.0 = cm units for dollar bill)
     * flipY  : negate Y — bill uses +Y downward
     * flipZ  : negate Z — bill solvePnP convention
     */
    void build(cv::Vec3f offset = cv::Vec3f(0, 0, 0),
               float     scale  = 1.0f,
               bool      flipY  = false,
               bool      flipZ  = false);

    /*
     * draw()
     * Projects all 3D line endpoints via cv::projectPoints, draws each line.
     * If spinning is enabled, rotates the object around its center each frame.
     */
    void draw(cv::Mat&        frame,
              const cv::Mat& rvec,
              const cv::Mat& tvec,
              const cv::Mat& cameraMatrix,
              const cv::Mat& distCoeffs) const;

    /* clear() */
    void clear() { m_lines.clear(); }

private:
    /* addLine() — appends one transformed EagleLineSegment
     * Applies ySign, zSign, scale, and offset before storing */
    void addLine(const cv::Vec3f& start,
                 const cv::Vec3f& end,
                 const cv::Scalar& color,
                 int thickness = 2);

    /* Part builders */
    void buildBody();    // diamond-shaped torso
    void buildHead();    // hexagonal head + beak
    void buildWings();   // spread wings left and right
    void buildTail();    // fan of tail feathers

    /* Line storage */
    std::vector<EagleLineSegment> m_lines;

    /* Transform */
    cv::Vec3f m_offset;
    float     m_scale;
    float     m_ySign;
    float     m_zSign;

    /* Eagle geometry constants (local space, units before scaling) */
    static constexpr float E_BODY_W  = 0.6f;    // body half-width in X/Y
    static constexpr float E_BODY_Z0 = 0.0f;    // body base Z
    static constexpr float E_BODY_Z1 = 2.0f;    // body top Z
    static constexpr float E_HEAD_Z0 = 2.2f;    // head bottom Z
    static constexpr float E_HEAD_Z1 = 3.4f;    // head top Z
    static constexpr float E_HEAD_W  = 0.5f;    // head half-width
    static constexpr float E_WING_W  = 3.0f;    // wing tip X distance
    static constexpr float E_WING_ZB = 1.0f;    // wing base Z
    static constexpr float E_WING_ZT = 2.0f;    // wing tip Z
    static constexpr float E_TAIL_Z  = -0.8f;   // tail feather tip Z
};
