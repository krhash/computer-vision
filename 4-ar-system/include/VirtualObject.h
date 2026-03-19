////////////////////////////////////////////////////////////////////////////////
// VirtualObject.h - Virtual Object Class Header
// Author:      Krushna Sanjay Sharma
// Description: Declares the VirtualObject class responsible for defining a
//              3D line-based rocket model in world space and projecting it
//              onto the image plane using cv::projectPoints.
//
// Task coverage:
//   Task 6 - Create and render a virtual 3D object (rocket) floating above
//            the chessboard target, correctly oriented as the camera moves.
//
// Object design:
//   The rocket is defined entirely as line segments between 3D world-space
//   points. It is centered at (4, -2.5, 0) — the middle of the 9x6 board —
//   and floats above it (Z < 0, toward the camera).
//
//   Four parts, four colors (BGR):
//     Body      : box  x:[3,5] y:[1.5,3.5] z:[0,-3]   → Blue  (255, 100, 0)
//     Nose cone : pyramid from body top to apex (4,2.5,-5) → Cyan  (255, 255, 0)
//     Fins      : 4 triangular fins at the base          → Amber (0,  180, 255)
//     Exhaust   : triangle below the base                → Red   (0,   50, 255)
//     Window    : small square on the body front face    → Green (0,  220, 100)
//
// Coordinate convention (matches CameraCalibration and PoseEstimator):
//   X → rightward along board columns
//   Y → downward along board rows (negative = upward)
//   Z → negative = toward the camera (above the board)
//
// Key OpenCV function:
//   cv::projectPoints() — projects the 3D line endpoints onto the 2D image
//   using rvec, tvec, cameraMatrix, distCoeffs from PoseEstimator.
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// LineSegment
// Represents one 3D line — a start point, end point, and draw color.
// The VirtualObject stores the rocket as a flat list of these.
////////////////////////////////////////////////////////////////////////////////
struct LineSegment
{
    cv::Vec3f start;    // 3D world-space start point
    cv::Vec3f end;      // 3D world-space end point
    cv::Scalar color;   // BGR draw color
    int thickness;      // line thickness in pixels
};

////////////////////////////////////////////////////////////////////////////////
// VirtualObject
//
// Stores the rocket geometry as a list of LineSegments defined in 3D world
// space. On each frame, draw() collects all unique endpoints, calls
// cv::projectPoints() once to project them all to 2D, then draws each line
// between its projected endpoints.
//
// Usage:
//   VirtualObject rocket;
//   rocket.buildRocket();          // define geometry once
//   rocket.draw(frame, rvec, tvec, cameraMatrix, distCoeffs);  // each frame
////////////////////////////////////////////////////////////////////////////////
class VirtualObject
{
public:
    VirtualObject() = default;

    // -------------------------------------------------------------------------
    // buildRocket()
    // Populates m_lines with all line segments that make up the rocket.
    // Call once after construction. Geometry defined in world-space square units.
    // -------------------------------------------------------------------------
    void buildRocket();

    // -------------------------------------------------------------------------
    // draw()
    // Projects all 3D line endpoints onto the image plane using
    // cv::projectPoints, then draws each line segment in its assigned color.
    //
    // Parameters (all from PoseEstimator after a successful solvePnP call):
    //   frame        : BGR video frame to draw onto
    //   rvec         : rotation vector    (3x1 CV_64F, Rodrigues)
    //   tvec         : translation vector (3x1 CV_64F)
    //   cameraMatrix : 3x3 intrinsic matrix (CV_64F)
    //   distCoeffs   : distortion coefficients (CV_64F)
    // -------------------------------------------------------------------------
    void draw(cv::Mat&        frame,
              const cv::Mat& rvec,
              const cv::Mat& tvec,
              const cv::Mat& cameraMatrix,
              const cv::Mat& distCoeffs) const;

    // -------------------------------------------------------------------------
    // clear()
    // Removes all line segments (useful for rebuilding geometry).
    // -------------------------------------------------------------------------
    void clear() { m_lines.clear(); }

private:
    // -------------------------------------------------------------------------
    // addLine()
    // Convenience helper — appends one LineSegment to m_lines.
    // -------------------------------------------------------------------------
    void addLine(const cv::Vec3f& start,
                 const cv::Vec3f& end,
                 const cv::Scalar& color,
                 int thickness = 2);

    // -------------------------------------------------------------------------
    // buildBody()    — blue box (4 vertical edges + top/bottom faces)
    // buildNose()    — cyan pyramid from top face to apex
    // buildFins()    — amber triangular fins at the base corners
    // buildExhaust() — red exhaust triangle below the base
    // buildWindow()  — green square viewport on the front face
    //
    // Each helper appends its line segments to m_lines.
    // -------------------------------------------------------------------------
    void buildBody();
    void buildNose();
    void buildFins();
    void buildExhaust();
    void buildWindow();

    // All line segments making up the rocket
    std::vector<LineSegment> m_lines;

    // ── Rocket geometry constants (world space, square units) ─────────────────
    // World points use (col, -row, 0) so Y must be NEGATIVE to sit on the board.
    // Body centered at column 4, row 2.5 → world Y = -2.5
    // Z POSITIVE = toward camera (above board)
    static constexpr float X_LEFT   =  3.0f;
    static constexpr float X_RIGHT  =  5.0f;
    static constexpr float Y_FRONT  = -1.5f;  // -row 1.5 = front face
    static constexpr float Y_BACK   = -3.5f;  // -row 3.5 = back face
    static constexpr float Z_BASE   =  0.0f;  // sits on board surface
    static constexpr float Z_TOP    =  3.0f;  // top of body (3 squares above board)
    static constexpr float Z_APEX   =  5.0f;  // nose cone tip (5 squares above board)
    static constexpr float Z_FIN    =  2.0f;  // fin attachment height on body
    static constexpr float Z_EXHST  = -1.0f;  // exhaust tip (below board surface)
    static constexpr float X_CENTER =  4.0f;  // horizontal center of rocket
    static constexpr float Y_CENTER = -2.5f;  // depth center (-row 2.5)
};
