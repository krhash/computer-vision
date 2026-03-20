/*
 * PoseEstimator.cpp - Pose Estimator Class Implementation
 * Author:      Krushna Sanjay Sharma
 * Description: Implements pose estimation using cv::solvePnP and projects
 *              outer board corners and 3D axes onto the image using
 *              cv::projectPoints for real-time AR visualization.
 *
 * Reference:
 *   OpenCV Documentation - solvePnP
 *   https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
 *
 * Date: March 2026
 */

#include "PoseEstimator.h"
#include <iostream>
#include <iomanip>

/* Constructor */
PoseEstimator::PoseEstimator(const std::string& calibrationFile, int cameraId)
    : m_calibrationFile(calibrationFile)
    , m_cameraId(cameraId)
    , m_displayMode(DisplayMode::AXES_ONLY)
    , m_poseValid(false)
    , m_boardSize(BOARD_WIDTH, BOARD_HEIGHT)
{
    /* Initialize rvec and tvec as empty 3x1 double matrices.
     * solvePnP will fill these each frame. */
    m_rvec = cv::Mat::zeros(3, 1, CV_64F);
    m_tvec = cv::Mat::zeros(3, 1, CV_64F);
}

/* run() - Main video loop for Task 4 */
bool PoseEstimator::run()
{
    /* Load calibration from XML */
    if (!loadCalibration())
    {
        std::cerr << "[ERROR] Failed to load calibration. "
                  << "Run calibrateCamera.exe first.\n";
        return false;
    }

    /* Open webcam */
    cv::VideoCapture cap(m_cameraId);
    if (!cap.isOpened())
    {
        std::cerr << "[ERROR] Cannot open camera ID: " << m_cameraId << "\n";
        return false;
    }

    std::cout << "========================================\n";
    std::cout << " Pose Estimator - Task 4\n";
    std::cout << "========================================\n";
    std::cout << " Calibration: " << m_calibrationFile << "\n";
    std::cout << " Point camera at the chessboard target.\n";
    std::cout << " Rotation and translation printed each frame.\n";
    std::cout << " Controls:\n";
    std::cout << "   SPACE  - cycle display mode (corners / axes / both)\n";
    std::cout << "   'q'/ESC - quit\n";
    std::cout << "========================================\n\n";

    // Pre-build the fixed 3D world point set once — it never changes
    std::vector<cv::Vec3f> worldPoints;
    buildWorldPoints(worldPoints);

    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        std::vector<cv::Point2f> corners;
        bool found = detectCorners(frame, corners);

        if (found && estimatePose(corners))
        {
            printPose();

            if (m_displayMode == DisplayMode::CORNERS_ONLY ||
                m_displayMode == DisplayMode::CORNERS_AXES)
                projectOuterCorners(frame);

            if (m_displayMode == DisplayMode::AXES_ONLY ||
                m_displayMode == DisplayMode::CORNERS_AXES)
                projectAxes(frame);
        }

        // Overlay status and current pose values on the frame
        overlayStatus(frame, found && m_poseValid);

        cv::imshow("Pose Estimator", frame);

        char key = static_cast<char>(cv::waitKey(30));
        if (key == 'q' || key == 27) break;

        // Cycle display mode: CORNERS_AXES → CORNERS_ONLY → AXES_ONLY → ...
        if (key == ' ')
        {
            switch (m_displayMode)
            {
                case DisplayMode::CORNERS_AXES:
                    m_displayMode = DisplayMode::CORNERS_ONLY;
                    std::cout << "[Mode] Outer corners only\n"; break;
                case DisplayMode::CORNERS_ONLY:
                    m_displayMode = DisplayMode::AXES_ONLY;
                    std::cout << "[Mode] 3D axes only\n"; break;
                case DisplayMode::AXES_ONLY:
                    m_displayMode = DisplayMode::CORNERS_AXES;
                    std::cout << "[Mode] Corners + axes\n"; break;
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return true;
}

/* loadCalibration()
 * Reads camera_matrix and distortion_coefficients from the calibration XML. */
bool PoseEstimator::loadCalibration()
{
    cv::FileStorage fs(m_calibrationFile, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "[ERROR] Cannot open calibration file: "
                  << m_calibrationFile << "\n";
        return false;
    }

    fs["camera_matrix"]           >> m_cameraMatrix;
    fs["distortion_coefficients"] >> m_distCoeffs;
    fs.release();

    // Validate that we actually read something
    if (m_cameraMatrix.empty() || m_distCoeffs.empty())
    {
        std::cerr << "[ERROR] Calibration file is missing required keys "
                  << "(camera_matrix / distortion_coefficients).\n";
        return false;
    }

    std::cout << "[INFO] Calibration loaded from: " << m_calibrationFile << "\n";
    std::cout << "[INFO] Camera matrix:\n"          << m_cameraMatrix    << "\n";
    std::cout << "[INFO] Distortion coefficients:\n"<< m_distCoeffs      << "\n\n";

    return true;
}

/* detectCorners()
 * Finds and refines chessboard corners. Reuses the same approach as Task 1. */
bool PoseEstimator::detectCorners(const cv::Mat& frame,
                                   std::vector<cv::Point2f>& corners)
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    int flags = cv::CALIB_CB_ADAPTIVE_THRESH
              | cv::CALIB_CB_NORMALIZE_IMAGE
              | cv::CALIB_CB_FAST_CHECK;

    bool found = cv::findChessboardCorners(gray, m_boardSize, corners, flags);

    if (found)
    {
        // Refine to sub-pixel accuracy — no drawing, pure detection only
        cv::cornerSubPix(
            gray, corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(
                cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001)
        );
    }

    return found;
}

/* buildWorldPoints()
 * Fixed 3D world coordinates for all internal chessboard corners.
 * Convention: (col, -row, 0)
 *   +Y away from board (north), -Y down along rows, +Z toward camera. */
void PoseEstimator::buildWorldPoints(std::vector<cv::Vec3f>& pointSet) const
{
    pointSet.clear();
    pointSet.reserve(BOARD_WIDTH * BOARD_HEIGHT);

    for (int row = 0; row < BOARD_HEIGHT; ++row)
        for (int col = 0; col < BOARD_WIDTH; ++col)
            pointSet.emplace_back(
                static_cast<float>(col),    // X = column
                static_cast<float>(-row),   // Y = -row (+Y away from board)
                0.0f                        // Z = 0 (planar)
            );
}

/* estimatePose() - Core of Task 4
 *
 * cv::solvePnP parameters:
 *   objectPoints  : 3D world coords of board corners (Vec3f vector)
 *   imagePoints   : detected 2D corner positions     (Point2f vector)
 *   cameraMatrix  : 3x3 intrinsic matrix from calibration
 *   distCoeffs    : distortion coefficients from calibration
 *   rvec          : OUTPUT rotation vector (Rodrigues, 3x1)
 *   tvec          : OUTPUT translation vector (3x1, in world units = squares)
 *   useExtrinsicGuess : false — compute from scratch each frame
 *   flags         : SOLVEPNP_ITERATIVE — Levenberg-Marquardt, best for planar
 *
 * What the outputs mean (Task 4 requirement):
 *   rvec: axis-angle rotation. ||rvec|| = angle in radians.
 *         Rotating the board changes rvec direction.
 *   tvec: position of the world origin (0,0,0 = upper-left board corner)
 *         expressed in camera coordinates (in square units).
 *         tvec[2] ~ distance from camera to board.
 *         Move camera right → tvec[0] becomes more negative.
 *         Move camera closer → tvec[2] decreases. */
bool PoseEstimator::estimatePose(const std::vector<cv::Point2f>& corners)
{
    std::vector<cv::Vec3f> worldPoints;
    buildWorldPoints(worldPoints);

    try
    {
        bool ok = cv::solvePnP(
            worldPoints,      // 3D object points in world space
            corners,          // 2D image points detected this frame
            m_cameraMatrix,   // intrinsic matrix (from calibration XML)
            m_distCoeffs,     // distortion coefficients (from calibration XML)
            m_rvec,           // OUTPUT: rotation vector  (Rodrigues)
            m_tvec,           // OUTPUT: translation vector
            false,            // useExtrinsicGuess: false = solve from scratch
            cv::SOLVEPNP_ITERATIVE  // method: Levenberg-Marquardt
        );

        m_poseValid = ok;
        return ok;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[ERROR] solvePnP failed: " << e.what() << "\n";
        m_poseValid = false;
        return false;
    }
}

/* printPose()
 * Prints rvec and tvec to console. Required by Task 4 to verify values
 * change correctly as the camera or board is moved.
 *
 * Expected behaviour to verify:
 *   Move camera LEFT  → tvec[0] increases  (board origin moves right in cam)
 *   Move camera RIGHT → tvec[0] decreases
 *   Move camera CLOSER→ tvec[2] decreases
 *   Move camera AWAY  → tvec[2] increases
 *   Tilt board        → rvec direction changes noticeably */
void PoseEstimator::printPose() const
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[Pose] "
              << "rvec: ["
              << m_rvec.at<double>(0) << ", "
              << m_rvec.at<double>(1) << ", "
              << m_rvec.at<double>(2) << "]  "
              << "tvec: ["
              << m_tvec.at<double>(0) << ", "
              << m_tvec.at<double>(1) << ", "
              << m_tvec.at<double>(2) << "]\n";
}

/* projectOuterCorners() - Task 5a
 *
 * Projects the 4 outer corners of the chessboard grid onto the image plane.
 *
 * cv::projectPoints transformation chain (per OpenCV docs):
 *   World coords → Camera coords (rvec/tvec) → Normalized coords
 *   → Distortion applied → Pixel coords (cameraMatrix)
 *
 * Full signature:
 *   void cv::projectPoints(
 *       InputArray  objectPoints,        // 3D world points (Vec3f, Nx3)
 *       InputArray  rvec,                // rotation vector from solvePnP
 *       InputArray  tvec,                // translation vector from solvePnP
 *       InputArray  cameraMatrix,        // 3x3 intrinsic matrix
 *       InputArray  distCoeffs,          // distortion coefficients
 *       OutputArray imagePoints,         // OUTPUT: 2D pixel positions (Point2f)
 *       OutputArray jacobian=noArray(),  // optional — not needed for rendering
 *       double      aspectRatio=0        // optional — 0 = no constraint
 *   )
 *
 * The 4 outer corners in world coordinates (square units, Z=0):
 *   Top-left     : (0,  0,  0)
 *   Top-right    : (8,  0,  0)   — last col internal corner index
 *   Bottom-left  : (0, -5,  0)   — last row internal corner index (negative)
 *   Bottom-right : (8, -5,  0) */
void PoseEstimator::projectOuterCorners(cv::Mat& frame) const
{
    if (!m_poseValid) return;

    // 4 outer corners. Y = -row so bottom corners are negative Y.
    std::vector<cv::Vec3f> corners3D = {
        { 0.0f,  0.0f, 0.0f },   // top-left     (0,  0)
        { 8.0f,  0.0f, 0.0f },   // top-right    (8,  0)
        { 0.0f, -5.0f, 0.0f },   // bottom-left  (0, -5)
        { 8.0f, -5.0f, 0.0f }    // bottom-right (8, -5)
    };

    /* Project 3D world corners → 2D image pixel positions.
     * Jacobian omitted (cv::noArray()) — only needed for optimization, not rendering. */
    std::vector<cv::Point2f> projected;
    cv::projectPoints(
        corners3D,          // 3D object points in world space  (Vec3f vector)
        m_rvec,             // rotation vector    from solvePnP
        m_tvec,             // translation vector from solvePnP
        m_cameraMatrix,     // 3x3 intrinsic matrix
        m_distCoeffs,       // distortion coefficients
        projected,          // OUTPUT: 2D pixel coordinates
        cv::noArray(),      // jacobian: not needed for rendering
        0                   // aspectRatio: 0 = unconstrained
    );

    cv::Rect imgBounds(0, 0, frame.cols, frame.rows);

    cv::Scalar dotColor (0, 165, 255);   // orange
    cv::Scalar lineColor(0, 165, 255);   // orange

    std::vector<std::string> labels = { "TL", "TR", "BL", "BR" };
    for (int i = 0; i < 4; ++i)
    {
        cv::Point pt = static_cast<cv::Point>(projected[i]);
        if (!imgBounds.contains(pt)) continue;

        // Small dot
        cv::circle(frame, pt, 5, cv::Scalar(0, 0, 0), -1);
        cv::circle(frame, pt, 4, dotColor, -1);

        // Label drawn ABOVE the dot so it never hides under lines
        cv::Point labelPos = pt + cv::Point(-10, -10);
        cv::putText(frame, labels[i], labelPos,
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 3);
        cv::putText(frame, labels[i], labelPos,
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, dotColor, 1);
    }

    // Board outline in orange with dark shadow for contrast
    auto inBounds = [&](cv::Point2f p) {
        return imgBounds.contains(static_cast<cv::Point>(p));
    };
    auto drawLine = [&](cv::Point2f a, cv::Point2f b) {
        if (!inBounds(a) || !inBounds(b)) return;
        cv::line(frame, a, b, cv::Scalar(0, 0, 0), 3);  // dark shadow
        cv::line(frame, a, b, lineColor, 1);             // orange on top
    };
    drawLine(projected[0], projected[1]);   // TL → TR
    drawLine(projected[1], projected[3]);   // TR → BR
    drawLine(projected[3], projected[2]);   // BR → BL
    drawLine(projected[2], projected[0]);   // BL → TL
}

/* projectAxes() - Task 5b
 *
 * Projects 3D coordinate axes from the board origin onto the image using
 * cv::projectPoints. These axes also serve as the anchor frame for Task 6.
 *
 * Axis endpoints in world coordinates (length = 3 squares):
 *   Origin : (0, 0,  0)  — top-left internal corner of chessboard
 *   X tip  : (3, 0,  0)  drawn in BLUE
 *   Y tip  : (0, 3,  0)  drawn in GREEN  (positive Y = downward in our convention)
 *   Z tip  : (0, 0, -3)  drawn in RED    (negative Z = toward viewer/camera)
 *
 * The Z axis pointing toward the viewer (negative Z) means a virtual object
 * placed at Z < 0 will float above the board toward the camera — which is
 * exactly what Task 6 requires. */
void PoseEstimator::projectAxes(cv::Mat& frame) const
{
    if (!m_poseValid) return;

    /* Axis tips in world space. Convention: (col, -row, 0)
     *   +X = right along columns
     *   +Y = away from board (north) — because Y = -row
     *   +Z = toward camera           — because Z positive = toward viewer */
    std::vector<cv::Vec3f> axisPoints = {
        { 0.0f,  0.0f,  0.0f },   // [0] origin
        { 3.0f,  0.0f,  0.0f },   // [1] X tip — right along columns
        { 0.0f,  3.0f,  0.0f },   // [2] Y tip — AWAY from board (+Y north)
        { 0.0f,  0.0f,  3.0f }    // [3] Z tip — toward camera   (+Z)
    };

    /* Project all 4 points in one call — more efficient than calling per-axis.
     * Jacobian omitted as it is only needed for optimization, not drawing. */
    std::vector<cv::Point2f> projected;
    cv::projectPoints(
        axisPoints,         // 3D world points  (Vec3f vector, Nx3 format)
        m_rvec,             // rotation vector    from solvePnP
        m_tvec,             // translation vector from solvePnP
        m_cameraMatrix,     // 3x3 intrinsic matrix
        m_distCoeffs,       // distortion coefficients
        projected,          // OUTPUT: 2D pixel coordinates
        cv::noArray(),      // jacobian: not needed for rendering
        0                   // aspectRatio: 0 = unconstrained
    );

    cv::Rect  imgBounds(0, 0, frame.cols, frame.rows);
    cv::Point origin = static_cast<cv::Point>(projected[0]);
    cv::Point xTip   = static_cast<cv::Point>(projected[1]);
    cv::Point yTip   = static_cast<cv::Point>(projected[2]);
    cv::Point zTip   = static_cast<cv::Point>(projected[3]);

    // Only draw if the origin itself is on screen
    if (!imgBounds.contains(origin)) return;

    /* Draw axes as thick arrowed lines.
     * OpenCV BGR color order: X=Blue, Y=Green, Z=Red
     * arrowedLine params: img, pt1, pt2, color, thickness, lineType, shift, tipLength */
    if (imgBounds.contains(xTip))
        cv::arrowedLine(frame, origin, xTip, cv::Scalar(255, 0,   0  ), 3, 8, 0, 0.2);
    if (imgBounds.contains(yTip))
        cv::arrowedLine(frame, origin, yTip, cv::Scalar(0,   255, 0  ), 3, 8, 0, 0.2);
    if (imgBounds.contains(zTip))
        cv::arrowedLine(frame, origin, zTip, cv::Scalar(0,   0,   255), 3, 8, 0, 0.2);

    // Axis labels at the tips
    if (imgBounds.contains(xTip))
        cv::putText(frame, "X", xTip + cv::Point(5, 0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0,   0  ), 2);
    if (imgBounds.contains(yTip))
        cv::putText(frame, "Y", yTip + cv::Point(5, 0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,   255, 0  ), 2);
    if (imgBounds.contains(zTip))
        cv::putText(frame, "Z", zTip + cv::Point(5, 0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,   0,   255), 2);

    // Small white dot at the origin for visual reference
    cv::circle(frame, origin, 5, cv::Scalar(255, 255, 255), -1);
}

/* overlayStatus()
 * Draws current rvec/tvec values and instructions onto the video frame. */
void PoseEstimator::overlayStatus(cv::Mat& frame, bool poseFound) const
{
    cv::Scalar color = poseFound
        ? cv::Scalar(0, 255, 0)
        : cv::Scalar(0, 0, 255);

    // Line 1: pose status
    std::string status = poseFound ? "Pose estimated" : "No board detected";
    cv::putText(frame, status, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

    if (poseFound)
    {
        auto fmt = [](double v) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << v;
            return ss.str();
        };

        // Line 2: tvec (dark outline + bright text for readability)
        std::string tvecStr = "tvec: ["
            + fmt(m_tvec.at<double>(0)) + ", "
            + fmt(m_tvec.at<double>(1)) + ", "
            + fmt(m_tvec.at<double>(2)) + "]";
        cv::putText(frame, tvecStr, cv::Point(10, 55),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 3);
        cv::putText(frame, tvecStr, cv::Point(10, 55),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

        // Line 3: rvec
        std::string rvecStr = "rvec: ["
            + fmt(m_rvec.at<double>(0)) + ", "
            + fmt(m_rvec.at<double>(1)) + ", "
            + fmt(m_rvec.at<double>(2)) + "]";
        cv::putText(frame, rvecStr, cv::Point(10, 75),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 3);
        cv::putText(frame, rvecStr, cv::Point(10, 75),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

        // Line 4: display mode
        std::string modeStr;
        switch (m_displayMode)
        {
            case DisplayMode::CORNERS_ONLY:  modeStr = "Mode: Corners";       break;
            case DisplayMode::AXES_ONLY:     modeStr = "Mode: Axes";          break;
            case DisplayMode::CORNERS_AXES:  modeStr = "Mode: Corners+Axes";  break;
        }
        cv::putText(frame, modeStr, cv::Point(10, 95),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 3);
        cv::putText(frame, modeStr, cv::Point(10, 95),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    }
}
