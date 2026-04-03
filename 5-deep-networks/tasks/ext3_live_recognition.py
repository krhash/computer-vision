# tasks/ext3_live_recognition.py
# Project 5: Recognition using Deep Networks — Extension 3
# Author: Krushna Sanjay Sharma
# Description: Live webcam digit recognition using a trained DigitNetwork.
#              Captures frames, extracts a centred ROI, preprocesses
#              identically to HandwrittenLoader, and overlays the
#              predicted digit and confidence in real time.
#
# Controls:
#   Q — quit
#   S — save screenshot to outputs/
#   R — reset (clears last prediction display)
#
# Requires: models/mnist_cnn.pth  (produced by task1_build_train.py)
#
# Usage:
#   python tasks/ext3_live_recognition.py
#   python tasks/ext3_live_recognition.py --camera 0 --model mnist_cnn.pth
#   python tasks/ext3_live_recognition.py --model gabor_network.pth

import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch

from src.inference.live_predictor import LivePredictor
from src.utils.device_utils       import get_device
from src.utils.model_io           import ModelIO
from src.network.digit_network    import DigitNetwork


# ------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------

DEFAULT_CAMERA_ID  = 0
DEFAULT_MODEL_FILE = "mnist_cnn.pth"
DEFAULT_MODEL_DIR  = "./models"
DEFAULT_OUTPUT_DIR = "./outputs"
WINDOW_TITLE       = "Live Digit Recognition — Project 5"
FRAME_WIDTH        = 840
FRAME_HEIGHT       = 620


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def load_model(
    model_file: str,
    model_dir:  str,
    device:     torch.device,
) -> DigitNetwork:
    """
    Loads the trained DigitNetwork from disk and sets eval() mode.

    Args:
        model_file (str):          Filename of the saved .pth model.
        model_dir  (str):          Directory containing the model file.
        device     (torch.device): Target device.

    Returns:
        DigitNetwork in eval() mode on the given device.
    """
    model_io = ModelIO(model_dir=model_dir)
    model    = DigitNetwork()
    model_io.load(model, model_file)
    model    = model.to(device)
    model.eval()
    return model


def open_camera(camera_id: int) -> cv2.VideoCapture:
    """
    Opens the webcam and sets the capture resolution.

    Args:
        camera_id (int): OpenCV camera index (0 = default webcam).

    Returns:
        cv2.VideoCapture: Opened capture object.

    Raises:
        RuntimeError: If the camera cannot be opened.
    """
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {camera_id}. "
            f"Check that a webcam is connected and not in use."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera opened: {actual_w}x{actual_h} @ camera index {camera_id}")

    return cap


def save_screenshot(
    frame:         np.ndarray,
    roi_processed: np.ndarray,
    prediction:    int,
    confidence:    float,
    output_dir:    str,
) -> None:
    """
    Saves a screenshot of the current frame and the preprocessed ROI.

    Two files are saved:
        ext3_screenshot_<timestamp>.png — full annotated frame
        ext3_roi_<timestamp>.png        — 28x28 preprocessed ROI (scaled up)

    Args:
        frame         (ndarray): Annotated BGR frame.
        roi_processed (ndarray): 28x28 preprocessed ROI.
        prediction    (int):     Current predicted digit.
        confidence    (float):   Current prediction confidence.
        output_dir    (str):     Directory to save screenshots.
    """
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save full annotated frame
    frame_path = out_path / f"ext3_screenshot_{ts}.png"
    cv2.imwrite(str(frame_path), frame)

    # Save ROI scaled up to 140x140 for visibility
    roi_big    = cv2.resize(roi_processed, (140, 140),
                            interpolation=cv2.INTER_NEAREST)
    roi_path   = out_path / f"ext3_roi_{ts}_pred{prediction}.png"
    cv2.imwrite(str(roi_path), roi_big)

    print(f"  Screenshot saved: {frame_path}")
    print(f"  ROI saved:        {roi_path}")
    print(f"  Prediction: {prediction}  Confidence: {confidence*100:.1f}%")


def build_roi_preview(roi_processed: np.ndarray, size: int = 112) -> np.ndarray:
    """
    Builds a scaled-up preview of the 28x28 preprocessed ROI.

    Displayed in the top-right corner of the live feed so the user
    can see exactly what the network is receiving.

    Args:
        roi_processed (ndarray): 28x28 uint8 preprocessed ROI.
        size          (int):     Preview size in pixels (default 112).

    Returns:
        ndarray: 3-channel BGR preview image of shape (size, size, 3).
    """
    preview = cv2.resize(roi_processed, (size, size),
                         interpolation=cv2.INTER_NEAREST)
    # Convert greyscale to BGR for overlay on colour frame
    return cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)


def embed_roi_preview(
    frame:        np.ndarray,
    roi_preview:  np.ndarray,
    padding:      int = 8,
) -> np.ndarray:
    """
    Embeds the ROI preview in the top-right corner of the frame.

    Draws a border around the preview for visibility.

    Args:
        frame       (ndarray): Full BGR display frame.
        roi_preview (ndarray): Scaled ROI preview image.
        padding     (int):     Padding from frame edge in pixels.

    Returns:
        ndarray: Frame with embedded ROI preview.
    """
    h, w      = frame.shape[:2]
    ph, pw    = roi_preview.shape[:2]
    y1        = padding
    y2        = y1 + ph
    x1        = w - pw - padding
    x2        = x1 + pw

    # Draw white border
    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                  (255, 255, 255), 2)

    # Label
    cv2.putText(frame, "Network input",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (200, 200, 200), 1, cv2.LINE_AA)

    frame[y1:y2, x1:x2] = roi_preview
    return frame


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def run_live_recognition(
    camera_id:  int,
    model_file: str,
    model_dir:  str,
    output_dir: str,
) -> None:
    """
    Main live recognition loop.

    Opens the webcam, loads the model, and runs the preprocessing +
    inference pipeline on every captured frame. Displays the annotated
    feed in a resizable OpenCV window.

    Controls:
        Q — quit
        S — save screenshot of current frame + ROI
        R — display reset (cosmetic only)

    Args:
        camera_id  (int): OpenCV camera index.
        model_file (str): Trained model filename.
        model_dir  (str): Directory containing model file.
        output_dir (str): Directory for screenshots.
    """
    device = get_device()

    # Load model
    print(f"\n  Loading model: {model_file}")
    model = load_model(model_file, model_dir, device)

    # Initialise predictor
    predictor = LivePredictor(model=model, device=device)

    # Open camera
    cap = open_camera(camera_id)

    print(f"\n  Starting live recognition...")
    print(f"  Hold a digit inside the green box.")
    print(f"  Controls:  Q = quit  |  S = screenshot  |  R = reset")
    print(f"  Window: {WINDOW_TITLE}\n")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, FRAME_WIDTH, FRAME_HEIGHT)

    last_prediction = -1
    last_confidence = 0.0
    last_roi        = None
    frame_count     = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  [WARNING] Failed to read frame. Retrying...")
                continue

            frame_count += 1

            # Run inference on every frame
            display_frame, prediction, confidence, roi_processed = \
                predictor.process_frame(frame)

            last_prediction = prediction
            last_confidence = confidence
            last_roi        = roi_processed

            # Embed the 28x28 preprocessed ROI preview (top-right corner)
            roi_preview   = build_roi_preview(roi_processed)
            display_frame = embed_roi_preview(display_frame, roi_preview)

            # Frame counter (bottom-left)
            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (10, display_frame.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (120, 120, 120), 1, cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_TITLE, display_frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:   # Q or Escape
                print("\n  Quit signal received.")
                break

            elif key == ord("s"):              # S — screenshot
                if last_roi is not None:
                    save_screenshot(
                        display_frame,
                        last_roi,
                        last_prediction,
                        last_confidence,
                        output_dir,
                    )

            elif key == ord("r"):              # R — reset display
                print(f"  Reset. Last prediction: {last_prediction} "
                      f"({last_confidence*100:.1f}%)")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("  Camera released. Window closed.")


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args(argv: list) -> argparse.Namespace:
    """
    Parses CLI arguments for standalone execution.

    Args:
        argv (list): sys.argv

    Returns:
        argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Extension 3 — Live webcam digit recognition."
    )
    parser.add_argument(
        "--camera",
        dest    = "camera_id",
        type    = int,
        default = DEFAULT_CAMERA_ID,
        help    = f"Webcam index (default: {DEFAULT_CAMERA_ID})",
    )
    parser.add_argument(
        "--model",
        dest    = "model_file",
        default = DEFAULT_MODEL_FILE,
        help    = f"Model filename (default: {DEFAULT_MODEL_FILE}). "
                  f"Use gabor_network.pth to test the Gabor network.",
    )
    parser.add_argument(
        "--model-dir",
        dest    = "model_dir",
        default = DEFAULT_MODEL_DIR,
        help    = f"Model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        dest    = "output_dir",
        default = DEFAULT_OUTPUT_DIR,
        help    = f"Screenshot output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv[1:])


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def main(argv: list) -> None:
    """
    Main function for Extension 3.

    Args:
        argv (list): sys.argv passed from the if __name__ guard.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  Extension 3: Live Webcam Digit Recognition")
    print(f"  Model:  {args.model_file}")
    print(f"  Camera: index {args.camera_id}")
    print("=" * 60)

    run_live_recognition(
        camera_id  = args.camera_id,
        model_file = args.model_file,
        model_dir  = args.model_dir,
        output_dir = args.output_dir,
    )

    print("\n[Extension 3] Complete.\n")


if __name__ == "__main__":
    main(sys.argv)