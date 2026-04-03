# src/inference/live_predictor.py
# Project 5: Recognition using Deep Networks — Extension 3
# Author: Krushna Sanjay Sharma
# Description: LivePredictor processes individual webcam frames,
#              extracts a digit ROI, applies the MNIST preprocessing
#              pipeline, and runs inference using a trained DigitNetwork.
#
# Preprocessing pipeline (matches HandwrittenLoader exactly):
#   greyscale -> Otsu threshold -> invert -> resize 28x28 -> normalise

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Tuple

# MNIST normalisation constants — must match training pipeline
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)

# Digit class labels
DIGIT_LABELS = [str(i) for i in range(10)]


class LivePredictor:
    """
    Processes webcam frames for real-time digit recognition.

    Responsibilities:
        - Extract a centred square ROI from each frame
        - Apply the MNIST preprocessing pipeline to the ROI
        - Run the trained DigitNetwork and return prediction + confidence
        - Draw the prediction overlay onto the frame

    The ROI is a fixed square region in the centre of the frame,
    visualised as a green rectangle. Users hold their digit inside
    this box for recognition.

    Author: Krushna Sanjay Sharma
    """

    # ROI size as fraction of the smaller frame dimension
    ROI_FRACTION = 0.4

    # Overlay display constants
    OVERLAY_COLOR_HIGH = (0,   255,  0)    # green  — high confidence (>80%)
    OVERLAY_COLOR_MED  = (0,   200, 255)   # yellow — medium confidence (50-80%)
    OVERLAY_COLOR_LOW  = (0,   0,   255)   # red    — low confidence (<50%)
    BOX_COLOR          = (0,   255,  0)    # green ROI box
    BOX_THICKNESS      = 2

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialises the predictor.

        Args:
            model  (nn.Module):    Trained DigitNetwork in eval() mode.
            device (torch.device): CPU or CUDA device.
        """
        self._model  = model
        self._device = device
        self._model.eval()

        # Normalisation transform — matches HandwrittenLoader
        self._normalise = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """
        Processes one webcam frame end-to-end.

        Steps:
            1. Convert to greyscale
            2. Extract centred square ROI
            3. Apply MNIST preprocessing pipeline
            4. Run inference
            5. Draw overlay on original colour frame

        Args:
            frame (ndarray): BGR frame from cv2.VideoCapture.

        Returns:
            Tuple of:
                - display_frame (ndarray): Frame with prediction overlay.
                - prediction    (int):     Predicted digit class (0-9).
                - confidence    (float):   Confidence in [0, 1].
                - roi_processed (ndarray): 28x28 preprocessed ROI.
        """
        h, w    = frame.shape[:2]
        rx, ry, rw, rh = self._get_roi_coords(w, h)

        # Extract and preprocess ROI
        roi           = frame[ry:ry+rh, rx:rx+rw]
        roi_processed = self._preprocess_roi(roi)

        # Run inference
        prediction, confidence, log_probs = self._predict(roi_processed)

        # Draw overlay on a copy of the frame
        display_frame = self._draw_overlay(
            frame.copy(), rx, ry, rw, rh,
            prediction, confidence, log_probs,
        )

        return display_frame, prediction, confidence, roi_processed

    def get_roi_coords(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """
        Returns the ROI rectangle coordinates for a given frame size.

        Args:
            w (int): Frame width in pixels.
            h (int): Frame height in pixels.

        Returns:
            Tuple (x, y, width, height) of the ROI rectangle.
        """
        return self._get_roi_coords(w, h)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_roi_coords(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """
        Computes centred square ROI coordinates.

        Args:
            w (int): Frame width.
            h (int): Frame height.

        Returns:
            Tuple (x, y, size, size).
        """
        size = int(min(w, h) * self.ROI_FRACTION)
        x    = (w - size) // 2
        y    = (h - size) // 2
        return x, y, size, size

    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Applies the MNIST preprocessing pipeline to a BGR ROI.

        Pipeline (matches HandwrittenLoader._process_single_image):
            1. Convert BGR -> greyscale
            2. Otsu threshold at full resolution
            3. Invert (dark digit on white -> white digit on black)
            4. Resize to 28x28 using area interpolation

        Args:
            roi (ndarray): BGR ROI crop from the webcam frame.

        Returns:
            ndarray: Preprocessed 28x28 uint8 greyscale image.
        """
        # Step 1: greyscale
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Step 2: Otsu threshold at full resolution
        _, binary = cv2.threshold(
            grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Step 3: invert — dark digit on white -> white digit on black
        inverted = cv2.bitwise_not(binary)

        # Step 4: resize to 28x28
        resized = cv2.resize(inverted, (28, 28), interpolation=cv2.INTER_AREA)

        return resized.astype(np.uint8)

    def _predict(
        self,
        roi_processed: np.ndarray,
    ) -> Tuple[int, float, torch.Tensor]:
        """
        Runs inference on a preprocessed 28x28 ROI.

        Args:
            roi_processed (ndarray): 28x28 uint8 greyscale image.

        Returns:
            Tuple of:
                - prediction (int):          Predicted class index (0-9).
                - confidence (float):        Softmax probability of top class.
                - log_probs  (torch.Tensor): Full (10,) log-probability tensor.
        """
        # Normalise and convert to tensor (1, 1, 28, 28)
        tensor = self._normalise(roi_processed).unsqueeze(0).to(self._device)

        with torch.no_grad():
            log_probs  = self._model(tensor).squeeze(0)   # (10,)
            probs      = torch.exp(log_probs)              # convert log -> prob
            prediction = int(probs.argmax().item())
            confidence = float(probs[prediction].item())

        return prediction, confidence, log_probs.cpu()

    def _draw_overlay(
        self,
        frame:      np.ndarray,
        rx:         int,
        ry:         int,
        rw:         int,
        rh:         int,
        prediction: int,
        confidence: float,
        log_probs:  torch.Tensor,
    ) -> np.ndarray:
        """
        Draws the ROI box, prediction, confidence bar, and top-3
        probabilities onto the display frame.

        Args:
            frame      (ndarray): BGR frame to draw on.
            rx,ry,rw,rh (int):    ROI rectangle coordinates.
            prediction (int):     Predicted digit.
            confidence (float):   Confidence of top prediction.
            log_probs  (Tensor):  Full log-probability tensor.

        Returns:
            ndarray: Annotated BGR frame.
        """
        h, w = frame.shape[:2]

        # Choose colour based on confidence
        if confidence >= 0.80:
            color = self.OVERLAY_COLOR_HIGH
        elif confidence >= 0.50:
            color = self.OVERLAY_COLOR_MED
        else:
            color = self.OVERLAY_COLOR_LOW

        # Draw ROI rectangle
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh),
                      self.BOX_COLOR, self.BOX_THICKNESS)

        # Main prediction label above ROI box
        label = f"Digit: {prediction}  ({confidence*100:.1f}%)"
        cv2.putText(
            frame, label,
            (rx, ry - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
        )

        # Top-3 probabilities in bottom-left corner
        probs   = torch.exp(log_probs)
        top3    = torch.topk(probs, 3)
        y_start = h - 90

        cv2.putText(frame, "Top 3:", (10, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        for rank, (val, idx) in enumerate(
            zip(top3.values.tolist(), top3.indices.tolist())
        ):
            txt = f"  {idx}: {val*100:.1f}%"
            cv2.putText(
                frame, txt,
                (10, y_start + 22 * (rank + 1)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (200, 200, 200), 1, cv2.LINE_AA,
            )

        # Instructions bottom-right
        instructions = [
            "Q: quit",
            "S: screenshot",
            "R: reset",
        ]
        for i, txt in enumerate(instructions):
            cv2.putText(
                frame, txt,
                (w - 140, h - 60 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA,
            )

        return frame