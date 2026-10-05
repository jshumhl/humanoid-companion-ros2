"""Face detection and embedding behind a swappable interface.

FaceMemory only needs something with a `faces(frame)` method that returns
Face objects whose embeddings are unit vectors, so that the dot product of two
embeddings is their cosine similarity. SFaceEmbedder is the default; any other
model can be dropped in by implementing the same method.
"""

from dataclasses import dataclass
from typing import List, Protocol, Tuple

import numpy as np

from .models import DETECTOR, RECOGNIZER, ensure_model


@dataclass(frozen=True)
class Face:
    box: Tuple[int, int, int, int]   # x, y, width, height in the frame's pixels
    score: float                     # detection confidence, 0-1
    embedding: np.ndarray            # unit length

    @property
    def size(self):
        """The shorter side of the box, in pixels."""
        return min(self.box[2], self.box[3])


class FaceEmbedder(Protocol):
    def faces(self, frame) -> List[Face]:
        """Every face found in a BGR frame, with its embedding."""


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("embedding is all zeros")
    return vector / norm


class SFaceEmbedder:
    """OpenCV's YuNet detector and SFace recognizer, on CPU.

    Frames are scaled down to `max_image_side` for detection; the face crop
    for the embedding is taken from the full-resolution frame.
    """

    def __init__(self, model_dir, detect_confidence=0.8, max_image_side=640, download=True):
        import cv2

        self._cv2 = cv2
        detector_path = ensure_model(DETECTOR, model_dir, download)
        recognizer_path = ensure_model(RECOGNIZER, model_dir, download)
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), detect_confidence, 0.3, 5000)
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self._max_side = max_image_side

    def faces(self, frame):
        height, width = frame.shape[:2]
        scale = min(1.0, self._max_side / max(height, width))
        small = frame if scale == 1.0 else self._cv2.resize(
            frame, (round(width * scale), round(height * scale)),
            interpolation=self._cv2.INTER_AREA)
        self._detector.setInputSize((small.shape[1], small.shape[0]))
        _, rows = self._detector.detect(small)
        if rows is None:
            return []

        faces = []
        for row in rows:
            row = row.copy()
            row[:14] /= scale    # box and five landmarks back to full-resolution pixels
            crop = self._recognizer.alignCrop(frame, row)
            embedding = unit(self._recognizer.feature(crop))
            x, y, w, h = (int(round(v)) for v in row[:4])
            faces.append(Face((x, y, w, h), float(row[14]), embedding))
        return faces
