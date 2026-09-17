"""YOLO26n object detection on CPU."""

from ultralytics import YOLO


class Detector:
    def __init__(self, model_path, confidence, imgsz):
        self._model = YOLO(model_path)
        self._confidence = confidence
        self._imgsz = imgsz

    def detect(self, frame):
        """Return the class name of every detection above the confidence threshold."""
        result = self._model.predict(
            frame,
            conf=self._confidence,
            imgsz=self._imgsz,
            device="cpu",
            verbose=False,
        )[0]
        return [result.names[int(c)] for c in result.boxes.cls]
