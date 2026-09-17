"""Single-shot scene description, for callers that want one sentence on demand."""

import time

from .sentence import build_sentence, count_classes
from .source import CameraSource, open_source

# Cameras often return dark or unfocused frames right after opening.
CAMERA_WARMUP_SEC = 0.5
CAMERA_TIMEOUT_SEC = 5.0

_detectors = {}


def _detector_for(config):
    # Loading the model takes a second or two, so reuse it across calls.
    from .detector import Detector

    key = (config.model, config.confidence, config.imgsz)
    if key not in _detectors:
        _detectors[key] = Detector(config.model, config.confidence, config.imgsz)
    return _detectors[key]


def _grab_frame(source):
    if isinstance(source, CameraSource):
        warm_until = time.monotonic() + CAMERA_WARMUP_SEC
        for frame in source.frames(timeout_sec=CAMERA_TIMEOUT_SEC):
            if time.monotonic() >= warm_until:
                return frame
    else:
        for frame in source.frames():
            return frame
    raise RuntimeError("No frame available from source")


def describe_scene(config) -> str:
    """Capture one frame, run detection, return the Chinese sentence.

    No speech, no loop. `config` is an object_narrator Config. For a video
    file source, the first frame is used.
    """
    detector = _detector_for(config)
    source = open_source(config.source, interval_sec=0)
    try:
        frame = _grab_frame(source)
    finally:
        source.close()
    counts = count_classes(detector.detect(frame))
    return build_sentence(counts, config.language)
