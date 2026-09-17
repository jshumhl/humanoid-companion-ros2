"""Frame sources: a live camera or a video file, both via OpenCV."""

import threading
import time

import cv2


def open_source(source, interval_sec):
    if isinstance(source, int) or str(source).startswith("/dev/video"):
        return CameraSource(source, interval_sec)
    return VideoFileSource(source, interval_sec)


class CameraSource:
    """Reads a camera on a background thread and keeps only the newest frame.

    Detection and speech are slower than the camera frame rate, so reading
    frames on demand would return stale buffered frames.
    """

    def __init__(self, index, interval_sec):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index!r}")
        self._interval = interval_sec
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def frames(self, timeout_sec=None):
        """Yield new frames. With `timeout_sec`, raise if the camera stops delivering."""
        last_frame_at = time.monotonic()
        while self._running:
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                if timeout_sec is not None and time.monotonic() - last_frame_at > timeout_sec:
                    raise RuntimeError("Camera stopped delivering frames")
                time.sleep(0.01)
                continue
            yield frame
            last_frame_at = time.monotonic()
            time.sleep(self._interval)

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


class VideoFileSource:
    """Reads a video file, sampling one frame per interval of video time."""

    def __init__(self, path, interval_sec):
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file {path!r}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._stride = max(1, round(fps * interval_sec))

    def frames(self):
        index = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            if index % self._stride == 0:
                yield frame
            index += 1

    def close(self):
        self._cap.release()
