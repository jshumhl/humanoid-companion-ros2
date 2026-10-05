"""Frame sources: a camera, an image file, or a ROS 2 image topic.

Each `grab(count)` opens the source, takes up to `count` recent frames and
lets go of it again, so other programs (such as a scene describer) can use the
same camera between calls.
"""

import time

import cv2
import numpy as np

ROS2_PREFIX = "ros2:"
CAMERA_WARMUP_FRAMES = 5     # let exposure settle after opening
FRAME_SPACING_SEC = 0.1      # between frames of one grab


def open_frames(source, timeout_sec=5.0):
    if isinstance(source, str) and source.startswith(ROS2_PREFIX):
        return RosImageFrames(source[len(ROS2_PREFIX):], timeout_sec)
    if isinstance(source, int) or str(source).startswith("/dev/video"):
        return CameraFrames(source, timeout_sec)
    return ImageFrames(source)


class ImageFrames:
    """One still image, returned once per grab."""

    def __init__(self, path):
        self._path = str(path)

    def grab(self, count=1):
        frame = cv2.imread(self._path)
        if frame is None:
            raise RuntimeError(f"Cannot read image {self._path!r}")
        return [frame]


class CameraFrames:
    def __init__(self, device, timeout_sec):
        self._device = device
        self._timeout = timeout_sec

    def grab(self, count=1):
        cap = cv2.VideoCapture(self._device)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self._device!r}; "
                                   f"another program may be using it")
            deadline = time.monotonic() + self._timeout
            frames, read = [], 0
            while len(frames) < count:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Camera {self._device!r} delivered no frame "
                                       f"within {self._timeout:.0f} s")
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                read += 1
                if read > CAMERA_WARMUP_FRAMES:
                    frames.append(frame)
                    if len(frames) < count:
                        time.sleep(FRAME_SPACING_SEC)
            return frames
        finally:
            cap.release()


class RosImageFrames:
    """Subscribes to a sensor_msgs/Image topic only while grabbing."""

    def __init__(self, topic, timeout_sec):
        try:
            import rclpy
            import rclpy.executors
            from sensor_msgs.msg import Image
        except ImportError as e:
            raise RuntimeError(
                f"source {ROS2_PREFIX}{topic} needs ROS 2 ({e}). "
                f"Run inside a sourced ROS 2 environment.") from e
        self._image_type = Image
        self._topic = topic
        self._timeout = timeout_sec
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("face_memory_frames")
        # Its own executor: other threads may be spinning their nodes on the global one.
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)

    def grab(self, count=1):
        from rclpy.qos import qos_profile_sensor_data

        messages = []
        subscription = self._node.create_subscription(
            self._image_type, self._topic, messages.append, qos_profile_sensor_data)
        try:
            deadline = time.monotonic() + self._timeout
            frames, last = [], 0.0
            while len(frames) < count:
                if time.monotonic() > deadline:
                    if frames:
                        return frames
                    raise RuntimeError(f"No image on {self._topic} within {self._timeout:.0f} s")
                self._executor.spin_once(timeout_sec=0.05)
                if messages and time.monotonic() - last >= FRAME_SPACING_SEC:
                    frames.append(image_to_bgr(messages[-1]))
                    messages.clear()
                    last = time.monotonic()
            return frames
        finally:
            self._node.destroy_subscription(subscription)


CHANNELS = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}
TO_BGR = {"rgb8": cv2.COLOR_RGB2BGR, "bgra8": cv2.COLOR_BGRA2BGR,
          "rgba8": cv2.COLOR_RGBA2BGR, "mono8": cv2.COLOR_GRAY2BGR}


def image_to_bgr(msg):
    """Convert a sensor_msgs/Image (8-bit colour or mono) to a BGR array."""
    channels = CHANNELS.get(msg.encoding)
    if channels is None:
        raise RuntimeError(f"Unsupported image encoding {msg.encoding!r}; "
                           f"expected one of {', '.join(CHANNELS)}")
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    frame = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding in TO_BGR:
        return cv2.cvtColor(frame, TO_BGR[msg.encoding])
    return frame.copy()
