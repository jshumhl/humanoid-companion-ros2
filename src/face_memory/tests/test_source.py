from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_memory.source import CameraFrames, ImageFrames, image_to_bgr, open_frames


def image_msg(encoding, pixels, padding=0):
    height, width = pixels.shape[:2]
    rows = pixels.reshape(height, -1)
    step = rows.shape[1] + padding
    data = np.zeros((height, step), np.uint8)
    data[:, :rows.shape[1]] = rows
    return SimpleNamespace(encoding=encoding, height=height, width=width, step=step,
                           data=data.tobytes())


BGR = np.array([[[10, 20, 30], [40, 50, 60]]], np.uint8)


@pytest.mark.parametrize("encoding, pixels", [
    ("bgr8", BGR),
    ("rgb8", BGR[..., ::-1]),
    ("bgra8", np.dstack([BGR, np.full(BGR.shape[:2], 255, np.uint8)])),
    ("rgba8", np.dstack([BGR[..., ::-1], np.full(BGR.shape[:2], 255, np.uint8)])),
])
def test_image_to_bgr(encoding, pixels):
    np.testing.assert_array_equal(image_to_bgr(image_msg(encoding, pixels, padding=2)), BGR)


def test_mono_image_to_bgr():
    frame = image_to_bgr(image_msg("mono8", np.array([[7, 9]], np.uint8)))
    assert frame.shape == (1, 2, 3)
    assert frame[0, 1].tolist() == [9, 9, 9]


def test_unsupported_encoding():
    with pytest.raises(RuntimeError, match="16UC1"):
        image_to_bgr(image_msg("16UC1", BGR))


def test_image_frames(tmp_path):
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), np.zeros((4, 4, 3), np.uint8))
    (frame,) = ImageFrames(path).grab(3)
    assert frame.shape == (4, 4, 3)


def test_missing_image(tmp_path):
    with pytest.raises(RuntimeError):
        ImageFrames(tmp_path / "missing.jpg").grab()


def test_open_frames_picks_the_source_type(tmp_path):
    assert isinstance(open_frames(0), CameraFrames)
    assert isinstance(open_frames("/dev/video2"), CameraFrames)
    assert isinstance(open_frames(str(tmp_path / "a.jpg")), ImageFrames)


def test_ros2_source_without_ros_explains_itself():
    pytest.importorskip("cv2")
    try:
        import rclpy  # noqa: F401
        pytest.skip("ROS 2 is installed here")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="ROS 2"):
        open_frames("ros2:/camera/image")
