import cv2
import numpy as np

from object_narrator import scene
from object_narrator.config import Config


class FakeDetector:
    def __init__(self, names):
        self.names = names
        self.frames = []

    def detect(self, frame):
        self.frames.append(frame)
        return self.names


def write_video(path, n_frames=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(n_frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


def test_describe_scene_uses_one_frame_and_returns_sentence(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    write_video(video)
    detector = FakeDetector(["dog", "person", "dog"])
    monkeypatch.setattr(scene, "_detector_for", lambda config: detector)

    sentence = scene.describe_scene(Config(source=str(video), language="zh-CN"))

    assert sentence == "我看到了两只狗和一个人。"
    assert len(detector.frames) == 1


def test_describe_scene_empty(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    write_video(video)
    monkeypatch.setattr(scene, "_detector_for", lambda config: FakeDetector([]))

    assert scene.describe_scene(Config(source=str(video))) == "我没有看到任何东西。"
