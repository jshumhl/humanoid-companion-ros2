from pathlib import Path

import numpy as np
import pytest

from face_memory.config import Phrases
from face_memory.embedder import Face, unit

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = PACKAGE_DIR / "config.yaml"
DIM = 8


def vec(*pairs):
    """A unit vector from (axis, weight) pairs, e.g. vec((0, 1.0), (1, 0.5))."""
    v = np.zeros(DIM, dtype=np.float32)
    for axis, weight in pairs:
        v[axis] = weight
    return unit(v)


def at_similarity(base_axis, other_axis, similarity):
    """A unit vector whose cosine similarity with axis `base_axis` is `similarity`."""
    return vec((base_axis, similarity), (other_axis, float(np.sqrt(1 - similarity ** 2))))


def face(embedding, size=100, x=0):
    return Face((x, 0, size, size), 0.9, embedding)


class FakeEmbedder:
    """A frame is just the list of faces the embedder should report for it."""

    def __init__(self):
        self.calls = 0

    def faces(self, frame):
        self.calls += 1
        return list(frame)


class FakeFrames:
    """Returns the queued frames, one batch per grab."""

    def __init__(self, *batches):
        self.batches = list(batches)
        self.counts = []

    def grab(self, count=1):
        self.counts.append(count)
        return self.batches.pop(0)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def phrases():
    return Phrases()
