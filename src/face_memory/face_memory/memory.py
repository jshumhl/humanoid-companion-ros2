"""Names for faces, kept in memory for one session."""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .embedder import Face, unit

KNOWN = "known"      # similarity >= match_threshold
UNSURE = "unsure"    # unsure_threshold <= similarity < match_threshold
UNKNOWN = "unknown"  # below unsure_threshold, or nobody remembered yet


class NoFace(Exception):
    """None of the frames had a face large enough to enroll."""


@dataclass(frozen=True)
class Sighting:
    face: Face
    status: str               # KNOWN, UNSURE or UNKNOWN
    name: Optional[str]       # closest remembered person; None if UNKNOWN
    similarity: float         # cosine similarity to that person, -1 to 1


class FaceMemory:
    """Remembers faces by name until the session goes idle or the process ends.

    Nothing is written to disk. Each person keeps up to `max_samples_per_name`
    averaged embeddings, and a face is compared with the closest of them.
    """

    def __init__(self, embedder, match_threshold=0.5, unsure_threshold=0.363,
                 session_idle_sec=1800.0, min_face_px=40, max_samples_per_name=5,
                 clock=time.monotonic):
        if not 0.0 < unsure_threshold <= match_threshold <= 1.0:
            raise ValueError("need 0 < unsure_threshold <= match_threshold <= 1")
        self._embedder = embedder
        self._match = match_threshold
        self._unsure = unsure_threshold
        self._idle_sec = session_idle_sec
        self._min_face_px = min_face_px
        self._max_samples = max_samples_per_name
        self._clock = clock
        self._people = {}           # name -> list of unit embeddings, oldest first
        self._last_used = clock()
        self._lock = threading.Lock()

    def names(self):
        with self._lock:
            self._expire()
            return list(self._people)

    def forget(self):
        with self._lock:
            self._people.clear()

    def enroll(self, name, frames):
        """Remember the largest face in `frames` as `name`. Returns the face used.

        The embeddings of that face across the frames are averaged into one
        sample. Saying the same name again adds a sample. A face that already
        matched someone else is taken off that person: being told a new name
        for a face is a correction.
        """
        name = clean_name(name)
        if not name:
            raise ValueError("name is empty")
        faces = [f for f in (self._largest_face(frame) for frame in frames) if f is not None]
        if not faces:
            raise NoFace()
        sample = unit(np.mean([f.embedding for f in faces], axis=0))

        with self._lock:
            self._expire()
            for other in list(self._people):
                if other == name:
                    continue
                kept = [s for s in self._people[other] if float(s @ sample) < self._match]
                if kept:
                    self._people[other] = kept
                else:
                    del self._people[other]
            samples = self._people.setdefault(name, [])
            samples.append(sample)
            del samples[:-self._max_samples]
            self._last_used = self._clock()
        return max(faces, key=lambda f: f.size)

    def recall(self, frame):
        """A Sighting for every usable face in `frame`, largest face first.

        One name is given to at most one face: if two faces match the same
        person, the closer match keeps the name and the other is UNKNOWN.
        """
        faces = sorted(self._usable(frame), key=lambda f: f.size, reverse=True)
        with self._lock:
            self._expire()
            sightings = [self._identify(face) for face in faces]
            self._last_used = self._clock()

        best = {}
        for i, s in enumerate(sightings):
            if s.name is not None and (s.name not in best or
                                       s.similarity > sightings[best[s.name]].similarity):
                best[s.name] = i
        return [s if s.name is None or best[s.name] == i
                else Sighting(s.face, UNKNOWN, None, s.similarity)
                for i, s in enumerate(sightings)]

    def _identify(self, face):
        name, similarity = None, -1.0
        for person, samples in self._people.items():
            score = max(float(s @ face.embedding) for s in samples)
            if score > similarity:
                name, similarity = person, score
        if similarity >= self._match:
            return Sighting(face, KNOWN, name, similarity)
        if similarity >= self._unsure:
            return Sighting(face, UNSURE, name, similarity)
        return Sighting(face, UNKNOWN, None, similarity)

    def _usable(self, frame):
        return [f for f in self._embedder.faces(frame) if f.size >= self._min_face_px]

    def _largest_face(self, frame):
        faces = self._usable(frame)
        return max(faces, key=lambda f: f.size) if faces else None

    def _expire(self):
        if self._people and self._clock() - self._last_used > self._idle_sec:
            self._people.clear()


def clean_name(name):
    """Strip spaces and the punctuation speech recognition tends to add."""
    return str(name).strip().strip("。，！？、,.!?:：;；\"'“”‘’ ").strip()
