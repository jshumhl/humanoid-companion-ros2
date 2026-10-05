"""The two calls a voice agent needs, each returning one sentence to speak."""

import logging
import threading

from .memory import KNOWN, UNSURE, FaceMemory, NoFace, clean_name

log = logging.getLogger(__name__)

COUNTS = "零一两三四五六七八九十"


class FaceService:
    """enroll_face(name) and who_is_here(), over a FaceMemory and a frame source.

    Errors from the camera or the models are raised, not spoken: the caller
    decides what to say when the tool itself fails.
    """

    def __init__(self, memory, frames, phrases, frames_per_enroll=3):
        self.memory = memory
        self._frames = frames
        self._phrases = phrases
        self._frames_per_enroll = frames_per_enroll
        # One call at a time: the camera and the OpenCV models are not shared safely.
        self._lock = threading.Lock()

    def enroll_face(self, name):
        name = clean_name(name or "")
        if not name:
            return self._phrases.no_name
        with self._lock:
            frames = self._frames.grab(self._frames_per_enroll)
            try:
                self.memory.enroll(name, frames)
            except NoFace:
                return self._phrases.no_face
        log.info("Enrolled a face as %s", name)
        return self._phrases.enrolled.format(name=name)

    def who_is_here(self):
        with self._lock:
            frame = self._frames.grab(1)[-1]
            sightings = self.memory.recall(frame)
        for s in sightings:
            log.info("Face %s: %s %s (%.2f)", s.face.box, s.status, s.name or "-", s.similarity)
        return describe(sightings, self._phrases)


def describe(sightings, phrases):
    """One sentence saying who is in view, asking about at most one unsure face."""
    p = phrases
    if not sightings:
        return p.nobody
    known = [s.name for s in sightings if s.status == KNOWN]
    unsure = [s.name for s in sightings if s.status == UNSURE]
    strangers = len(sightings) - len(known) - len(unsure)

    parts = []
    if known:
        parts.append(p.known.format(names="和".join(_join_names(known))))
    if strangers and (known or unsure):
        parts.append(p.others.format(count=spoken_count(strangers)))
    elif strangers:
        parts.append(p.stranger if strangers == 1 else p.strangers)
    if unsure:
        # Ask about one person at a time; the largest face comes first.
        parts.append(p.unsure.format(name=unsure[0]))
    return "".join(parts)


def _join_names(names):
    """['甲', '乙', '丙'] -> ['甲、乙', '丙'], to be joined with 和."""
    if len(names) <= 2:
        return names
    return ["、".join(names[:-1]), names[-1]]


def spoken_count(n):
    return COUNTS[n] if 0 <= n < len(COUNTS) else str(n)


def build_service(config, download=True):
    """A FaceService for a face_memory Config, with the default SFace embedder."""
    from .embedder import SFaceEmbedder
    from .source import open_frames

    embedder = SFaceEmbedder(config.model_dir, config.detect_confidence,
                             config.max_image_side, download=download)
    memory = FaceMemory(embedder, config.match_threshold, config.unsure_threshold,
                        session_idle_sec=config.session_idle_min * 60,
                        min_face_px=config.min_face_px,
                        max_samples_per_name=config.max_samples_per_name)
    frames = open_frames(config.source, config.frame_timeout_sec)
    return FaceService(memory, frames, config.phrases, config.frames_per_enroll)
