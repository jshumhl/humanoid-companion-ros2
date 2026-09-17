"""Load and validate the YAML config file."""

from dataclasses import dataclass, fields
from typing import Optional, Union

import yaml

from .vocab import LANGUAGES


@dataclass
class Config:
    # Required settings
    source: Union[int, str] = 0      # camera index or path to a video file
    language: str = "zh-CN"          # zh-CN | zh-TW | zh-HK
    confidence: float = 0.5          # detection confidence threshold, 0-1

    # Optional settings
    model: str = "yolo26n.pt"        # downloaded automatically on first run
    imgsz: int = 640                 # inference image size
    interval_sec: float = 1.0        # time between detections
    stable_frames: int = 2           # detections a scene must persist before it is spoken
    announce_empty: bool = False     # speak when nothing is detected
    voice: Optional[str] = None      # edge-tts voice; defaults per language
    player: Optional[str] = None     # audio player command; auto-detected if unset


def load_config(path):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")

    config = Config(**raw)
    validate(config)
    return config


def validate(config):
    if isinstance(config.source, str) and config.source.isdigit():
        config.source = int(config.source)
    if config.language not in LANGUAGES:
        raise ValueError(f"language must be one of {', '.join(LANGUAGES)}, got {config.language!r}")
    if not 0.0 <= config.confidence <= 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {config.confidence}")
    if config.interval_sec < 0:
        raise ValueError("interval_sec must be >= 0")
    if config.stable_frames < 1:
        raise ValueError("stable_frames must be >= 1")
