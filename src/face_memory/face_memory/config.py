"""Load and validate the YAML config file."""

from dataclasses import dataclass, field, fields
from typing import Union

import yaml


@dataclass
class Phrases:
    """What the robot says. {name}, {names} and {count} are filled in."""

    enrolled: str = "好的，{name}，我记住你了。"
    no_face: str = "我没看清你的脸，可以正对着我，站近一点吗？"
    no_name: str = "可以再告诉我一次你的名字吗？"
    nobody: str = "我面前好像没有人。"
    known: str = "我看到了{names}。"
    unsure: str = "你是{name}吗？"
    stranger: str = "我还不认识你，可以告诉我你的名字吗？"
    strangers: str = "我还不认识你们，可以告诉我你们的名字吗？"
    others: str = "还有{count}位我还不认识。"


@dataclass
class Config:
    source: Union[int, str] = 0
    model_dir: str = "~/.cache/face_memory/models"
    match_threshold: float = 0.5
    unsure_threshold: float = 0.363
    session_idle_min: float = 30.0

    # Optional settings
    frames_per_enroll: int = 3
    detect_confidence: float = 0.8
    min_face_px: int = 40
    max_image_side: int = 640
    frame_timeout_sec: float = 5.0
    max_samples_per_name: int = 5
    phrases: Phrases = field(default_factory=Phrases)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return config_from_dict(raw)


def config_from_dict(raw):
    raw = dict(raw)
    phrases = raw.pop("phrases", None) or {}
    _check_keys(raw, Config, "")
    _check_keys(phrases, Phrases, "phrases.")
    config = Config(**raw, phrases=Phrases(**phrases))
    validate(config)
    return config


def validate(config):
    if isinstance(config.source, str) and config.source.isdigit():
        config.source = int(config.source)
    if isinstance(config.source, str) and config.source.startswith("ros2:"):
        _require(config.source[len("ros2:"):].startswith("/"),
                 f"source: a ROS 2 topic must start with /, got {config.source!r}")
    _require(0.0 < config.unsure_threshold <= config.match_threshold <= 1.0,
             "thresholds must satisfy 0 < unsure_threshold <= match_threshold <= 1, got "
             f"unsure_threshold={config.unsure_threshold}, match_threshold={config.match_threshold}")
    _require(config.session_idle_min > 0, "session_idle_min must be > 0")
    _require(config.frames_per_enroll >= 1, "frames_per_enroll must be >= 1")
    _require(0.0 < config.detect_confidence < 1.0, "detect_confidence must be between 0 and 1")
    _require(config.min_face_px >= 1, "min_face_px must be >= 1")
    _require(config.max_image_side >= 160, "max_image_side must be >= 160")
    _require(config.frame_timeout_sec > 0, "frame_timeout_sec must be > 0")
    _require(config.max_samples_per_name >= 1, "max_samples_per_name must be >= 1")
    for f in fields(Phrases):
        _require(getattr(config.phrases, f.name).strip() != "", f"phrases.{f.name} must not be empty")


def _check_keys(raw, cls, prefix):
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix.rstrip('.') or 'config'} must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(prefix + k for k in sorted(unknown))}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)
