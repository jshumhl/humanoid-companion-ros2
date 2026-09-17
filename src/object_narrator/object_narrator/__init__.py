"""Detect objects with YOLO26n and describe them aloud in Chinese."""

from .config import Config, load_config
from .scene import describe_scene

__all__ = ["Config", "describe_scene", "load_config"]
