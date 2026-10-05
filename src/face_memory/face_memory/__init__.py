"""Remember faces by name for one session, and say who is in front of the camera."""

from .config import Config, Phrases, config_from_dict, load_config
from .embedder import Face, FaceEmbedder, SFaceEmbedder
from .memory import KNOWN, UNKNOWN, UNSURE, FaceMemory, NoFace, Sighting
from .service import FaceService, build_service

__all__ = [
    "Config", "Phrases", "config_from_dict", "load_config",
    "Face", "FaceEmbedder", "SFaceEmbedder",
    "KNOWN", "UNKNOWN", "UNSURE", "FaceMemory", "NoFace", "Sighting",
    "FaceService", "build_service",
]
