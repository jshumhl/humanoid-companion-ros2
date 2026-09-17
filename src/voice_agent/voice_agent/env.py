"""Minimal .env loader, so secrets stay out of config.yaml and the shell history."""

import os
from pathlib import Path


def find_dotenv(start=None):
    """Nearest .env in `start` (default: cwd) or any parent directory."""
    directory = Path(start or os.getcwd()).resolve()
    for candidate in [directory, *directory.parents]:
        path = candidate / ".env"
        if path.is_file():
            return path
    return None


def load_dotenv(path=None):
    """Set KEY=VALUE lines as environment variables. Existing variables win.

    Returns the path loaded, or None.
    """
    path = Path(path) if path else find_dotenv()
    if path is None or not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return path
