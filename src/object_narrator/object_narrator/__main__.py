"""Command-line entry point: python -m object_narrator --config config.yaml"""

import argparse
import sys

from .config import load_config, validate
from .sentence import build_sentence, count_classes


class SceneTracker:
    """Decides when a scene should be spoken.

    A scene (the set of classes and their counts) is spoken once it has been
    seen for `stable_frames` consecutive detections and differs from the last
    scene spoken. This stops flickering detections from repeating speech.
    """

    def __init__(self, stable_frames, announce_empty):
        self._stable_frames = stable_frames
        self._announce_empty = announce_empty
        self._candidate = None
        self._streak = 0
        self._last_spoken = None

    def update(self, counts):
        """Return True when `counts` should be spoken now."""
        scene = tuple(sorted(counts))
        if scene == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = scene, 1

        if self._streak < self._stable_frames or scene == self._last_spoken:
            return False
        self._last_spoken = scene
        return bool(scene) or self._announce_empty


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="object_narrator",
        description="Detect objects with YOLO26n and describe them aloud in Chinese.",
    )
    parser.add_argument("--config", default="config.yaml", help="path to the YAML config file")
    parser.add_argument("--source", help="override the config source (camera index or video path)")
    parser.add_argument("--no-audio", action="store_true", help="print sentences without speaking")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    if args.source is not None:
        config.source = args.source
        validate(config)

    # Imported here so config errors surface before the slow model/torch import.
    from .detector import Detector
    from .source import open_source

    speaker = None
    if not args.no_audio:
        from .speaker import Speaker
        speaker = Speaker(config.language, config.voice, config.player)

    detector = Detector(config.model, config.confidence, config.imgsz)
    tracker = SceneTracker(config.stable_frames, config.announce_empty)
    source = open_source(config.source, config.interval_sec)

    try:
        for frame in source.frames():
            counts = count_classes(detector.detect(frame))
            if not tracker.update(counts):
                continue
            sentence = build_sentence(counts, config.language)
            print(sentence, flush=True)
            if speaker:
                speaker.speak(sentence)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
