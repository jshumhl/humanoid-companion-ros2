"""Command-line entry point.

  python -m face_memory download-models
  python -m face_memory try --enroll 张三 a.jpg --enroll 李四 c.jpg b.jpg d.jpg
  python -m face_memory live --config config.yaml
"""

import argparse
import sys
from pathlib import Path

from .config import load_config
from .models import DETECTOR, RECOGNIZER, ModelUnavailable, ensure_model

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="face_memory",
                                     description="Remember faces by name for one session.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("download-models", help="fetch the model files into model_dir")

    try_ = commands.add_parser("try", help="enroll from still images, then recognise others")
    try_.add_argument("--enroll", nargs=2, action="append", metavar=("NAME", "IMAGE"),
                      default=[], help="remember the largest face in IMAGE as NAME (repeatable)")
    try_.add_argument("images", nargs="+", help="images to recognise faces in")

    commands.add_parser("live", help="type enroll NAME / who / forget / quit; uses `source`")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError, TypeError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    try:
        if args.command == "download-models":
            for model in (DETECTOR, RECOGNIZER):
                print(ensure_model(model, config.model_dir))
            return 0
        if args.command == "try":
            return try_images(config, args.enroll, args.images)
        return live(config)
    except ModelUnavailable as e:
        print(f"Model error: {e}", file=sys.stderr)
        return 2


def try_images(config, enrollments, images):
    from .memory import NoFace
    from .service import build_service, describe
    from .source import ImageFrames

    service = build_service(config)
    for name, image in enrollments:
        try:
            face = service.memory.enroll(name, ImageFrames(image).grab())
        except NoFace:
            print(f"enroll {name}: no usable face in {image}")
            return 1
        print(f"enroll {name}: face at {face.box} in {image}")

    for image in images:
        sightings = service.memory.recall(ImageFrames(image).grab()[0])
        print(f"\n{image}: {len(sightings)} face(s)")
        for s in sightings:
            print(f"  {str(s.face.box):24} {s.status:8} {s.name or '-':12} {s.similarity:+.3f}")
        print(f"  says: {describe(sightings, config.phrases)}")
    return 0


def live(config):
    from .service import build_service

    service = build_service(config)
    print(f"Source: {config.source!r}. Commands: enroll NAME | who | forget | quit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        command, _, rest = line.partition(" ")
        if command in ("quit", "exit"):
            return 0
        try:
            if command == "enroll":
                print(service.enroll_face(rest))
            elif command == "who":
                print(service.who_is_here())
            elif command == "forget":
                service.memory.forget()
                print("Forgotten.")
            elif command:
                print("Commands: enroll NAME | who | forget | quit")
        except RuntimeError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    sys.exit(main())
