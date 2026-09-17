# humanoid-companion-ros2

A ROS2 companion behavior framework for humanoid robots. Provides person
detection, head tracking, conversational speech, and a behavior state machine
that coordinates them into natural companion interactions.

## Overview

The framework runs a behavior state machine over four cooperating packages:

IDLE → PERSON_DETECTED → GREETING → CONVERSATION → RETURN_TO_IDLE


Perception publishes person detections, the behavior node consumes them and
drives state transitions, and the speech node handles conversation when the
state machine enters `CONVERSATION`.

## Packages

| Package | Purpose |
|---|---|
| `companion_bringup` | Launch files and system configuration |
| `companion_behavior` | Behavior state machine and transition logic |
| `companion_perception` | Person detection and head tracking |
| `companion_speech` | Speech recognition, language model, and synthesis |
| `companion_msgs` | Custom message and service definitions |

Standalone modules (no ROS required):

| Module | Purpose |
|---|---|
| [`object_narrator`](src/object_narrator/README.md) | YOLO26n object detection spoken as a Chinese sentence via edge-tts |
| [`voice_agent`](src/voice_agent/README.md) | Push-to-talk voice agent for 巴克机器人: ASR → LLM with tools → TTS |

First-time setup on Ubuntu 22.04: [SETUP.md](SETUP.md).

## Requirements

- ROS2 Humble
- Ubuntu 22.04
- Python 3.10

## Supported hardware

- **BXI ELF3** — 31 DOF humanoid, 2-DOF neck, RealSense D435i, Livox MID-360
- Development is possible on any Linux machine with a webcam; no robot required

## Quickstart

```bash
# Documentation in progress
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
EOF