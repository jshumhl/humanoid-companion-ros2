# object_narrator

Watches a camera or video file, detects objects with YOLO26n, and says what it
sees in Chinese:

```
[person, person, dog, chair]  →  我看到了两个人、一只狗和一把椅子。  →  🔊 edge-tts
```

- Runs on CPU. No GPU, robot SDK, or vendor camera driver needed; frames come from OpenCV.
- Merges repeated classes into counts, with the right measure word for each
  (一**只**狗, 两**辆**汽车, 三**把**椅子).
- Speaks a scene only after it has held steady for a few detections, and only
  when it differs from the last scene spoken. Detections that flicker on and
  off don't cause repeated speech.
- Supports Mandarin in Simplified (`zh-CN`) or Traditional (`zh-TW`)
  characters, and Cantonese (`zh-HK`).

This is a plain Python module that doesn't depend on ROS. It has a
`COLCON_IGNORE` file, so `colcon build` skips it.

## Requirements

- Python 3.10+
- Internet access. edge-tts uses Microsoft's online speech service, and
  `yolo26n.pt` (about 5 MB) downloads on the first run.
- An MP3 player. `afplay` comes with macOS. On Ubuntu, run `sudo apt install mpg123`.

## Install

```bash
cd src/object_narrator
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

To get a smaller CPU-only PyTorch build on Linux, install it before the
requirements:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

To import it from other modules, such as `voice_agent`, install it as a package
from the repository root: `pip install -e src/object_narrator`.

## Use from Python

`describe_scene()` returns one sentence for one frame. It doesn't speak and
doesn't loop:

```python
from object_narrator import describe_scene, load_config

config = load_config("src/object_narrator/config.yaml")
print(describe_scene(config))   # 我看到了两个人和一把椅子。
```

It reuses the detector across calls. For a camera, it waits 0.5 s after opening
so exposure can settle, and raises if no frame arrives within 5 s. For a video
file, it uses the first frame.

## Configure

Edit `config.yaml`:

```yaml
source: 0          # camera index, /dev/videoN, or path to a video file
language: zh-CN    # zh-CN | zh-TW | zh-HK
confidence: 0.5    # minimum detection confidence, 0-1
```

Optional settings (all commented out in `config.yaml`):

| Key | Default | Meaning |
|---|---|---|
| `model` | `yolo26n.pt` | Ultralytics model file |
| `imgsz` | `640` | Inference size. Try `320` for faster CPU inference. |
| `interval_sec` | `1.0` | Time between detections. For video files, this is video time. |
| `stable_frames` | `2` | Number of consecutive detections a scene must hold before it is spoken |
| `announce_empty` | `false` | Speak "我没有看到任何东西。" when the scene becomes empty |
| `voice` | per language | Any edge-tts voice (`edge-tts --list-voices`) |
| `player` | auto | MP3 player command, e.g. `ffplay -nodisp -autoexit` |

Default voices are `zh-CN-XiaoxiaoNeural`, `zh-TW-HsiaoChenNeural`, and
`zh-HK-HiuGaaiNeural`.

## Usage

```bash
cd src/object_narrator

# Webcam, using config.yaml
python -m object_narrator --config config.yaml

# Video file, overriding the config source
python -m object_narrator --config config.yaml --source ~/Videos/living_room.mp4

# Print sentences only, without speaking (no audio player needed)
python -m object_narrator --config config.yaml --source clip.mp4 --no-audio
```

Example output for a street clip with `language: zh-CN`:

```
我看到了四个人和一辆公交车。
我看到了两个人。
```

With `language: zh-HK`, the same clip gives:

```
我見到四個人同一架巴士。
我見到兩個人。
```

Each line is also spoken aloud. For a camera, stop with `Ctrl+C`. For a video
file, the program exits when the file ends.

## Tests

```bash
pip install pytest
python -m pytest tests
```

## Layout

| File | Purpose |
|---|---|
| `object_narrator/__main__.py` | CLI and main loop. Decides when a scene is spoken. |
| `object_narrator/scene.py` | `describe_scene()`: one frame in, one sentence out |
| `object_narrator/config.py` | Loads and validates the YAML config |
| `object_narrator/source.py` | Camera (latest frame only) and video file (sampled) readers |
| `object_narrator/detector.py` | YOLO26n inference on CPU |
| `object_narrator/sentence.py` | Counts classes and builds the Chinese sentence |
| `object_narrator/vocab.py` | Chinese names and measure words for the 80 COCO classes |
| `object_narrator/speaker.py` | Speech synthesis with edge-tts and audio playback |
