# face_memory

Remembers faces by name for one session and says who is in front of the
camera, in Chinese:

```
"这是张三"   → enroll_face("张三") → 好的，张三，我记住你了。
"你认识我吗" → who_is_here()       → 我看到了张三。 / 你是张三吗？ / 我还不认识你，可以告诉我你的名字吗？
```

- Runs on CPU: 30–50 ms per frame on a 4-core x86 server. No GPU and no cloud service:
  frames are processed in this process and never sent anywhere.
- Memory only. Names and face embeddings are never written to disk, and are
  forgotten when the program exits or after `session_idle_min` without use.
- One-shot enrollment: one sentence, three camera frames, averaged.
- Low-confidence matches are asked about (你是张三吗？) rather than asserted.
- The face model is swappable behind a one-method interface.
- Plain Python, no ROS needed (a ROS 2 image topic can be the source). It has a
  `COLCON_IGNORE` file, so `colcon build` skips it.

## Licences

**The default models are licensed for commercial use.** They are OpenCV's
[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
face detector (MIT) and
[SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
face recognizer (Apache 2.0). Both are downloaded from the OpenCV model zoo and
checked by SHA-256.

**Check the licence of any model you swap in.** Several popular face models
are not licensed for commercial use. For example, InsightFace's pretrained
weights (`buffalo_l` and others) are for non-commercial research only.

## Install

```bash
pip install -e src/face_memory
python -m face_memory download-models     # about 39 MB into ~/.cache/face_memory/models
```

Without `download-models`, the models download on first use.

## Try it on still images

`try` enrolls the largest face in each `--enroll` image, then reports every
face it finds in the other images:

```bash
python -m face_memory try --enroll 张三 a.jpg --enroll 李四 c.jpg b.jpg d.jpg
```

```
enroll 张三: face at (371, 91, 241, 329) in a.jpg
enroll 李四: face at (427, 177, 287, 398) in c.jpg

b.jpg: 1 face(s)
  (153, 98, 216, 314)      known    张三         +0.827
  says: 我看到了张三。
```

The columns are the face box (x, y, width, height), the result, the closest
remembered name and the cosine similarity to it. Use photos of the same people
taken on different days to see where your thresholds should be.

## Try it with a camera

```bash
python -m face_memory live --config config.yaml
> enroll 张三
好的，张三，我记住你了。
> who
我看到了张三。
```

Commands are `enroll NAME`, `who`, `forget` and `quit`.

## Use from Python

```python
from face_memory import build_service, load_config

faces = build_service(load_config("src/face_memory/config.yaml"))
faces.enroll_face("张三")   # 好的，张三，我记住你了。
faces.who_is_here()          # 我看到了张三。
```

Both return one sentence to speak. Camera and model errors are raised, so the
caller decides what to say. One call runs at a time.

[voice_agent](../voice_agent/README.md#face-memory) registers these two calls
as tools when `face_memory.enabled` is true.

## How recognition decides

Each face becomes a unit-length embedding. A face is compared with every
remembered person by cosine similarity, and the closest person decides:

| Similarity to the closest person | Result | Spoken |
|---|---|---|
| `>= match_threshold` (0.5) | known | 我看到了张三。 |
| `>= unsure_threshold` (0.363), below `match_threshold` | unsure | 你是张三吗？ |
| below `unsure_threshold` | unknown | 我还不认识你，可以告诉我你的名字吗？ |

- Several faces are reported largest first. One name goes to at most one face.
  Only one unsure face is asked about at a time.
- Enrolling the same name again adds a sample, up to `max_samples_per_name`.
  This is what a 是 after 你是张三吗？ does.
- Enrolling a new name for a face that matched someone else moves that face to
  the new name, so 我不是张三，我叫李四 corrects the mistake.
- Faces smaller than `min_face_px` are ignored: they are too far away to
  recognise reliably.
- When the person who said the name is not the largest face in view, the wrong
  face is enrolled. Ask them to stand in front of the camera.

0.363 is the threshold published with SFace. 0.5 leaves a margin above it so a
spoken name is a confident one.

## Configure

[config.yaml](config.yaml) documents each option:

| Key | Default | Meaning |
|---|---|---|
| `source` | `0` | Camera index, `/dev/videoN`, an image file, or `ros2:<topic>` for a `sensor_msgs/Image` topic |
| `model_dir` | `~/.cache/face_memory/models` | Where the model files are kept |
| `match_threshold` | `0.5` | Similarity at which a face is recognised by name |
| `unsure_threshold` | `0.363` | Similarity at which the robot asks 你是X吗？ |
| `session_idle_min` | `30` | Minutes without use before everything is forgotten |
| `frames_per_enroll` | `3` | Frames averaged for one enrollment |
| `detect_confidence` | `0.8` | Minimum face detection score |
| `min_face_px` | `40` | Smallest face side, in pixels, that is used |
| `max_image_side` | `640` | Frames are scaled down to this for detection; embeddings use full resolution |
| `frame_timeout_sec` | `5.0` | Longest wait for a frame |
| `max_samples_per_name` | `5` | Samples kept per person |
| `phrases.*` | Chinese | Everything it says; `{name}`, `{names}` and `{count}` are filled in |

The camera is opened for each call and released afterwards, so another program
can use the same camera between calls. A `ros2:` source subscribes only while
it grabs frames, and needs a sourced ROS 2 environment with `rclpy` and
`sensor_msgs`. Supported encodings are `bgr8`, `rgb8`, `bgra8`, `rgba8` and `mono8`.

## Swap the face model

Anything with a `faces(frame)` method can replace the default:

```python
from face_memory import Face, FaceMemory

class MyEmbedder:
    def faces(self, frame):
        """Return a Face(box, score, embedding) per face; embeddings unit length."""

memory = FaceMemory(MyEmbedder(), match_threshold=..., unsure_threshold=...)
```

Thresholds depend on the model, so set them again with still images (`try`)
for any other model.

## Tests

```bash
cd src/face_memory
python -m pytest tests
```

Most tests use a fake embedder with fixed vectors. They cover the similarity
bands, averaging, sample limits, name correction, one name per face, session
expiry, every spoken sentence, config validation, image conversion and model
download checks. `test_still_images.py` runs the real models on public-domain
US government portraits. Each person is enrolled from one photo and must be
recognised in another taken years apart, and two other people must not be
recognised. The tests also cover a photo with two people, a low-resolution
photo and a blank image. The photos are downloaded on first run, pinned by
commit and SHA-256, and cached in `~/.cache/face_memory/test-images`; they are
not stored in this repository. These tests skip themselves when the models or
photos can't be downloaded.

## Layout

| File | Purpose |
|---|---|
| `face_memory/embedder.py` | `Face`, the `FaceEmbedder` interface, and `SFaceEmbedder` |
| `face_memory/models.py` | Model download and SHA-256 check |
| `face_memory/memory.py` | `FaceMemory`: enrollment, recognition, session expiry |
| `face_memory/service.py` | `enroll_face` / `who_is_here` and the spoken sentences |
| `face_memory/source.py` | Camera, image file and ROS 2 image sources |
| `face_memory/config.py` | Config dataclasses and validation |
| `face_memory/__main__.py` | `download-models`, `try` and `live` |
