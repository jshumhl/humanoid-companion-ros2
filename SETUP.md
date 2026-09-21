# First-time setup (Ubuntu 22.04)

This guide takes a fresh Ubuntu 22.04 machine to a working 巴克机器人 voice
agent: you press Enter, ask 你是谁？ or 你看见什么？, and the robot answers out
loud. You test the speaker, microphone, camera and cloud connection one at a
time before starting the agent, so a problem is easy to trace to one part.

You need:

- Ubuntu 22.04 with internet access. A desktop or the robot's onboard computer both work.
- A USB or built-in microphone, a speaker, and a camera. On the BXI ELF3, the
  RealSense D435i's color stream appears as an ordinary V4L2 camera.
- A DashScope (Alibaba Cloud Model Studio) API key, with these models enabled
  for its workspace: `qwen-plus-character` and `qwen-audio-3.0-asr-flash`.

No GPU is needed. All commands below run from the repository root unless noted.

---

## 1. System packages

```bash
sudo apt update
sudo apt install -y \
    git python3.10 python3.10-venv python3-pip \
    libportaudio2 alsa-utils pulseaudio-utils mpg123 \
    v4l-utils ffmpeg libgl1 libglib2.0-0
```

What each group is for:

| Packages | Used for |
|---|---|
| `python3.10`, `python3.10-venv`, `python3-pip` | Python and virtual environments |
| `libportaudio2` | Microphone access from Python (`sounddevice`) |
| `alsa-utils`, `pulseaudio-utils` | `arecord`, `aplay`, `speaker-test`, `pactl` for testing audio |
| `mpg123` | Plays the robot's speech (MP3) |
| `v4l-utils`, `ffmpeg` | `v4l2-ctl` and `ffplay` for testing the camera |
| `libgl1`, `libglib2.0-0` | Libraries OpenCV needs |

Give your user access to audio and video devices, then **log out and back in**
so the change takes effect:

```bash
sudo usermod -aG audio,video "$USER"
```

## 2. Get the code and create a Python environment

```bash
git clone <repository-url> humanoid-companion-ros2
cd humanoid-companion-ros2

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Run `source .venv/bin/activate` again in every new terminal before using the agent.

Install the CPU-only build of PyTorch first. It is much smaller than the default
build, which includes GPU libraries:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the two modules from this repository in editable mode:

```bash
pip install -e src/object_narrator -e src/voice_agent
```

Optional: to use hands-free voice activity detection instead of push-to-talk
later, also run `pip install webrtcvad-wheels`.

Check the install:

```bash
python -m voice_agent --help
```

## 3. Environment variables (API key)

The API key is a secret. It goes in a `.env` file at the repository root. It
never goes in `config.yaml`, and never into git.

```bash
cat > .env <<'EOF'
LLM_PROVIDER=dashscope
DASHSCOPE_API_KEY=sk-your-key-here
DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/api/v1
EOF
chmod 600 .env
git check-ignore .env    # must print ".env"; if it prints nothing, stop and add .env to .gitignore
```

| Variable | Required | Value |
|---|---|---|
| `LLM_PROVIDER` | yes | `dashscope` (the only provider so far) |
| `DASHSCOPE_API_KEY` | yes | Your key from the Model Studio console |
| `DASHSCOPE_BASE_URL` | no | Which DashScope endpoint to use (below). Default: international |

Pick the base URL that matches where your key was created:

| Account / key | `DASHSCOPE_BASE_URL` |
|---|---|
| International (Singapore) | `https://dashscope-intl.aliyuncs.com/api/v1` |
| Mainland China (Beijing) | `https://dashscope.aliyuncs.com/api/v1` |
| Workspace endpoint shown in your console | `https://<workspace-id>.<region>.maas.aliyuncs.com/api/v1` |

You can also `export` these variables in your shell instead of using `.env`.
Variables already set in the shell take priority over `.env`.

## 4. Offline menu recognizer (Vosk model)

When the network drops, the robot offers a spoken menu (一 检查设置, 二 重试,
三 退出). Cloud speech recognition is down at exactly that moment, so menu
answers are recognized **on this machine** with a small Vosk model. Download it
once:

```bash
mkdir -p ~/.cache/voice_agent
cd ~/.cache/voice_agent
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip vosk-model-small-cn-0.22.zip && rm vosk-model-small-cn-0.22.zip
cd -
```

That leaves `~/.cache/voice_agent/vosk-model-small-cn-0.22`, about 65 MB on
disk, which matches `local_asr.model_path` in
[src/voice_agent/config.yaml](src/voice_agent/config.yaml). Put it elsewhere if
you prefer and change that setting to match. The `vosk` Python package is
already installed from step 2.

This model is used *only* for menu answers, never for conversation. If you skip
this step the agent still runs and the menu is still spoken; menu answers then
have to be typed (1, 2 or 3), and a warning is printed at startup.

## 5. Choose audio devices

### Microphone

List what Python can see:

```bash
python -m voice_agent --list-devices
```

Example output. `>` marks the default input and `<` the default output:

```
   0 HDA Intel PCH: ALC257 Analog (hw:0,0), ALSA (2 in, 2 out)
   1 USB PnP Sound Device: Audio (hw:1,0), ALSA (1 in, 0 out)
   ...
   5 pulse, ALSA (32 in, 32 out)
>  6 default, ALSA (32 in, 32 out)
```

Set the microphone in [src/voice_agent/config.yaml](src/voice_agent/config.yaml):

```yaml
audio:
  input_device: null      # system default
  # input_device: pulse   # recommended if the default gives trouble
  # input_device: 1       # a specific device by index
```

**Prefer `pulse` or `default` over raw `hw:` devices.** Speech recognition
records at 16 kHz. Many USB microphones only support 44.1 or 48 kHz directly
and fail with `Invalid sample rate`. The `pulse` and `default` devices convert
the sample rate automatically.

To make a particular microphone the system default under PulseAudio:

```bash
pactl list short sources                  # find its name
pactl set-default-source <source-name>
```

### Speaker

Speech is played with `mpg123` on the system default output. To choose a
different speaker:

```bash
pactl list short sinks
pactl set-default-sink <sink-name>
```

### Camera

```bash
v4l2-ctl --list-devices
```

A RealSense D435i shows several `/dev/video*` nodes, and usually only one of
them is the color stream. Set the camera in
[src/object_narrator/config.yaml](src/object_narrator/config.yaml). The voice
agent reads its camera and language settings from this file:

```yaml
source: 0              # or /dev/video4, etc.
language: zh-CN
confidence: 0.5
```

## 6. Verify each component

Do these in order. Each step tests one thing, so a failure points to a single
cause. Every `--check` prints PASS or FAIL.

### 6.1 Speaker

```bash
speaker-test -t wav -c 2 -l 1        # you should hear "Front Left", "Front Right"
python -m voice_agent --check speaker
```

The second command speaks 你好，我是巴克机器人。 through edge-tts and asks
whether you heard it.

- **No sound from `speaker-test`:** check the volume with `alsamixer` and the
  default output with `pactl set-default-sink`.
- **`speaker-test` works but the check fails with "No MP3 player found":**
  install `mpg123`.
- **The check fails with a connection error:** edge-tts needs internet access.

### 6.2 Microphone

```bash
arecord -f S16_LE -r 16000 -c 1 -d 3 /tmp/mic.wav && aplay /tmp/mic.wav
python -m voice_agent --check mic
```

Speak while it records. The check prints the recording level, plays it back,
and asks whether you heard yourself.

- **Level below about -60 dBFS:** wrong device, muted input, or gain too low.
  Open `alsamixer`, press F4 for capture controls, and raise the level.
- **`Invalid sample rate`:** set `audio.input_device: pulse`.
- **`Error querying device`:** the `input_device` index no longer exists.
  Indexes change when USB devices are plugged in, so run `--list-devices` again.

### 6.3 Camera

```bash
ffplay -f v4l2 /dev/video0           # a window with live video; press q to close
python -m voice_agent --check camera
```

The check captures one frame and prints a sentence such as
我看到了一个人和一把椅子。 The first run downloads the detection model
`yolo26n.pt` (about 5 MB) into the current directory.

- **`Cannot open camera`:** wrong `source`, the camera is in use by another
  program, or you are not in the `video` group (log out and back in after step 1).
- **Always 我没有看到任何东西。:** point the camera at a person, or lower `confidence`.

### 6.4 Language model

```bash
python -m voice_agent --check llm
```

Expected output:

```
Raw chat reply: '好'
你是谁？ → 我是巴克机器人。…   [tool=-, fallback=-]
你看见什么？ → （摄像头检查已跳过）   [tool=look_around, fallback=-]
PASS
```

This check doesn't use the camera. It only confirms that the model chose the
`look_around` tool.

| Error | Meaning |
|---|---|
| `DASHSCOPE_API_KEY is not set` | `.env` missing, or not in the repository root |
| `401 InvalidApiKey` | Wrong key, or a key from a different region than `DASHSCOPE_BASE_URL` |
| `403 AccessDenied.Unpurchased` | The model isn't enabled for this workspace; enable it in the Model Studio console |
| `unreachable ... NameResolutionError` | No internet, or a typo in `DASHSCOPE_BASE_URL` |

### 6.5 Speech recognition

```bash
python -m voice_agent --check asr
```

Say 你看见什么？ during the 4-second recording. It should print a transcript
such as `'你看见什么。'`. If the transcript is empty, fix the microphone first (6.2).

### 6.6 Offline menu recognizer

```bash
python -m voice_agent --check local-asr
```

This uses no network. It loads the Vosk model from step 4, prints the grammar
built from your menu options, records for 4 seconds and shows which option your
answer selected:

```
Model: ~/.cache/voice_agent/vosk-model-small-cn-0.22
Grammar: 设置  设 置  一  重试  重 试  二  两  退出  退 出  三  [unk]
Recording 4 s. Say one of: 设置、重试、退出 (or 一/二/三)...
Heard: '重 试'
Selected: retry
```

- **`Vosk model not found`:** the path in `local_asr.model_path` doesn't match
  where you unzipped the model in step 4.
- **`Heard: '[unk]'`:** speak one word clearly, close to the microphone. Extra
  words around the keyword are fine; background chatter is not.
- **You can skip this** if you are happy to type 1/2/3 when offline. Set
  `local_asr.enabled: false` to stop the startup warning.

### All at once

After each part passes on its own, this command reruns every check and prints a summary:

```bash
python -m voice_agent --check all
```

## 7. Run the agent

```bash
source .venv/bin/activate
python -m voice_agent
```

1. Press **Enter** and say 你是谁？
2. Press **Enter** again to stop recording. The robot answers 我是巴克机器人。…
3. Press **Enter** and say 你看见什么？, then **Enter**. The robot looks
   through the camera and says, for example, 我看到了两个人和一把椅子。
4. Type **q** and press Enter to quit.

The first 你看见什么？ takes a few seconds longer, because the detection model loads then.

Without a microphone or speaker, you can still try the conversation by typing:

```bash
python -m voice_agent --text --no-audio
```

Add `-v` to any command to see the raw model output and debug logs.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Robot offers the menu but can't hear the answer | Run `--check local-asr`. Type 1, 2 or 3 in the meantime |
| Warning at startup: `Vosk model not found` | Do step 4, or set `local_asr.enabled: false` |
| Robot always says 我现在连不上网络 | Check internet access and `DASHSCOPE_BASE_URL`, then run `--check llm` to see the real error |
| Robot always says 我没听清 | Run `--check mic` and `--check asr`. Hold Enter-to-Enter for at least half a second |
| Robot says 抱歉，我刚才走神了 | Run with `-v`. Usually an API error (401/403) or a malformed model reply |
| Robot says 抱歉，我现在看不清周围 | Run `--check camera` |
| `Config error: ...` at startup | The message names the key. Compare with the comments in `config.yaml` |
| `ModuleNotFoundError: object_narrator` | Activate the virtual environment, or rerun `pip install -e src/object_narrator -e src/voice_agent` |
| `ImportError: libGL.so.1` | `sudo apt install -y libgl1 libglib2.0-0` |
| `PortAudio library not found` | `sudo apt install -y libportaudio2` |

More detail on configuration, the provider interface, and adding tools:
[src/voice_agent/README.md](src/voice_agent/README.md).
