# voice_agent

A voice agent for **巴克机器人** (Bake Robot). You press Enter, ask a question
in Mandarin, and the robot answers out loud. When you ask what it sees, it
looks through the camera.

```
microphone → push-to-talk → ASR → LLM (JSON reply, may call a tool) → TTS → speaker
                                              │
                                              ├─ look_around() → object_narrator.describe_scene()
                                              ├─ current_time(), today() → this machine's clock
                                              └─ enroll_face(name), who_is_here() → face_memory (optional)
```

The steps run one after another in a single thread, and each turn prints what
was heard and what was said.

- Runs on CPU, Python 3.10+, no ROS. It has a `COLCON_IGNORE` file, like `object_narrator`.
- Push-to-talk or always-on listening, and you can cut the robot off mid-sentence.
- The backend is chosen with `LLM_PROVIDER`. The first adapter is DashScope:
  `qwen-plus-character` for chat and `qwen-audio-3.0-asr-flash` for speech recognition.
- Tool calls use our own JSON reply format, parsed locally, so provider-specific
  tool-calling APIs aren't needed.
- Speech output uses `object_narrator`'s edge-tts `Speaker`, the provider's TTS,
  or the robot's own offline TTS service (`speech_output.engine`).
- The agent never speaks an error. Failures map to fixed, friendly phrases, and
  the technical detail goes to the terminal log.

For first-time setup on a new Ubuntu machine, see [SETUP.md](../../SETUP.md).

## Usage example

Run from the repository root, with the virtual environment active and `.env` filled in:

```bash
python -m voice_agent
```

```
Push-to-talk: Enter to start recording, Enter again to stop. q + Enter to quit.

[Enter] to talk:
Recording... press Enter to stop.

你：你是谁。
巴克：我是巴克机器人。我是一个陪伴你左右的人形机器人，很高兴认识你！

[Enter] to talk:
Recording... press Enter to stop.

你：你看见什么。
巴克：我看到了四个人和一辆公交车。   [tool=look_around]
```

For the second question, the model replies `{"tool": "look_around", "args": {}}`.
The agent runs the camera and detector and speaks the resulting sentence
unchanged. The model isn't called again to rephrase it.

Other ways to run it:

```bash
python -m voice_agent --text             # type instead of talking (no microphone needed)
python -m voice_agent --text --no-audio  # no microphone, no speaker: pure debugging
python -m voice_agent --check all        # test speaker, mic, camera, face memory, LLM, ASR, offline recognizer
python -m voice_agent --check llm        # just one component
python -m voice_agent --list-devices     # audio device indexes for config.yaml
python -m voice_agent -v                 # debug log, including raw model output
python -m voice_agent --config my.yaml   # another config file
```

## Configuration

**Secrets go in the environment, never in `config.yaml`.** The CLI reads a
`.env` file. It looks beside the config file and in its parents first, then in
the current directory and its parents, so a deployment's own `.env` wins.

```bash
# .env beside config.yaml or at the repository root (keep it out of git)
LLM_PROVIDER=dashscope
DASHSCOPE_API_KEY=sk-...
DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/api/v1   # optional, see SETUP.md
```

Everything else is in [config.yaml](config.yaml), which documents each
option. The main ones:

| Key | Default | Meaning |
|---|---|---|
| `narrator_config` | `narrator.yaml` | Shared camera `source`, `language`, edge-tts `voice` and `player` |
| `audio.input_device` | `null` | Microphone index or name (`--list-devices`) |
| `listening.mode` | `push_to_talk` | `always_on` listens continuously and allows spoken interruption (below) |
| `conversation.max_reply_sentences` | `3` | Sentences spoken before asking 还要继续吗 |
| `local_asr.model_path` | `~/.cache/voice_agent/vosk-model-small-cn-0.22` | Offline recognizer for menu answers (SETUP.md step 4) |
| `speech_output.engine` | `edge-tts` | `provider` tries the provider's TTS first, then falls back to edge-tts. `ros2` speaks through the robot's offline TTS service |
| `speech_output.ros2.service`, `.type` | empty | The robot's TTS service name and type (`pkg/srv/Name`); required for `ros2` |
| `speech_output.ros2.text_field` | `text` | Request field that carries the text; `request` sets any other fields, `max_chars` caps the length |
| `timeouts.*_sec` | 15–20 | Longest wait for ASR, LLM, TTS or a tool before using a fallback phrase |
| `clock.timezone` | empty | Time zone for `current_time()` and `today()`, e.g. `Asia/Shanghai`; empty uses this machine's |
| `conversation.max_history_turns` | `10` | Past exchanges sent with each request |
| `providers.dashscope.*` | see file | Model names |
| `gestures.catalogue_file` | `gestures.yaml` | Which gestures exist and when to use them (below) |
| `gestures.backend` | `stub` | `stub` logs only; `ros2` publishes to `ros2_topic` |
| `gestures.log_path` | `~/.cache/voice_agent/gesture-choices.log` | Per-turn record of the chosen gesture, for tuning |
| `status.backend` | `stub` | `stub` logs only; `ros2` publishes the current phase to `ros2_topic` (below) |
| `face_memory.enabled` | `false` | Adds the `enroll_face` and `who_is_here` tools (below) |
| `face_memory.config_file` | `face_memory.yaml` | Camera source, models and thresholds, in face_memory's format |
| `system_prompt` | persona | Must contain `{tools}` and `{gestures}`, replaced with the tool and gesture lists |
| `fallback_phrases.*` | Chinese phrases | `not_heard`, `offline`, `error`, `tool_failed` |
| `offline_menu.*` | 3 options | Spoken menu offered when the provider is unreachable (below) |

The config is checked when it loads. Errors name the key and the problem, for example:

```
Config error: audio.sample_rate must be one of (8000, 16000, 32000, 48000) when audio.vad.enabled is true
Config error: Unknown key(s) audio.sample_rat. Allowed: audio.input_device, audio.sample_rate, ...
```

## When things fail

| Situation | Robot says | Terminal shows |
|---|---|---|
| Recording too short, or empty transcript | 我没听清，可以再说一遍吗？ | – |
| Network down, DNS failure, timeout, 5xx | 我现在连不上网络，请稍后再试。 | `WARNING ... unreachable` |
| Auth, quota, bad model, malformed JSON reply | 抱歉，我刚才走神了，请再说一遍。 | `WARNING ... 401 InvalidApiKey` etc. |
| Camera or detector failure | 抱歉，我现在看不清周围。 | `WARNING Tool look_around failed: ...` |
| Machine clock never set (year before 2025) | 我现在不太确定准确的时间。 | – |
| Face memory cannot start (models missing, camera unknown) | – (its two tools are left out) | `Face memory unavailable, enroll_face and who_is_here are off: ...` |

### Offline menu

After an "offline" reply, the robot offers a short spoken menu instead of
leaving the conversation there:

```
你：你是谁？
巴克：我现在连不上网络。你可以说 一 检查设置，二 重试，三 退出。
你：二
巴克：好的，我再试一次。
```

The menu's opening line replaces the plain `offline` phrase, so the robot
doesn't say it can't connect twice in a row. With `offline_menu.enabled: false`
the plain phrase is spoken as before.

Answer with the keyword (设置 / 重试 / 退出) or the position (一 / 1 / 第一).
The actions are:

| Action | Effect |
|---|---|
| `settings` | Prints provider, endpoint, whether the key is set, and the last error in the terminal. Never spoken |
| `retry` | Asks the same question again, once. If that fails, the robot returns to idle rather than offering the menu again |
| `quit` | Leaves the conversation and returns to idle |

After `max_attempts` answers that aren't understood (2 by default), it speaks
`closing` and returns to idle. The menu is never offered twice in a row, so it
cannot loop.

Everything about it is in the `offline_menu` section of `config.yaml`: up to
three options, each with its own keyword and spoken reply. All of its audio is
cached at startup along with the fallback phrases, so it can always be spoken
with the network down.

#### Hearing the answer offline

Cloud speech recognition is down exactly when the menu is needed, so menu
answers are recognized **on this machine** with Vosk and a small Chinese model.
Cloud recognition is never attempted here: it would only wait for a timeout.

The grammar is built from the configured options, so the recognizer only
chooses between a few phrases plus `[unk]`:

```
设置  设 置  一  重试  重 试  二  两  退出  退 出  三  [unk]
```

Two details of the model shape it:

- Some words are missing from its vocabulary, 重试 among them, and are silently
  dropped from the grammar. Each keyword is therefore also offered split into
  characters (`重 试`), which the model does recognize.
- Unrelated speech comes back as `[unk]`, sometimes with a stray word attached,
  e.g. 今天天气怎么样 → `[unk] 一`. A keyword next to `[unk]` still counts as a
  choice, but a bare position does not, since one syllable next to
  unrecognized speech is too easily a mishearing.

Measured with recorded speech and no network:

| Said | Heard | Result |
|---|---|---|
| 设置 | `设置` | settings |
| 重试 | `重 试` | retry |
| 三 | `三` | quit |
| 我想重试一次 | `[unk] 两 重 试 一 置` | retry |
| 今天天气怎么样 | `[unk] 一` | not understood |

**Typing 1/2/3 always works** and is the fallback when the model is missing.
The menu prompt is `[Enter] to answer, or type 1/2/3:`; typing answers
directly, and pressing Enter alone records instead. If the model isn't
installed, the agent says so at startup rather than during an outage, and
`--text` mode needs no recognizer at all.

Set it up with SETUP.md step 4, and verify it with `--check local-asr`.
`local_asr.enabled: false` turns it off and silences the startup warning.

Each network call and tool call has a hard time limit, so the loop can't hang.
Fallback phrases are turned into audio at startup and cached in
`~/.cache/voice_agent/tts`. Once the agent has run online once, it can still
say them with the network down. Any other reply needs the network, because
edge-tts is an online service. With `speech_output.engine: ros2` every reply can
be spoken offline, since the robot synthesizes speech itself. If a reply can't
be spoken, it is still printed.

## Reply format and tools

The system prompt tells the model to output exactly one JSON object:

```json
{"say": "我是巴克机器人。"}
{"tool": "look_around", "args": {}}
```

[protocol.py](voice_agent/protocol.py) parses it. Code fences and extra text
around the object are tolerated. A plain-text reply with no JSON is spoken as
is, and anything else counts as a format error. Before speaking, `say` text
has Markdown and emoji removed and is cut to `max_reply_sentences`. Tool
results are spoken exactly as returned.

To add a tool, register it in [tools.py](voice_agent/tools.py):

```python
Tool(name="what_time", signature="what_time()",
     description="返回现在几点的中文句子。", run=lambda args: f"现在是{...}。")
```

The tool list in the system prompt is generated from the registry, so no prompt
edit is needed unless the model needs guidance on when to use the tool.

### Face memory

With `face_memory.enabled: true`, two more tools are registered, backed by
[face_memory](../face_memory/README.md):

| Tool | When the model calls it | Says, for example |
|---|---|---|
| `enroll_face(name)` | 我叫张三 / 这是张三, or 是 after 你是张三吗？ | 好的，张三，我记住你了。 |
| `who_is_here()` | 你认识我吗 / 你还记得我吗 / 我是谁 | 我看到了张三。 / 你是张三吗？ / 我还不认识你，可以告诉我你的名字吗？ |

The model passes the name as `{"tool": "enroll_face", "args": {"name": "张三"}}`.
A low-confidence match is spoken as a question; when the person answers 是,
the model calls `enroll_face` with that name, which adds a sample, and when
they give another name, that name replaces the old one for this face. Faces
and names are kept in memory only, for one session. The gesture tuning log
(`gestures.log_path`) does record each turn's words, so a spoken name ends up
there; set `log_path: ""` where that matters.

Install the module and fetch its models once, while online:

```bash
pip install -e src/face_memory
python -m face_memory --config src/voice_agent/face_memory.yaml download-models
python -m voice_agent --check face
```

If face_memory cannot start, the agent runs without these two tools and says so
in the terminal.

## Listening and interruption

```yaml
listening:
  mode: push_to_talk   # push_to_talk | always_on
  interrupt_ms: 300    # always_on: speech this long during playback interrupts
  playback_poll_ms: 20 # how often playback checks whether to stop
```

| Mode | Listening | Interrupting |
|---|---|---|
| `push_to_talk` (default) | Enter starts recording, Enter stops it | Enter while the robot speaks |
| `always_on` | VAD finds each utterance (`pip install webrtcvad-wheels`) | Enter, or speaking for `interrupt_ms` during playback |

**An interruption stops the audio, abandons the rest of the reply, and hands
the turn back.** The robot never resumes an interrupted sentence: by the time
it could, the person has moved on. Measured with the real player, playback
goes silent 16–23 ms after the interrupt; anything over 200 ms is logged as a
warning.

With `speech_output.engine: ros2`, the robot's TTS service plays each reply
itself and cannot be stopped once it starts. An interruption then takes effect
when the robot finishes what it is saying, at most `max_reply_sentences`
sentences; the rest of the turn is dropped as usual, and the logged stop
latency is the real time until the robot went quiet.

A reply is spoken as a single piece of audio rather than sentence by sentence.
Splitting it sounded wrong: each playback reopens the audio device, which added
0.2–1.0 s of silence at every 。 Interruption does not need the split, since
the player can be stopped mid-word. What is held back by the length guard is
synthesized while the first part plays, so 继续 starts speaking at once.

```
INFO voice_agent.delivery: Interrupted by enter after 12 characters; playback stopped in 18 ms
```

The watcher is armed for the whole turn, the model call included, so
interrupting while the robot is still thinking abandons the request instead of
waiting for an answer nobody wants. The HTTP call may still finish in its own
thread — a blocking socket read cannot be cancelled from outside — but its
result is discarded and never spoken.

Conversation history records roughly what was actually heard, estimated from
how long the audio played:

```json
{"say": "我是巴克机器人。", "interrupted": true, "note": "用户打断了这句话，后面的内容没有说完"}
```

Without it the model assumes its whole reply landed, and answers follow-up
questions about things the person never heard.

**Echo:** in `always_on` mode the microphone hears the robot's own speaker,
and there is no echo cancellation here. `interrupt_ms` is the defence: a
syllable of its own voice is ignored, sustained speech is not. On a robot
whose microphone hears its speaker clearly, raise `interrupt_ms` or use
push-to-talk.

### Length guard

A reply longer than `conversation.max_reply_sentences` (3) is not delivered as
a monologue. The robot speaks that many sentences, asks `continue_prompt`, and
waits:

```
你：请用五句话讲讲你自己
巴克：我是巴克机器人。我是一个人形陪伴机器人，专门来陪大家聊天的。我虽然不能像人类一样吃饭睡觉，但我可以一直陪在你身边。
巴克：还要继续吗？
你：继续
巴克：如果你好奇周围有什么，我还能用眼睛帮你看看。很高兴能成为你的朋友！
```

The rest is spoken only if the answer contains one of
`conversation.continue_words`; any other reply drops it and starts a new turn.

## Gestures

The model may attach at most one gesture to a reply:

```json
{"say": "你好，我是巴克机器人。", "gesture": "hello"}
{"tool": "look_around", "gesture": "point"}
```

The gesture is requested **when speech playback starts**, so the movement runs
with the voice instead of after it. If the sentence can't be synthesized, no
gesture is sent: a robot waving in silence is worse than one standing still.

Only catalogued names are accepted. A made-up name is logged and dropped, and
the reply is still spoken normally.

**While a gesture is playing, further requests are skipped, not queued.** By
the time a queued gesture ran, the sentence it belonged to would be over.

### The catalogue: gestures.yaml

Which gestures exist, and when to use them, live in
[gestures.yaml](gestures.yaml), not in code and not in `config.yaml`:

```yaml
gestures:
  - name: hello                                  # what the model writes
    use_when: 第一次见到人，或者有人跟你打招呼的时候   # the situation, not the pose
    duration: 2.0                                # requests during this are skipped

examples:
  - user: 你好！
    say: 你好，我是巴克机器人。
    gesture: hello
  - user: 现在几点？
    say: 现在是下午三点十分。
    gesture: null                                # the common case
```

`use_when` describes the *occasion*, because that is what the model is
choosing. "有人跟你打招呼的时候" guides the choice; "挥手" only describes the
movement and tells the model nothing about when it applies.

`config.yaml` just points at the file:

```yaml
gestures:
  enabled: true
  backend: stub                  # stub | ros2
  ros2_topic: /gesture/request
  catalogue_file: gestures.yaml  # relative to config.yaml
```

The file is validated on load: names unique, `use_when` non-empty, `duration`
positive, example gestures present in the list. Problems are reported as
config errors that name the entry, e.g.
`gestures.catalogue_file: .../gestures.yaml: gestures[1].duration must be a number greater than 0`.

### The generated prompt section

At startup the whole gesture section of the system prompt is built from that
file and inserted in place of `{gestures}`:

```
只能从下列动作中选择。
如果都不合适，gesture 填 null。
大多数回复不需要动作，只在自然的时候使用。

可用动作：
- hello：第一次见到人，或者有人跟你打招呼的时候
- goodbye：有人要离开、跟你说再见的时候
- point：指出你看到的东西，或者回答东西在哪里、往哪边走
- nod：表示同意、答应对方，或者听明白了

例子：
用户：你好！
你：{"say": "你好，我是巴克机器人。很高兴认识你！", "gesture": "hello"}
用户：现在几点？
你：{"say": "现在是下午三点十分。", "gesture": null}
用户：洗手间在哪边？
你：{"say": "洗手间在那边，走廊尽头就是。", "gesture": "point"}
```

Adding a gesture or rewording a `use_when` is therefore a YAML edit; no code
changes, and the robot side decides what a name means.

### Tuning log

Every turn is appended to a rotating log, one JSON object per line:

```json
{"time": "2026-09-21T18:15:03", "user": "你好！", "gesture": "hello", "reply": "你好呀，我是巴克机器人。…"}
{"time": "2026-09-21T18:15:05", "user": "现在几点？", "gesture": null, "reply": "现在是下午三点半。"}
```

Turns with no gesture are recorded too, as `null`. What usually needs tuning is
how often a gesture is chosen at all, and that can't be seen from the gestures
alone. Tool and fallback turns carry a `tool` or `fallback` field.

```yaml
gestures:
  log_path: ~/.cache/voice_agent/gesture-choices.log   # "" turns the log off
  log_max_bytes: 1000000
  log_backups: 3
```

Counting choices from a session:

```bash
jq -r '.gesture // "null"' ~/.cache/voice_agent/gesture-choices.log | sort | uniq -c
```

### Backends

| Backend | Behavior |
|---|---|
| `stub` (default) | Logs `would request gesture: hello` and prints it. Works with no robot and no ROS |
| `ros2` | Publishes the gesture name as `std_msgs/String` on `ros2_topic`, default `/gesture/request` |

`rclpy` is imported inside the ROS 2 backend's constructor, so nothing
ROS-related loads unless that backend is selected: **voice_agent runs normally
with no ROS installed.** If `ros2` is selected but ROS is unavailable, it logs
a warning and falls back to the stub rather than failing to start. Run the ROS 2
backend in the container from `docker-compose.yml` (Ubuntu 22.04, ROS 2 Humble),
and watch it with `ros2 topic echo /gesture/request`.

## Status

The agent reports what it is doing, so a face display or a status light can
follow along. There are four phases, and each is reported only when it changes:

| Phase | When |
|---|---|
| `idle` | Waiting for Enter (push-to-talk) or for typed input, and on exit |
| `listening` | Recording; in always-on mode, the whole time it waits for speech |
| `thinking` | From the end of the recording to the start of playback: ASR, model, tool |
| `speaking` | From the moment playback starts, the same moment a gesture starts |

A typical turn in always-on mode is `listening → thinking → speaking → listening`.
An interruption goes straight back to `listening`.

```yaml
status:
  enabled: true
  backend: stub                  # stub | ros2
  ros2_topic: /voice_agent/state
```

The `ros2` backend publishes the phase name as `std_msgs/String` with
transient-local durability (depth 1), so a node that starts after the agent
still receives the current phase. Like the gesture backend, it imports `rclpy`
only when selected and falls back to the stub if ROS is unavailable. Watch it
with:

```bash
ros2 topic echo --qos-durability transient_local /voice_agent/state
```

A failing status backend is logged and ignored. It never stops the robot from
speaking.

## Providers

```
voice_agent/providers/
├── __init__.py    create_provider(): LLM_PROVIDER → adapter
├── base.py        Provider interface + ProviderError / ProviderUnavailable / ProviderConfigError
└── dashscope.py   DashScope adapter (plain HTTPS via requests)
```

Vendor-specific code lives only in this directory. A test checks that no vendor
SDK is imported anywhere else. To add a provider:

1. Create `providers/<name>.py` with a `Provider` subclass. It implements
   `chat(messages, timeout_sec) -> str` and
   `transcribe(wav_bytes, sample_rate, language, timeout_sec) -> str`, and
   optionally `synthesize(text, language, timeout_sec) -> bytes` (MP3).
   Network failures raise `ProviderUnavailable`, and other failures raise `ProviderError`.
2. Add it to `REGISTRY` in `providers/__init__.py`.
3. Add a `providers.<name>` section to `config.yaml` for its model names.
4. Run with `LLM_PROVIDER=<name>`.

No code outside `providers/` changes.

### DashScope notes

- Chat: `POST /services/aigc/text-generation/generation`.
- ASR: `POST /services/aigc/multimodal-generation/generation`, sending the WAV
  as a base64 data URI with `parameters.format` and `parameters.sample_rate`.
  Regional and workspace endpoints nest the result differently, and
  `extract_transcript` handles both.
- `qwen-audio-3.0-tts-plus` (`speech_output.engine: provider`) currently
  returns `InvalidParameter: [cosyvoice:]Engine error [411]` for every voice
  tried. That's why the default engine is edge-tts. The adapter falls back to
  edge-tts automatically if provider TTS fails.
- `403 AccessDenied.Unpurchased` means the model isn't enabled for the key's
  workspace. Enable it in the Model Studio console.

## Tests

```bash
cd src/voice_agent
python -m pytest tests
```

The tests cover reply parsing and cleanup, tool dispatch, the agent's turn
logic with a scripted provider (including offline, hanging, bad-JSON and
camera-failure cases), the offline menu (keyword and number answers, failed
attempts, `[unk]` handling), the Vosk grammar built from the options, gestures
(catalogue validation, skipping while one plays, backend fallback, firing at
playback start), config validation, provider selection, ASR response parsing,
the speech cache, the robot TTS engine against a fake service, `.env` lookup
order, and VAD segmentation. They need no network, microphone, Vosk
model or ROS: the ROS 2 tests skip themselves when `rclpy` is missing, and the
recognizer itself is covered by `--check local-asr`. They need no network,
microphone or camera. Use `--check` for those.

## Layout

| File | Purpose |
|---|---|
| `voice_agent/__main__.py` | CLI, push-to-talk / VAD / text loops |
| `voice_agent/agent.py` | One turn: ASR → LLM → parse → tool → reply; history; fallbacks |
| `voice_agent/delivery.py` | Speaking a reply: stoppable playback, length guard, continuations |
| `voice_agent/interrupt.py` | Enter and sustained-speech interrupt sources |
| `voice_agent/protocol.py` | JSON reply schema parsing, spoken-text cleanup |
| `voice_agent/menu.py` | Offline menu: keyword/number matching and attempt limit |
| `voice_agent/local_asr.py` | Offline recognition of menu answers (Vosk, grammar from the options) |
| `voice_agent/tools.py` | Tool registry, `look_around`, `enroll_face` and `who_is_here` |
| `voice_agent/gestures.py` | Gesture catalogue and prompt section, skip-while-playing, stub and ROS 2 backends |
| `voice_agent/gesture_log.py` | Rotating per-turn record of chosen gestures |
| `gestures.yaml` | The catalogue itself: names, `use_when`, durations, few-shot examples |
| `narrator.yaml` | Camera source, language, edge-tts voice and player (object_narrator format) |
| `face_memory.yaml` | Face camera source, models and thresholds (face_memory format) |
| `voice_agent/audio.py` | Microphone capture (push-to-talk, webrtcvad), WAV encoding |
| `voice_agent/speech.py` | TTS with offline phrase cache, playback via `object_narrator` |
| `voice_agent/robot_tts.py` | Speech through the robot's TTS service (`speech_output.engine: ros2`) |
| `voice_agent/config.py` | Config dataclasses and validation |
| `voice_agent/checks.py` | `--check` component tests |
| `voice_agent/timeouts.py` | Hard time limit for blocking calls |
| `voice_agent/env.py` | `.env` loader |
| `voice_agent/providers/` | Backend adapters |
