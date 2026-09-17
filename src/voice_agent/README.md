# voice_agent

A voice agent for **巴克机器人** (Bake Robot). You press Enter, ask a question
in Mandarin, and the robot answers out loud. When you ask what it sees, it
looks through the camera.

```
microphone → push-to-talk → ASR → LLM (JSON reply, may call a tool) → TTS → speaker
                                              │
                                              └─ look_around() → object_narrator.describe_scene()
```

The steps run one after another in a single thread, and each turn prints what
was heard and what was said.

- Runs on CPU, Python 3.10+, no ROS. It has a `COLCON_IGNORE` file, like `object_narrator`.
- The backend is chosen with `LLM_PROVIDER`. The first adapter is DashScope:
  `qwen-plus-character` for chat and `qwen-audio-3.0-asr-flash` for speech recognition.
- Tool calls use our own JSON reply format, parsed locally, so provider-specific
  tool-calling APIs aren't needed.
- Speech output uses `object_narrator`'s edge-tts `Speaker`.
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
python -m voice_agent --check all        # test speaker, mic, camera, LLM and ASR one by one
python -m voice_agent --check llm        # just one component
python -m voice_agent --list-devices     # audio device indexes for config.yaml
python -m voice_agent -v                 # debug log, including raw model output
python -m voice_agent --config my.yaml   # another config file
```

## Configuration

**Secrets go in the environment, never in `config.yaml`.** The CLI reads a
`.env` file. It looks in the current directory and its parents first, then in
the config file's directory and its parents.

```bash
# .env at the repository root (already in .gitignore)
LLM_PROVIDER=dashscope
DASHSCOPE_API_KEY=sk-...
DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/api/v1   # optional, see SETUP.md
```

Everything else is in [config.yaml](config.yaml), which documents each
option. The main ones:

| Key | Default | Meaning |
|---|---|---|
| `narrator_config` | `../object_narrator/config.yaml` | Shared camera `source`, `language`, edge-tts `voice` and `player` |
| `audio.input_device` | `null` | Microphone index or name (`--list-devices`) |
| `audio.vad.enabled` | `false` | Voice activity detection instead of push-to-talk (`pip install webrtcvad-wheels`) |
| `speech_output.engine` | `edge-tts` | `provider` tries the provider's TTS first, then falls back to edge-tts |
| `timeouts.*_sec` | 15–20 | Longest wait for ASR, LLM, TTS or a tool before using a fallback phrase |
| `conversation.max_history_turns` | `10` | Past exchanges sent with each request |
| `providers.dashscope.*` | see file | Model names |
| `system_prompt` | persona | Must contain `{tools}`, which is replaced with the tool list |
| `fallback_phrases.*` | Chinese phrases | `not_heard`, `offline`, `error`, `tool_failed` |

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

Each network call and tool call has a hard time limit, so the loop can't hang.
Fallback phrases are turned into audio at startup and cached in
`~/.cache/voice_agent/tts`. Once the agent has run online once, it can still
say them with the network down. Any other reply needs the network, because
edge-tts is an online service. If a reply can't be spoken, it is still printed.

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
camera-failure cases), config validation, provider selection, ASR response
parsing, the speech cache, and VAD segmentation. They need no network,
microphone or camera. Use `--check` for those.

## Layout

| File | Purpose |
|---|---|
| `voice_agent/__main__.py` | CLI, push-to-talk / VAD / text loops |
| `voice_agent/agent.py` | One turn: ASR → LLM → parse → tool → reply; history; fallbacks |
| `voice_agent/protocol.py` | JSON reply schema parsing, spoken-text cleanup |
| `voice_agent/tools.py` | Tool registry and `look_around` |
| `voice_agent/audio.py` | Microphone capture (push-to-talk, webrtcvad), WAV encoding |
| `voice_agent/speech.py` | TTS with offline phrase cache, playback via `object_narrator` |
| `voice_agent/config.py` | Config dataclasses and validation |
| `voice_agent/checks.py` | `--check` component tests |
| `voice_agent/timeouts.py` | Hard time limit for blocking calls |
| `voice_agent/env.py` | `.env` loader |
| `voice_agent/providers/` | Backend adapters |
