"""Load and validate config.yaml.

Camera source, language, and edge-tts voice/player are not repeated here:
they come from the object_narrator config that `narrator_config` points to.
"""

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class VadConfig:
    enabled: bool = False
    aggressiveness: int = 2          # 0 (least) to 3 (most aggressive filtering)
    silence_ms: int = 800            # trailing silence that ends an utterance
    start_timeout_sec: float = 10.0  # give up if nobody speaks for this long


@dataclass
class AudioConfig:
    input_device: Union[int, str, None] = None  # sounddevice index or name; None = default
    sample_rate: int = 16000
    max_record_sec: float = 20.0
    min_record_sec: float = 0.3
    vad: VadConfig = field(default_factory=VadConfig)


@dataclass
class LocalAsrConfig:
    """Offline recognizer used only for offline-menu answers."""

    enabled: bool = True
    model_path: str = "~/.cache/voice_agent/vosk-model-small-cn-0.22"


@dataclass
class SpeechOutputConfig:
    engine: str = "edge-tts"         # edge-tts | provider
    cache_dir: str = "~/.cache/voice_agent/tts"


@dataclass
class TimeoutConfig:
    asr_sec: float = 15.0
    llm_sec: float = 20.0
    tts_sec: float = 20.0
    tool_sec: float = 20.0


@dataclass
class ConversationConfig:
    max_history_turns: int = 10
    max_reply_sentences: int = 3


@dataclass
class MenuOption:
    action: str      # settings | retry | quit
    keyword: str     # spoken word that selects it, e.g. 重试
    reply: str       # spoken when it is selected


@dataclass
class OfflineMenuConfig:
    """Spoken menu offered when the provider is unreachable."""

    enabled: bool = True
    prompt: str = "我现在连不上网络。你可以说 一 检查设置，二 重试，三 退出。"
    closing: str = "我先休息一下，需要的时候再叫我。"
    max_attempts: int = 2
    options: List[MenuOption] = field(default_factory=lambda: [
        MenuOption("settings", "设置", "好的，我把设置情况显示在屏幕上了。"),
        MenuOption("retry", "重试", "好的，我再试一次。"),
        MenuOption("quit", "退出", "好的，我先休息一下。"),
    ])


@dataclass
class FallbackPhrases:
    not_heard: str = "我没听清，可以再说一遍吗？"
    offline: str = "我现在连不上网络，请稍后再试。"
    error: str = "抱歉，我刚才走神了，请再说一遍。"
    tool_failed: str = "抱歉，我现在看不清周围。"


@dataclass
class Config:
    system_prompt: str
    narrator_config: str = "../object_narrator/config.yaml"
    audio: AudioConfig = field(default_factory=AudioConfig)
    local_asr: LocalAsrConfig = field(default_factory=LocalAsrConfig)
    speech_output: SpeechOutputConfig = field(default_factory=SpeechOutputConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    providers: dict = field(default_factory=dict)
    fallback_phrases: FallbackPhrases = field(default_factory=FallbackPhrases)
    offline_menu: OfflineMenuConfig = field(default_factory=OfflineMenuConfig)

    # Filled in by load_config, not read from YAML.
    narrator: object = field(default=None, repr=False)


SPEECH_ENGINES = ("edge-tts", "provider")
VAD_SAMPLE_RATES = (8000, 16000, 32000, 48000)
MENU_ACTIONS = ("settings", "retry", "quit")
LOCAL_ASR_SAMPLE_RATE = 16000
MAX_MENU_OPTIONS = 3  # a spoken menu longer than this is hard to remember


def load_config(path):
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise ConfigError(f"Config file not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    if "narrator" in raw:
        raise ConfigError("narrator: set `narrator_config` to a file path instead")

    config = _build(Config, raw, "")
    validate(config)
    config.narrator = _load_narrator(path.parent, config.narrator_config)
    return config


def validate(config):
    audio, vad = config.audio, config.audio.vad
    if isinstance(audio.input_device, str) and audio.input_device.strip().isdigit():
        audio.input_device = int(audio.input_device)
    _require(audio.sample_rate > 0, "audio.sample_rate must be > 0")
    _require(audio.max_record_sec > 0, "audio.max_record_sec must be > 0")
    _require(0 <= audio.min_record_sec < audio.max_record_sec,
             "audio.min_record_sec must be >= 0 and less than audio.max_record_sec")
    _require(0 <= vad.aggressiveness <= 3, "audio.vad.aggressiveness must be 0, 1, 2 or 3")
    _require(vad.silence_ms >= 100, "audio.vad.silence_ms must be >= 100")
    _require(vad.start_timeout_sec > 0, "audio.vad.start_timeout_sec must be > 0")
    if vad.enabled:
        _require(audio.sample_rate in VAD_SAMPLE_RATES,
                 f"audio.sample_rate must be one of {VAD_SAMPLE_RATES} when audio.vad.enabled is true")

    if config.local_asr.enabled:
        _require(config.local_asr.model_path.strip() != "",
                 "local_asr.model_path must not be empty when local_asr.enabled is true")
        _require(audio.sample_rate == LOCAL_ASR_SAMPLE_RATE,
                 f"audio.sample_rate must be {LOCAL_ASR_SAMPLE_RATE} when local_asr.enabled is "
                 f"true (the Vosk model expects it), got {audio.sample_rate}")

    _require(config.speech_output.engine in SPEECH_ENGINES,
             f"speech_output.engine must be one of {', '.join(SPEECH_ENGINES)}, "
             f"got {config.speech_output.engine!r}")

    for f in dataclasses.fields(TimeoutConfig):
        _require(getattr(config.timeouts, f.name) > 0, f"timeouts.{f.name} must be > 0")

    _require(config.conversation.max_history_turns >= 0, "conversation.max_history_turns must be >= 0")
    _require(config.conversation.max_reply_sentences >= 1, "conversation.max_reply_sentences must be >= 1")

    _require(config.system_prompt.strip() != "", "system_prompt must not be empty")
    _require("{tools}" in config.system_prompt,
             "system_prompt must contain the {tools} placeholder, where the tool list is inserted")

    for f in dataclasses.fields(FallbackPhrases):
        _require(getattr(config.fallback_phrases, f.name).strip() != "",
                 f"fallback_phrases.{f.name} must not be empty")

    for name, settings in config.providers.items():
        _require(isinstance(settings, dict) or settings is None,
                 f"providers.{name} must be a mapping")

    _validate_menu(config.offline_menu)


def _validate_menu(menu):
    if not menu.enabled:
        return
    _require(menu.prompt.strip() != "", "offline_menu.prompt must not be empty")
    _require(menu.closing.strip() != "", "offline_menu.closing must not be empty")
    _require(menu.max_attempts >= 1, "offline_menu.max_attempts must be >= 1")
    _require(1 <= len(menu.options) <= MAX_MENU_OPTIONS,
             f"offline_menu.options must have 1 to {MAX_MENU_OPTIONS} entries, "
             f"got {len(menu.options)}")

    keywords = set()
    for i, option in enumerate(menu.options):
        where = f"offline_menu.options[{i}]"
        _require(option.action in MENU_ACTIONS,
                 f"{where}.action must be one of {', '.join(MENU_ACTIONS)}, got {option.action!r}")
        _require(option.keyword.strip() != "", f"{where}.keyword must not be empty")
        _require(option.reply.strip() != "", f"{where}.reply must not be empty")
        keyword = option.keyword.strip()
        _require(keyword not in keywords, f"{where}.keyword {keyword!r} is used twice")
        keywords.add(keyword)


def _load_narrator(base_dir, narrator_path):
    from object_narrator.config import load_config as load_narrator_config

    path = Path(narrator_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise ConfigError(f"narrator_config: file not found: {path}")
    try:
        return load_narrator_config(path)
    except (ValueError, TypeError) as e:
        raise ConfigError(f"narrator_config ({path}): {e}") from e


def _require(condition, message):
    if not condition:
        raise ConfigError(message)


def _build(cls, raw, prefix):
    """Construct dataclass `cls` from a dict, checking keys and value types."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{prefix.rstrip('.') or 'config'} must be a mapping")

    hints = typing.get_type_hints(cls)
    yaml_fields = {f.name: f for f in dataclasses.fields(cls) if f.name != "narrator"}
    unknown = set(raw) - set(yaml_fields)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {', '.join(prefix + k for k in sorted(unknown))}. "
            f"Allowed: {', '.join(prefix + k for k in yaml_fields)}"
        )

    values = {}
    for name, f in yaml_fields.items():
        key = prefix + name
        if name not in raw:
            required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
            if required:
                raise ConfigError(f"Missing required key: {key}")
            continue
        expected = hints[name]
        value = raw[name]
        if dataclasses.is_dataclass(expected):
            values[name] = _build(expected, value if value is not None else {}, key + ".")
        elif _dataclass_list_item(expected):
            item_type = _dataclass_list_item(expected)
            if not isinstance(value, list):
                raise ConfigError(f"{key} must be a list")
            values[name] = [_build(item_type, item, f"{key}[{i}].")
                            for i, item in enumerate(value)]
        else:
            values[name] = _check_type(value, expected, key)
    return cls(**values)


def _dataclass_list_item(expected):
    """For list[SomeDataclass], return SomeDataclass; otherwise None."""
    if typing.get_origin(expected) is not list:
        return None
    args = typing.get_args(expected)
    return args[0] if args and dataclasses.is_dataclass(args[0]) else None


def _check_type(value, expected, key):
    origin = typing.get_origin(expected)
    if origin in (Union, types.UnionType):
        options = typing.get_args(expected)
        for option in options:
            try:
                return _check_type(value, option, key)
            except ConfigError:
                pass
        names = " or ".join("null" if o is type(None) else o.__name__ for o in options)
        raise ConfigError(f"{key} must be {names}, got {value!r}")

    if expected is type(None):
        if value is None:
            return None
    elif expected is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif expected is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif expected is dict:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
    elif isinstance(value, expected):
        return value
    name = "null" if expected is type(None) else expected.__name__
    raise ConfigError(f"{key} must be {name}, got {value!r}")
