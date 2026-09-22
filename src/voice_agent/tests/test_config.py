from pathlib import Path

import pytest
import yaml

from voice_agent.config import ConfigError, load_config

from .conftest import PACKAGE_DIR, SHIPPED_CONFIG

NARRATOR_CONFIG = str(PACKAGE_DIR.parent / "object_narrator" / "config.yaml")


def write_config(tmp_path, **overrides):
    raw = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    # Paths in the shipped config are relative to it, and the copy lives elsewhere.
    raw["narrator_config"] = NARRATOR_CONFIG
    raw["gestures"]["catalogue_file"] = str(PACKAGE_DIR / "gestures.yaml")
    for dotted, value in overrides.items():
        node = raw
        *parents, leaf = dotted.split("__")
        for key in parents:
            node = node.setdefault(key, {})
        if value is DELETE:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


DELETE = object()


def test_shipped_config_loads():
    config = load_config(SHIPPED_CONFIG)
    assert config.narrator.language == "zh-CN"
    assert config.listening.mode == "push_to_talk"
    assert config.speech_output.engine == "edge-tts"
    assert "{tools}" in config.system_prompt
    assert [o.action for o in config.offline_menu.options] == ["settings", "retry", "quit"]


def test_gesture_catalogue_is_loaded_from_its_own_file():
    config = load_config(SHIPPED_CONFIG)
    assert config.gestures.backend == "stub"
    assert config.gestures.catalogue_file == "gestures.yaml"
    assert config.gesture_catalogue.names() == ["hello", "goodbye", "point", "nod"]
    assert config.gesture_catalogue.get("hello").duration == 2.0


def test_broken_catalogue_file_is_reported_as_a_config_error(tmp_path):
    catalogue = tmp_path / "gestures.yaml"
    catalogue.write_text("gestures:\n  - name: hello\n    duration: 2.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="gestures.catalogue_file:.*use_when"):
        load_config(write_config(tmp_path, gestures__catalogue_file=str(catalogue)))


def test_inline_catalogue_is_rejected_with_a_pointer_to_the_file(tmp_path):
    with pytest.raises(ConfigError, match="Unknown key\\(s\\) gestures.catalogue\\b"):
        load_config(write_config(tmp_path, gestures__catalogue=[{"name": "hello"}]))


def test_gestures_can_be_disabled_without_a_prompt_placeholder(tmp_path):
    config = load_config(write_config(tmp_path, gestures__enabled=False,
                                      system_prompt="你是巴克机器人。{tools}"))
    assert config.gestures.enabled is False


def test_other_sample_rates_allowed_without_the_local_recognizer(tmp_path):
    config = load_config(write_config(tmp_path, local_asr__enabled=False,
                                      audio__sample_rate=48000))
    assert config.audio.sample_rate == 48000


def test_disabled_menu_skips_option_validation(tmp_path):
    config = load_config(write_config(tmp_path, offline_menu__enabled=False,
                                      offline_menu__options=[]))
    assert config.offline_menu.enabled is False


def test_narrator_path_is_relative_to_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # cwd must not matter
    assert load_config(SHIPPED_CONFIG).narrator.source == 0


def test_digit_string_input_device_becomes_index(tmp_path):
    assert load_config(write_config(tmp_path, audio__input_device="2")).audio.input_device == 2


def test_device_name_allowed(tmp_path):
    assert load_config(write_config(tmp_path, audio__input_device="pulse")).audio.input_device == "pulse"


@pytest.mark.parametrize("overrides, message", [
    ({"audio__sample_rat": 16000}, "Unknown key(s) audio.sample_rat"),
    ({"speaker": "x"}, "Unknown key(s) speaker"),
    ({"audio__sample_rate": "16k"}, "audio.sample_rate must be int"),
    ({"listening__mode": "shouting"}, "listening.mode must be one of push_to_talk, always_on"),
    ({"listening__interrupt_ms": 10}, "listening.interrupt_ms must be >= 30"),
    ({"listening__playback_poll_ms": 500}, "listening.playback_poll_ms must be between"),
    ({"conversation__continue_prompt": " "}, "conversation.continue_prompt must not be empty"),
    ({"conversation__continue_words": []}, "conversation.continue_words must be"),
    ({"conversation__continue_words": ["好", " "]}, "conversation.continue_words must be"),
    ({"conversation__continue_words": "好"}, "conversation.continue_words must be a list"),
    ({"audio__max_record_sec": 0}, "audio.max_record_sec must be > 0"),
    ({"listening__mode": "always_on", "audio__sample_rate": 22050},
     "when listening.mode is always_on"),
    ({"audio__vad__aggressiveness": 5}, "aggressiveness must be 0, 1, 2 or 3"),
    ({"audio__input_device": 1.5}, "audio.input_device must be int or str or null"),
    ({"speech_output__engine": "espeak"}, "speech_output.engine must be one of"),
    ({"timeouts__llm_sec": -1}, "timeouts.llm_sec must be > 0"),
    ({"system_prompt": DELETE}, "Missing required key: system_prompt"),
    ({"system_prompt": "你是巴克机器人。"}, "{tools} placeholder"),
    ({"fallback_phrases__offline": " "}, "fallback_phrases.offline must not be empty"),
    ({"providers": ["dashscope"]}, "providers must be dict"),
    ({"narrator_config": "/nonexistent/config.yaml"}, "narrator_config: file not found"),
    ({"gestures__backend": "servo"}, "gestures.backend must be one of stub, ros2"),
    ({"gestures__ros2_topic": " "}, "gestures.ros2_topic must not be empty"),
    ({"gestures__catalogue_file": " "}, "gestures.catalogue_file must not be empty"),
    ({"gestures__catalogue_file": "/nonexistent/gestures.yaml"},
     "gestures.catalogue_file: Gesture catalogue not found"),
    ({"gestures__log_max_bytes": 0}, "gestures.log_max_bytes must be > 0"),
    ({"gestures__log_backups": -1}, "gestures.log_backups must be >= 0"),
    ({"system_prompt": "你是巴克机器人。{tools}"}, "{gestures} placeholder"),
    ({"local_asr__model_path": " "}, "local_asr.model_path must not be empty"),
    ({"audio__sample_rate": 48000}, "audio.sample_rate must be 16000 when local_asr.enabled"),
    ({"offline_menu__max_attempts": 0}, "offline_menu.max_attempts must be >= 1"),
    ({"offline_menu__prompt": " "}, "offline_menu.prompt must not be empty"),
    ({"offline_menu__options": [
        {"action": "retry", "keyword": "重试", "reply": "好"},
        {"action": "quit", "keyword": "退出", "reply": "好"},
        {"action": "settings", "keyword": "设置", "reply": "好"},
        {"action": "retry", "keyword": "再试", "reply": "好"},
    ]}, "offline_menu.options must have 1 to 3 entries"),
    ({"offline_menu__options": []}, "offline_menu.options must have 1 to 3 entries"),
    ({"offline_menu__options": [{"action": "dance", "keyword": "跳舞", "reply": "好"}]},
     "options[0].action must be one of settings, retry, quit"),
    ({"offline_menu__options": [{"action": "retry", "keyword": " ", "reply": "好"}]},
     "options[0].keyword must not be empty"),
    ({"offline_menu__options": [{"action": "retry", "keyword": "重试"}]},
     "Missing required key: offline_menu.options[0].reply"),
    ({"offline_menu__options": [
        {"action": "retry", "keyword": "重试", "reply": "好"},
        {"action": "quit", "keyword": "重试", "reply": "好"},
    ]}, "keyword '重试' is used twice"),
    ({"offline_menu__options": "重试"}, "offline_menu.options must be a list"),
])
def test_invalid_config_reports_clear_error(tmp_path, overrides, message):
    with pytest.raises(ConfigError) as excinfo:
        load_config(write_config(tmp_path, **overrides))
    assert message in str(excinfo.value)


def test_invalid_narrator_config_is_reported(tmp_path):
    narrator = tmp_path / "narrator.yaml"
    narrator.write_text("source: 0\nlanguage: ja-JP\nconfidence: 0.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="narrator_config.*language must be one of"):
        load_config(write_config(tmp_path, narrator_config=str(narrator)))


def test_missing_file():
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(Path("/nonexistent/voice.yaml"))


def test_invalid_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("audio: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)
