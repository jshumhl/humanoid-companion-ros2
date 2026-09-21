from pathlib import Path

import pytest
import yaml

from voice_agent.config import ConfigError, load_config

from .conftest import PACKAGE_DIR, SHIPPED_CONFIG

NARRATOR_CONFIG = str(PACKAGE_DIR.parent / "object_narrator" / "config.yaml")


def write_config(tmp_path, **overrides):
    raw = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    raw["narrator_config"] = NARRATOR_CONFIG
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
    assert config.audio.vad.enabled is False
    assert config.speech_output.engine == "edge-tts"
    assert "{tools}" in config.system_prompt
    assert [o.action for o in config.offline_menu.options] == ["settings", "retry", "quit"]


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
    ({"audio__vad__enabled": "yes"}, "audio.vad.enabled must be bool"),
    ({"audio__max_record_sec": 0}, "audio.max_record_sec must be > 0"),
    ({"audio__vad__enabled": True, "audio__sample_rate": 22050}, "when audio.vad.enabled is true"),
    ({"audio__vad__aggressiveness": 5}, "aggressiveness must be 0, 1, 2 or 3"),
    ({"audio__input_device": 1.5}, "audio.input_device must be int or str or null"),
    ({"speech_output__engine": "espeak"}, "speech_output.engine must be one of"),
    ({"timeouts__llm_sec": -1}, "timeouts.llm_sec must be > 0"),
    ({"system_prompt": DELETE}, "Missing required key: system_prompt"),
    ({"system_prompt": "你是巴克机器人。"}, "{tools} placeholder"),
    ({"fallback_phrases__offline": " "}, "fallback_phrases.offline must not be empty"),
    ({"providers": ["dashscope"]}, "providers must be dict"),
    ({"narrator_config": "/nonexistent/config.yaml"}, "narrator_config: file not found"),
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
