"""Command-line entry point: python -m voice_agent --config config.yaml"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from .agent import Reply
from .config import ConfigError, load_config

log = logging.getLogger("voice_agent")

CHECKS = ("speaker", "mic", "camera", "llm", "asr", "local-asr")
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="voice_agent",
        description="巴克机器人 voice agent: push-to-talk → ASR → LLM (with tools) → TTS.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.yaml")
    parser.add_argument("--text", action="store_true",
                        help="type instead of speaking into the microphone")
    parser.add_argument("--no-audio", action="store_true",
                        help="print replies without speaking them")
    parser.add_argument("--check", choices=[*CHECKS, "all"],
                        help="test one component and exit")
    parser.add_argument("--list-devices", action="store_true",
                        help="list audio devices and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Third-party debug logs drown out ours.
    for noisy in ("urllib3", "asyncio", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    # .env is looked up from the working directory, then from the config
    # file's directory, so the repo's .env is found wherever the agent is started.
    from .env import find_dotenv, load_dotenv
    env_path = find_dotenv() or find_dotenv(Path(args.config).resolve().parent)
    if env_path:
        load_dotenv(env_path)
        log.debug("Loaded environment from %s", env_path)

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    if args.check:
        from .checks import run_checks
        names = CHECKS if args.check == "all" else [args.check]
        return run_checks(names, config)

    from .providers import ProviderConfigError, create_provider
    try:
        provider = create_provider(config.providers)
    except ProviderConfigError as e:
        print(f"Provider error: {e}", file=sys.stderr)
        return 2

    from .agent import Agent
    from .tools import default_tools
    agent = Agent(provider, default_tools(config.narrator), config)

    speech = None
    if not args.no_audio:
        try:
            speech = make_speech_output(config, provider)
        except RuntimeError as e:
            print(f"Audio output error: {e}", file=sys.stderr)
            return 2
        # Cached now, while the network is up, so they can be spoken when it is not.
        from .menu import menu_phrases
        phrases = config.fallback_phrases
        speech.prepare([phrases.not_heard, phrases.offline, phrases.error, phrases.tool_failed,
                        *menu_phrases(config.offline_menu)])

    try:
        if args.text:
            text_loop(agent, speech, config)
        else:
            warn_if_local_recognizer_missing(config)
            voice_loop(agent, speech, config)
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


def make_speech_output(config, provider):
    from object_narrator.speaker import Speaker

    from .speech import SpeechOutput

    narrator = config.narrator
    speaker = Speaker(narrator.language, narrator.voice, narrator.player)
    return SpeechOutput(
        speaker,
        cache_dir=config.speech_output.cache_dir,
        timeout_sec=config.timeouts.tts_sec,
        provider=provider if config.speech_output.engine == "provider" else None,
        language=narrator.language,
    )


def deliver(reply, speech, say_reply=True):
    """Show and speak a reply. With say_reply=False only the transcript is shown,
    which is used when the offline menu is about to say the same thing."""
    if reply.heard:
        print(f"你：{reply.heard}")
    if not say_reply:
        return
    tags = [f"tool={reply.tool}"] if reply.tool else []
    tags += [f"fallback={reply.fallback}"] if reply.fallback else []
    print(f"巴克：{reply.text}" + (f"   [{', '.join(tags)}]" if tags else ""), flush=True)
    if speech:
        speech.say(reply.text)


def lazy_local_recognizer(config):
    """Return a callable that loads the Vosk model on first use, then reuses it.

    Loading takes a moment and most sessions never go offline, so it is not
    done at startup. The model path is checked at startup instead, by
    warn_if_local_recognizer_missing().
    """
    from .local_asr import load_menu_recognizer

    holder = {}

    def get():
        if "recognizer" not in holder:
            holder["recognizer"] = load_menu_recognizer(
                config.local_asr, config.offline_menu.options, config.audio.sample_rate)
        return holder["recognizer"]

    return get


def warn_if_local_recognizer_missing(config):
    """Say at startup, not mid-outage, that offline menu answers cannot be heard."""
    local = config.local_asr
    if not local.enabled or not config.offline_menu.enabled:
        return
    if not Path(local.model_path).expanduser().is_dir():
        print(f"Warning: Vosk model not found at {local.model_path}. The offline menu will "
              f"accept typed answers only. See SETUP.md, 'Offline menu recognizer'.",
              file=sys.stderr)


def menu_follows(reply, config):
    """True when the offline menu will be offered for this reply."""
    return reply.fallback == "offline" and config.offline_menu.enabled


def offer_offline_menu(agent, speech, config, listen, retry_text=None):
    """Offer the offline menu after an "offline" reply. Returns the action taken."""
    from .menu import IDLE, OfflineMenu

    menu = config.offline_menu
    if not menu.enabled:
        return IDLE

    action = OfflineMenu(menu, lambda text: deliver(Reply(text), speech), listen).run()
    if action == "settings":
        print_settings(agent, config)
    elif action == "retry" and retry_text:
        # One retry only: a failing retry returns to idle instead of
        # offering the menu again, so the robot cannot loop.
        deliver(agent.respond_to_text(retry_text), speech)
    return action


def print_settings(agent, config):
    """Operator-facing diagnostics. Never spoken."""
    from .providers import selected_provider_name

    print("  provider:            ", selected_provider_name())
    print("  DASHSCOPE_BASE_URL:  ", os.environ.get("DASHSCOPE_BASE_URL", "(default)"))
    print("  DASHSCOPE_API_KEY:   ", "set" if os.environ.get("DASHSCOPE_API_KEY") else "NOT SET")
    print("  config:              ", config.narrator_config)
    print("  last error:          ", agent.last_error or "(none)")
    print("  Try: python -m voice_agent --check llm", flush=True)


def text_loop(agent, speech, config):
    print("Text mode. Type a message, or q to quit.")
    while True:
        line = input("\n你：").strip()
        if line.lower() in ("q", "quit", "exit"):
            return
        if not line:
            continue
        reply = agent.respond_to_text(line)
        # The menu opens with its own "cannot connect" line, so the plain
        # offline phrase would just repeat it.
        deliver(reply, speech, say_reply=not menu_follows(reply, config))
        if reply.fallback == "offline":
            action = offer_offline_menu(
                agent, speech, config,
                listen=lambda: input("\n你：").strip(),
                retry_text=line,
            )
            if action == "quit":
                return


def voice_loop(agent, speech, config):
    from .audio import make_recorder, to_wav_bytes

    audio = config.audio
    try:
        recorder = make_recorder(audio)
    except RuntimeError as e:
        print(f"Audio input error: {e}", file=sys.stderr)
        return

    local_recognizer = lazy_local_recognizer(config)

    if audio.vad.enabled:
        print("Listening. Speak any time; Ctrl+C to quit.")
    else:
        print("Push-to-talk: Enter to start recording, Enter again to stop. q + Enter to quit.")

    while True:
        if audio.vad.enabled:
            print("\n(listening...)", flush=True)
        else:
            if input("\n[Enter] to talk: ").strip().lower() in ("q", "quit", "exit"):
                return
            print("Recording... press Enter to stop.", flush=True)

        try:
            pcm = recorder.record()
        except Exception as e:
            log.error("Microphone error: %s (check audio.input_device; see --list-devices)", e)
            deliver(agent.fallback("error"), speech)
            time.sleep(1)
            continue

        if audio.vad.enabled and len(pcm) == 0:
            continue  # nobody spoke; keep listening quietly
        if len(pcm) < audio.min_record_sec * audio.sample_rate:
            deliver(agent.fallback("not_heard"), speech)
            continue

        reply = agent.respond_to_audio(to_wav_bytes(pcm, audio.sample_rate), audio.sample_rate)
        deliver(reply, speech, say_reply=not menu_follows(reply, config))
        if reply.fallback == "offline":
            action = offer_offline_menu(
                agent, speech, config,
                listen=lambda: listen_once(recorder, audio, local_recognizer()),
                retry_text=reply.heard,
            )
            if action == "quit":
                return


def listen_once(recorder, audio, local_recognizer):
    """Get one menu answer: typed, or spoken and recognized on this machine.

    The menu only runs when the provider is unreachable, so cloud recognition
    is never attempted here: it would just wait for a timeout. Typing 1/2/3
    always works, and is the fallback when the local recognizer is missing.
    """
    typed = input("[Enter] to answer, or type 1/2/3: ").strip()
    if typed:
        return typed
    if local_recognizer is None:
        print("No offline recognizer, so type 1, 2 or 3 (see SETUP.md).", flush=True)
        return ""

    print("Recording... press Enter to stop.", flush=True)
    pcm = recorder.record()
    if len(pcm) < audio.min_record_sec * audio.sample_rate:
        return ""
    return local_recognizer.recognize(pcm)


if __name__ == "__main__":
    sys.exit(main())
