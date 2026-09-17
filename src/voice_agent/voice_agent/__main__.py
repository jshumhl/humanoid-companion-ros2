"""Command-line entry point: python -m voice_agent --config config.yaml"""

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import ConfigError, load_config

log = logging.getLogger("voice_agent")

CHECKS = ("speaker", "mic", "camera", "llm", "asr")
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
        phrases = config.fallback_phrases
        speech.prepare([phrases.not_heard, phrases.offline, phrases.error, phrases.tool_failed])

    try:
        if args.text:
            text_loop(agent, speech)
        else:
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


def deliver(reply, speech):
    if reply.heard:
        print(f"你：{reply.heard}")
    tags = [f"tool={reply.tool}"] if reply.tool else []
    tags += [f"fallback={reply.fallback}"] if reply.fallback else []
    print(f"巴克：{reply.text}" + (f"   [{', '.join(tags)}]" if tags else ""), flush=True)
    if speech:
        speech.say(reply.text)


def text_loop(agent, speech):
    print("Text mode. Type a message, or q to quit.")
    while True:
        line = input("\n你：").strip()
        if line.lower() in ("q", "quit", "exit"):
            return
        if line:
            deliver(agent.respond_to_text(line), speech)


def voice_loop(agent, speech, config):
    from .audio import make_recorder, to_wav_bytes

    audio = config.audio
    try:
        recorder = make_recorder(audio)
    except RuntimeError as e:
        print(f"Audio input error: {e}", file=sys.stderr)
        return

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
        deliver(agent.respond_to_audio(to_wav_bytes(pcm, audio.sample_rate), audio.sample_rate), speech)


if __name__ == "__main__":
    sys.exit(main())
