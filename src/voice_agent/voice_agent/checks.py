"""`--check`: test each component on its own, with technical detail on failure.

Unlike the agent, checks show real error messages; they are for whoever is
setting up the robot, not for the person talking to it.
"""

import time

TEST_PHRASE = "你好，我是巴克机器人。"
MIC_SECONDS = 3
ASR_SECONDS = 4


def run_checks(names, config):
    results = {}
    for name in names:
        print(f"\n=== {name} ===")
        try:
            ok = CHECK_FUNCTIONS[name](config)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            ok = False
        results[name] = ok
        print("PASS" if ok else "FAIL")

    if len(names) > 1:
        print("\n=== summary ===")
        for name, ok in results.items():
            print(f"{name:8} {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


def check_speaker(config):
    from object_narrator.speaker import Speaker

    narrator = config.narrator
    speaker = Speaker(narrator.language, narrator.voice, narrator.player)
    print(f"Speaking {TEST_PHRASE!r} with edge-tts voice {speaker.voice}...")
    speaker.speak(TEST_PHRASE)
    return _ask("Did you hear it?")


def check_mic(config):
    import sounddevice as sd

    from .audio import level_dbfs

    audio = config.audio
    device = sd.query_devices(audio.input_device, "input")
    print(f"Input device: {device['name']} (config audio.input_device={audio.input_device!r})")
    print(f"Recording {MIC_SECONDS} s at {audio.sample_rate} Hz. Speak now...")
    pcm = sd.rec(MIC_SECONDS * audio.sample_rate, samplerate=audio.sample_rate,
                 channels=1, dtype="int16", device=audio.input_device)
    sd.wait()
    level = level_dbfs(pcm[:, 0])
    print(f"Level: {level:.1f} dBFS")
    if level < -60:
        print("That is near silence: wrong device, muted input, or gain too low.")
        return False
    print("Playing the recording back on the default output device...")
    sd.play(pcm, audio.sample_rate)
    sd.wait()
    return _ask("Did you hear your voice?")


def check_camera(config):
    from object_narrator import describe_scene

    narrator = config.narrator
    print(f"Source: {narrator.source!r}, model: {narrator.model}")
    started = time.monotonic()
    sentence = describe_scene(narrator)
    print(f"{sentence}   ({time.monotonic() - started:.1f} s, includes model load)")
    return True


def check_llm(config):
    from .agent import Agent
    from .providers import create_provider, selected_provider_name
    from .tools import ToolRegistry, look_around_tool

    provider = create_provider(config.providers)
    print(f"Provider: {selected_provider_name()}")
    started = time.monotonic()
    raw = provider.chat([{"role": "user", "content": "请只回复：好"}], config.timeouts.llm_sec)
    print(f"Raw chat reply: {raw!r} ({time.monotonic() - started:.1f} s)")

    # Exercise the persona and tool routing without touching the camera.
    tools = ToolRegistry([look_around_tool(config.narrator, describe_scene=lambda _: "（摄像头检查已跳过）")])
    agent = Agent(provider, tools, config)
    ok = True
    for question, expect_tool in (("你是谁？", False), ("你看见什么？", True)):
        reply = agent.respond_to_text(question)
        routed = bool(reply.tool) == expect_tool
        ok = ok and routed and not reply.fallback
        print(f"{question} → {reply.text}   [tool={reply.tool or '-'}, fallback={reply.fallback or '-'}]"
              f"{'' if routed else '  ← unexpected tool routing'}")
    return ok


def check_asr(config):
    import sounddevice as sd

    from .audio import to_wav_bytes
    from .providers import create_provider

    provider = create_provider(config.providers)
    audio = config.audio
    print(f"Recording {ASR_SECONDS} s. Say something in Chinese now, e.g. 你看见什么？")
    pcm = sd.rec(ASR_SECONDS * audio.sample_rate, samplerate=audio.sample_rate,
                 channels=1, dtype="int16", device=audio.input_device)
    sd.wait()
    started = time.monotonic()
    text = provider.transcribe(to_wav_bytes(pcm[:, 0], audio.sample_rate), audio.sample_rate,
                               config.narrator.language, config.timeouts.asr_sec)
    print(f"Transcript: {text!r} ({time.monotonic() - started:.1f} s)")
    return bool(text)


def _ask(question):
    answer = input(f"{question} [y/n] ").strip().lower()
    return answer in ("y", "yes", "是", "")


CHECK_FUNCTIONS = {
    "speaker": check_speaker,
    "mic": check_mic,
    "camera": check_camera,
    "llm": check_llm,
    "asr": check_asr,
}
