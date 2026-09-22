"""Delivering replies: interruption, cancellation and the length guard.

No audio hardware: speech, the watcher and the player are all fakes.
"""

import logging
import threading

from voice_agent.agent import Reply
from voice_agent.delivery import ReplyDelivery
from voice_agent.speech import SpeakResult
from voice_agent.timeouts import CallCancelled

FIVE_SENTENCES = "第一句。第二句。第三句。第四句。第五句。"


class FakeWatcher:
    """Stands in for the interrupt watcher; tests set the event themselves."""

    def __init__(self):
        self.event = threading.Event()
        self.source = ""
        self.detected_at = None
        self.armed_count = 0
        self.disarmed_count = 0

    def arm(self):
        self.event.clear()
        self.armed_count += 1

    def disarm(self):
        self.disarmed_count += 1

    def interrupt(self, source="enter", at=100.0):
        self.source = source
        self.detected_at = at
        self.event.set()


class FakeSpeech:
    """Speaks sentences, optionally interrupting at one of them."""

    def __init__(self, watcher=None, interrupt_at=None, stop_after=0.05):
        self.spoken = []
        self.gestures = []
        self._watcher = watcher
        self._interrupt_at = interrupt_at
        self._stop_after = stop_after

    def say(self, text, on_playback_start=None, interrupt=None):
        if interrupt is not None and interrupt.is_set():
            return SpeakResult(spoken=False, interrupted=True, stopped_at=100.0)
        if on_playback_start is not None:
            on_playback_start()
            self.gestures.append(text)
        if self._interrupt_at is not None and len(self.spoken) == self._interrupt_at:
            self._watcher.interrupt(at=100.0)
            return SpeakResult(spoken=False, interrupted=True,
                               stopped_at=100.0 + self._stop_after)
        self.spoken.append(text)
        return SpeakResult(spoken=True)


class FakeAgent:
    def __init__(self):
        self.interrupted_with = None

    def mark_last_reply_interrupted(self, spoken):
        self.interrupted_with = spoken


class FakeGestures:
    def __init__(self):
        self.requested = []

    def request(self, name):
        self.requested.append(name)
        return True


def make_delivery(config, speech, watcher=None, gestures=None, agent=None):
    watcher = watcher or FakeWatcher()
    delivery = ReplyDelivery(agent or FakeAgent(), speech, gestures, config, watcher,
                             show=lambda reply: None)
    return delivery, watcher


def test_short_reply_is_spoken_in_full(config, ):
    speech = FakeSpeech()
    delivery, watcher = make_delivery(config, speech)

    outcome = delivery.respond_and_speak(lambda cancel: Reply("你好。很高兴见到你。"))

    assert speech.spoken == ["你好。", "很高兴见到你。"]
    assert outcome.delivered and not outcome.paused
    assert watcher.armed_count == 1 and watcher.disarmed_count == 1


def test_gesture_starts_with_the_first_sentence(config):
    speech, gestures = FakeSpeech(), FakeGestures()
    delivery, _ = make_delivery(config, speech, gestures=gestures)

    delivery.respond_and_speak(lambda cancel: Reply("你好。第二句。", gesture="hello"))

    assert gestures.requested == ["hello"]
    assert speech.gestures == ["你好。"]  # not repeated on later sentences


# --- interruption ---

def test_playback_stops_and_the_rest_is_discarded(config, caplog):
    watcher = FakeWatcher()
    speech = FakeSpeech(watcher, interrupt_at=1)  # interrupt during the second sentence
    agent = FakeAgent()
    delivery, _ = make_delivery(config, speech, watcher, agent=agent)

    with caplog.at_level(logging.INFO):
        outcome = delivery.respond_and_speak(lambda cancel: Reply(FIVE_SENTENCES))

    assert speech.spoken == ["第一句。"]          # nothing after the interruption
    assert outcome.interrupted and outcome.interrupted_by == "enter"
    assert not delivery.has_pending               # the robot never resumes
    assert agent.interrupted_with == "第一句。"    # history records what was heard
    assert "stopped in 50 ms" in caplog.text


def test_slow_stop_is_logged_as_a_warning(config, caplog):
    watcher = FakeWatcher()
    speech = FakeSpeech(watcher, interrupt_at=0, stop_after=0.4)  # 400 ms, over budget
    delivery, _ = make_delivery(config, speech, watcher)

    with caplog.at_level(logging.WARNING):
        delivery.respond_and_speak(lambda cancel: Reply(FIVE_SENTENCES))

    assert "over the 200 ms budget" in caplog.text


def test_interrupt_while_the_model_is_still_thinking(config, caplog):
    speech = FakeSpeech()
    agent = FakeAgent()
    delivery, watcher = make_delivery(config, speech, agent=agent)

    def respond(cancel):
        watcher.interrupt(source="voice")
        raise CallCancelled("abandoned")

    with caplog.at_level(logging.INFO):
        outcome = delivery.respond_and_speak(respond)

    assert outcome.interrupted and outcome.interrupted_by == "voice"
    assert speech.spoken == []                 # nothing was spoken
    assert agent.interrupted_with is None      # nothing was heard, nothing to mark
    assert "nothing was spoken" in caplog.text


def test_interrupt_event_is_cleared_for_the_next_turn(config):
    watcher = FakeWatcher()
    speech = FakeSpeech(watcher, interrupt_at=0)
    delivery, _ = make_delivery(config, speech, watcher)
    delivery.respond_and_speak(lambda cancel: Reply("第一句。"))

    speech = FakeSpeech()
    delivery, _ = make_delivery(config, speech, watcher)
    delivery.respond_and_speak(lambda cancel: Reply("新的回答。"))

    assert speech.spoken == ["新的回答。"]


# --- length guard ---

def test_long_reply_stops_after_n_sentences_and_asks(config):
    config.conversation.max_reply_sentences = 3
    speech = FakeSpeech()
    delivery, _ = make_delivery(config, speech)

    outcome = delivery.respond_and_speak(lambda cancel: Reply(FIVE_SENTENCES))

    assert speech.spoken == ["第一句。", "第二句。", "第三句。", config.conversation.continue_prompt]
    assert outcome.paused and delivery.has_pending


def test_continuing_speaks_the_rest(config):
    config.conversation.max_reply_sentences = 3
    speech = FakeSpeech()
    delivery, _ = make_delivery(config, speech)
    delivery.respond_and_speak(lambda cancel: Reply(FIVE_SENTENCES))
    speech.spoken.clear()

    assert delivery.wants_continue("好，继续") is True
    delivery.resume()

    assert speech.spoken == ["第四句。", "第五句。"]
    assert not delivery.has_pending


def test_a_new_question_drops_the_rest(config):
    config.conversation.max_reply_sentences = 3
    speech = FakeSpeech()
    delivery, _ = make_delivery(config, speech)
    delivery.respond_and_speak(lambda cancel: Reply(FIVE_SENTENCES))
    speech.spoken.clear()

    assert delivery.wants_continue("不用了，你看见什么？") is False
    delivery.respond_and_speak(lambda cancel: Reply("我看到了一个人。"))

    assert speech.spoken == ["我看到了一个人。"]
    assert not delivery.has_pending


def test_wants_continue_is_false_without_anything_pending(config):
    delivery, _ = make_delivery(config, FakeSpeech())
    assert delivery.wants_continue("继续") is False


def test_a_reply_of_exactly_n_sentences_does_not_ask(config):
    config.conversation.max_reply_sentences = 3
    speech = FakeSpeech()
    delivery, _ = make_delivery(config, speech)

    outcome = delivery.respond_and_speak(lambda cancel: Reply("第一句。第二句。第三句。"))

    assert speech.spoken == ["第一句。", "第二句。", "第三句。"]
    assert not outcome.paused and not delivery.has_pending


def test_interrupting_the_continue_prompt_drops_the_rest(config):
    config.conversation.max_reply_sentences = 1
    watcher = FakeWatcher()
    speech = FakeSpeech(watcher, interrupt_at=1)   # interrupt on the prompt itself
    delivery, _ = make_delivery(config, speech, watcher)

    outcome = delivery.respond_and_speak(lambda cancel: Reply(FIVE_SENTENCES))

    assert outcome.interrupted
    assert not delivery.has_pending


def test_works_without_audio_output(config):
    """--no-audio still applies the length guard, with nothing to interrupt."""
    config.conversation.max_reply_sentences = 2
    delivery, _ = make_delivery(config, None)

    outcome = delivery.respond_and_speak(lambda cancel: Reply(FIVE_SENTENCES))

    assert outcome.paused and delivery.has_pending
