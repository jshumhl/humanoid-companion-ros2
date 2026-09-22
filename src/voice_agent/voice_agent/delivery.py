"""Speaking a reply: stoppable, with a length guard.

A reply is spoken as one piece of audio, not sentence by sentence. Each
playback opens the audio device again, which was clearly audible as a pause at
every 。 Interruption does not need the split: the player is terminated
mid-word whenever the watcher fires.

Sentences still matter for the length guard: only the first N are spoken, and
the rest are held back behind 还要继续吗.

The interrupt watcher is armed for the whole turn, the model call included: a
person who interrupts while the robot is still thinking should not have to
wait for an answer they no longer want.
"""

import logging
from dataclasses import dataclass, field

from .protocol import split_sentences
from .speech import SpeakResult
from .timeouts import CallCancelled

log = logging.getLogger(__name__)

# Interruption should silence the robot within this long; anything slower is
# worth knowing about.
STOP_BUDGET_MS = 200

# Rough Mandarin TTS rate, used only to estimate how much of an interrupted
# reply the person actually heard. The history note says it is approximate.
CHARS_PER_SEC = 5.0


def heard_prefix(text, result):
    """Estimate the part of `text` that was heard before playback was stopped."""
    heard = int(result.played_sec * CHARS_PER_SEC)
    return text[:max(0, min(heard, len(text)))]


@dataclass
class Outcome:
    reply: object = None
    interrupted: bool = False
    interrupted_by: str = ""
    stop_latency_ms: float = None
    paused: bool = False            # stopped after N sentences, waiting for an answer
    spoken: str = ""

    @property
    def delivered(self):
        return not self.interrupted


@dataclass
class _Pending:
    sentences: list = field(default_factory=list)


class ReplyDelivery:
    """Speaks replies, handling interruption, the length guard and continuations."""

    def __init__(self, agent, speech, gestures, config, watcher, show):
        self._agent = agent
        self._speech = speech
        self._gestures = gestures
        self._config = config
        self._watcher = watcher
        self._show = show               # prints the reply for the operator
        self._pending = None

    # --- turns ---

    def respond_and_speak(self, respond):
        """Run one turn: `respond(cancel_event)` produces the reply, then speak it."""
        self.discard_pending()
        self._watcher.arm()
        try:
            try:
                reply = respond(self._watcher.event)
            except CallCancelled:
                return self._interrupted_while_thinking()
            if self._watcher.event.is_set() and not reply.text:
                return self._interrupted_while_thinking()

            self._show(reply)
            return self._speak(split_sentences(reply.text), reply)
        finally:
            self._watcher.disarm()

    def speak_plain(self, reply):
        """Speak a reply that is not a model turn, e.g. a fallback phrase."""
        self._show(reply)
        if self._speech:
            self._speech.say(reply.text)

    # --- continuations ---

    @property
    def has_pending(self):
        return self._pending is not None

    def wants_continue(self, text):
        """True if this utterance answers 还要继续吗 with yes."""
        if not self.has_pending:
            return False
        cleaned = (text or "").strip().strip("。！？!?, ，")
        return any(word and word in cleaned
                   for word in self._config.conversation.continue_words)

    def resume(self):
        """Speak the sentences held back by the length guard."""
        pending, self._pending = self._pending, None
        if pending is None:
            return Outcome()
        self._show_line("".join(pending.sentences))
        self._watcher.arm()
        try:
            return self._speak(pending.sentences, None)
        finally:
            self._watcher.disarm()

    def discard_pending(self):
        if self._pending is not None:
            log.info("Dropping %d sentences the person did not ask to hear",
                     len(self._pending.sentences))
            self._pending = None

    # --- internals ---

    def _speak(self, sentences, reply):
        limit = self._config.conversation.max_reply_sentences
        head, rest = sentences[:limit], sentences[limit:]
        gesture = getattr(reply, "gesture", "") if reply else ""

        # Spoken as one piece, not sentence by sentence: every separate
        # playback reopens the audio device, which was audible as a pause at
        # each 。 Interruption still stops it mid-word.
        spoken_text = "".join(head)
        prompt = self._config.conversation.continue_prompt
        # Prepared while the reply plays, so a continuation starts immediately.
        self._prefetch([prompt, "".join(rest)] if rest else [])

        result = self._say(spoken_text, gesture)
        if result.interrupted:
            return self._stopped(result, heard_prefix(spoken_text, result), reply)

        if not rest:
            return Outcome(reply=reply, spoken=spoken_text)

        # Long reply: offer the rest instead of talking over the person's patience.
        self._pending = _Pending(rest)
        self._show_line(prompt)
        result = self._say(prompt, "")
        if result.interrupted:
            self._pending = None
            return self._stopped(result, spoken_text, reply)
        return Outcome(reply=reply, paused=True, spoken=spoken_text)

    def _prefetch(self, sentences):
        if self._speech is not None:
            for sentence in sentences:
                self._speech.prefetch(sentence)

    def _say(self, sentence, gesture):
        if self._speech is None:
            return SpeakResult(spoken=True)   # --no-audio: nothing to stop
        on_start = None
        if gesture and self._gestures is not None:
            on_start = lambda: self._gestures.request(gesture)  # noqa: E731
        return self._speech.say(sentence, on_playback_start=on_start,
                                interrupt=self._watcher.event)

    def _stopped(self, result, spoken, reply):
        latency = self._latency_ms(result.stopped_at)
        self._log_interruption(latency, spoken)
        if reply is not None or spoken:
            # The model must know the person only heard this much.
            self._agent.mark_last_reply_interrupted(spoken)
        self._pending = None
        return Outcome(reply=reply, interrupted=True, interrupted_by=self._watcher.source,
                       stop_latency_ms=latency, spoken=spoken)

    def _interrupted_while_thinking(self):
        self._log_interruption(None, "")
        return Outcome(interrupted=True, interrupted_by=self._watcher.source)

    def _latency_ms(self, stopped_at):
        started = self._watcher.detected_at
        if started is None or stopped_at is None:
            return None
        return (stopped_at - started) * 1000

    def _log_interruption(self, latency_ms, spoken):
        source = self._watcher.source or "unknown"
        if latency_ms is None:
            log.info("Interrupted by %s while preparing the reply; nothing was spoken", source)
            return
        message = "Interrupted by %s after %d characters; playback stopped in %.0f ms"
        if latency_ms > STOP_BUDGET_MS:
            log.warning(message + " (over the %d ms budget)", source, len(spoken),
                        latency_ms, STOP_BUDGET_MS)
        else:
            log.info(message, source, len(spoken), latency_ms)

    def _show_line(self, text):
        print(f"巴克：{text}", flush=True)
