"""One conversation: audio or text in, one spoken reply out.

Every failure is mapped to a fixed fallback phrase. Technical details go to
the log, never into what the robot says.
"""

import json
import logging
from dataclasses import dataclass

from .protocol import ReplyFormatError, ToolCall, clean_spoken, parse_model_output
from .providers import ProviderUnavailable
from .timeouts import CallTimeout, call_with_timeout
from .tools import UnknownTool

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reply:
    text: str
    heard: str = ""          # transcript of the user's speech, if any
    tool: str = ""           # tool that produced the reply, if any
    fallback: str = ""       # fallback phrase key used, if any
    gesture: str = ""        # catalogued gesture to play as speech starts


class Agent:
    def __init__(self, provider, tools, config, gestures=None, gesture_log=None):
        self._provider = provider
        self._tools = tools
        self._config = config
        self._gestures = gestures
        self._gesture_log = gesture_log
        self._phrases = config.fallback_phrases
        self._timeouts = config.timeouts
        self._language = config.narrator.language
        section = gestures.prompt_section() if gestures else "（暂无可用动作，gesture 一律填 null。）"
        self._system_prompt = (config.system_prompt
                               .replace("{tools}", tools.catalog())
                               .replace("{gestures}", section))
        self._history = []
        self.last_error = ""  # most recent failure, for --check and the offline menu

    @property
    def history(self):
        return list(self._history)

    def reset(self):
        self._history.clear()

    def transcribe(self, wav_bytes, sample_rate):
        """Speech to text. Raises ProviderUnavailable/CallTimeout when offline."""
        heard = call_with_timeout(
            self._provider.transcribe, self._timeouts.asr_sec,
            wav_bytes, sample_rate, self._language, self._timeouts.asr_sec,
        )
        return (heard or "").strip()

    def respond_to_audio(self, wav_bytes, sample_rate):
        try:
            heard = self.transcribe(wav_bytes, sample_rate)
        except (ProviderUnavailable, CallTimeout) as e:
            self._note_error("ASR unavailable", e)
            return self.fallback("offline")
        except Exception as e:
            self._note_error("ASR failed", e)
            return self.fallback("not_heard")

        if not heard:
            return self.fallback("not_heard")
        reply = self.respond_to_text(heard)
        return Reply(reply.text, heard=heard, tool=reply.tool, fallback=reply.fallback,
                     gesture=reply.gesture)

    def respond_to_text(self, text):
        reply = self._respond_to_text(text)
        # Every turn is recorded, gesture or not: what needs tuning is how
        # often a gesture is chosen at all, and for which utterances.
        if self._gesture_log is not None and text.strip():
            self._gesture_log.record(text.strip(), reply.gesture, reply.text,
                                     reply.tool, reply.fallback)
        return reply

    def _respond_to_text(self, text):
        text = text.strip()
        if not text:
            return self.fallback("not_heard")

        messages = [{"role": "system", "content": self._system_prompt},
                    *self._history,
                    {"role": "user", "content": text}]
        try:
            raw = call_with_timeout(self._provider.chat, self._timeouts.llm_sec,
                                    messages, self._timeouts.llm_sec)
        except (ProviderUnavailable, CallTimeout) as e:
            self._note_error("LLM unavailable", e)
            return self.fallback("offline")
        except Exception as e:
            self._note_error("LLM failed", e)
            return self.fallback("error")

        log.debug("model output: %s", raw)
        try:
            action = parse_model_output(raw)
        except ReplyFormatError as e:
            log.warning("Model reply did not match the JSON schema (%s): %r", e, raw)
            return self.fallback("error")

        if isinstance(action, ToolCall):
            return self._run_tool(text, action)

        spoken = clean_spoken(action.text, self._config.conversation.max_reply_sentences)
        if not spoken:
            log.warning("Model reply was empty after cleanup: %r", raw)
            return self.fallback("error")
        self._remember(text, spoken)
        return Reply(spoken, gesture=self._known_gesture(action.gesture))

    def _known_gesture(self, name):
        """Keep the gesture only if it is in the catalogue. A made-up name is dropped."""
        if not name:
            return ""
        if self._gestures is None or name not in self._gestures:
            log.warning("Ignoring gesture %r: not in the catalogue", name)
            return ""
        return name

    def _run_tool(self, user_text, call):
        try:
            result = call_with_timeout(self._tools.dispatch, self._timeouts.tool_sec, call)
        except UnknownTool:
            log.warning("Model asked for unknown tool %r", call.name)
            return self.fallback("error")
        except Exception as e:
            log.warning("Tool %s failed: %s", call.name, e)
            return self.fallback("tool_failed")

        result = (result or "").strip()
        if not result:
            return self.fallback("tool_failed")
        # Spoken as-is: a second model call to rephrase would add latency and
        # could change the facts the tool reported.
        self._remember(user_text, result)
        return Reply(result, tool=call.name, gesture=self._known_gesture(call.gesture))

    def _remember(self, user_text, spoken):
        # The assistant turn is stored in the reply schema, so the model keeps
        # seeing examples of the format it must produce.
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant",
                              "content": json.dumps({"say": spoken}, ensure_ascii=False)})
        max_messages = 2 * self._config.conversation.max_history_turns
        if len(self._history) > max_messages:
            del self._history[:len(self._history) - max_messages]

    def _note_error(self, what, error):
        self.last_error = f"{what}: {error}"
        log.warning("%s: %s", what, error)

    def fallback(self, key):
        """Reply with a fixed phrase: not_heard, offline, error or tool_failed."""
        return Reply(getattr(self._phrases, key), fallback=key)
