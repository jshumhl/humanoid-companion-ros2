"""Spoken output through a TTS service on the robot itself.

Many robots ship an offline TTS node that takes text and plays it on the
robot's speaker as a ROS 2 service call. The service name and type differ from
robot to robot, so both come from config (speech_output.ros2), and the service
type is imported by name at runtime: this package depends on no robot's
interface package. Synthesis and playback both happen inside that one call,
and such services usually offer no way to stop playback once it has started.

That makes this engine differ from SpeechOutput in one way that matters: it
cannot be stopped once it starts talking. Delivery speaks a reply as one piece
(up to conversation.max_reply_sentences sentences), so that whole piece is what
plays out. An interrupt that arrives before the piece starts still cancels it.
One that arrives while the robot is talking cannot silence it; the piece plays
to the end and is then reported as interrupted, so the rest of the turn (the
length-guard question, held-back sentences) is dropped exactly as it is after a
real interruption. The stop latency that delivery logs is the true one, from
the interrupt to the moment the robot went quiet.

Everything else is simpler than SpeechOutput: the service is offline, so every
phrase works with no network, and there is nothing to cache or prefetch.
"""

import importlib
import logging
import time

from .speech import SpeakResult

log = logging.getLogger(__name__)

# Playback happens inside the service call, so its timeout has to cover
# speaking the text, not only synthesizing it. Mandarin TTS runs at about five
# characters a second; three leaves headroom for a slower voice or speed.
MIN_CHARS_PER_SEC = 3.0

# How long to wait for the service to appear before giving up on a sentence.
SERVICE_WAIT_SEC = 3.0


class RobotTtsUnavailable(RuntimeError):
    """ROS 2 or the configured service type is not installed on this machine."""


class RobotTtsOutput:
    """Speaks through the robot's TTS service. Same interface as SpeechOutput."""

    name = "ros2"

    def __init__(self, settings, timeout_sec, client=None):
        """
        settings:    speech_output.ros2 from config.yaml
        timeout_sec: timeouts.tts_sec, the base of every call's timeout
        client:      injected by tests; otherwise a ROS 2 service client
        """
        self._settings = settings
        self._timeout = timeout_sec
        self._client = client if client is not None else Ros2TtsClient(settings)

    def say(self, text, on_playback_start=None, interrupt=None):
        """Speak text on the robot. Never raises; see SpeechOutput.say."""
        text = (text or "").strip()
        if not text:
            return SpeakResult(spoken=False)
        if interrupt is not None and interrupt.is_set():
            return SpeakResult(spoken=False, interrupted=True)

        limit = self._settings.max_chars
        if len(text) > limit:
            log.warning("Reply is %d characters; the robot TTS is given the first %d",
                        len(text), limit)
            text = text[:limit]

        if on_playback_start is not None:
            try:
                on_playback_start()
            except Exception as e:
                # A gesture must never stop the robot from speaking.
                log.warning("Playback-start callback failed: %s", e)

        timeout = self._timeout + len(text) / MIN_CHARS_PER_SEC
        started_at = time.monotonic()
        try:
            self._client.speak(text, timeout)
        except Exception as e:
            log.warning("Robot TTS could not speak %r: %s", text, e)
            return SpeakResult(spoken=False)
        stopped_at = time.monotonic()

        interrupted = interrupt is not None and interrupt.is_set()
        if interrupted:
            log.info("Interrupted while the robot TTS was speaking; it cannot stop "
                     "mid-sentence, so the sentence was finished first")
        return SpeakResult(spoken=True, interrupted=interrupted,
                           started_at=started_at, stopped_at=stopped_at)

    def prefetch(self, text):
        """Nothing to do: the robot synthesizes as it speaks."""

    def prepare(self, phrases):
        """Nothing to do: the robot TTS is offline, so every phrase already works."""


def import_service_type(type_name):
    """The service class for 'pkg/srv/Name', imported from the sourced ROS 2 environment.

    rosidl generates every service type as the Python class pkg.srv.Name, so
    the type named in config can be imported without depending on its package.
    """
    package, kind, name = type_name.split("/")
    if kind != "srv":
        raise ValueError(f"{type_name!r} is not a service type (pkg/srv/Name)")
    module = importlib.import_module(f"{package}.srv")
    try:
        return getattr(module, name)
    except AttributeError:
        raise ImportError(f"{package}.srv has no service named {name}") from None


def _check_request_fields(settings, request_type):
    """Fail at startup, not on the first reply, when config names a field the type lacks."""
    if not hasattr(request_type, "get_fields_and_field_types"):
        return
    known = set(request_type.get_fields_and_field_types())
    for name in [settings.text_field, *settings.request]:
        if name not in known:
            raise ValueError(
                f"speech_output.ros2: {settings.type} requests have no field {name!r}; "
                f"they have {', '.join(sorted(known))}"
            )


class Ros2TtsClient:
    """Calls the configured TTS service and waits for playback to finish.

    The request carries the text in settings.text_field and every entry of
    settings.request as a field of the same name. A response field named
    `success` is checked when the type has one, with `message` as the reason.
    """

    def __init__(self, settings):
        try:
            import rclpy
            service_type = import_service_type(settings.type)
        except ImportError as e:
            raise RobotTtsUnavailable(
                f"the robot TTS needs ROS 2 and the {settings.type} service type ({e}). "
                f"Run inside the robot's sourced ROS 2 environment, or set "
                f"speech_output.engine: edge-tts."
            ) from e

        self._rclpy = rclpy
        self._request_type = service_type.Request
        _check_request_fields(settings, self._request_type)
        self._text_field = settings.text_field
        self._fields = dict(settings.request)
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("voice_agent_tts")
        self._client = self._node.create_client(service_type, settings.service)
        self.service = settings.service
        log.info("Speaking through %s (%s)", settings.service, settings.type)

    def speak(self, text, timeout_sec):
        if not self._client.wait_for_service(timeout_sec=SERVICE_WAIT_SEC):
            raise RuntimeError(f"{self.service} is not available; is the robot's TTS node running?")

        request = self._request_type()
        setattr(request, self._text_field, text)
        for name, value in self._fields.items():
            setattr(request, name, value)
        future = self._client.call_async(request)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout_sec)

        if not future.done():
            future.cancel()
            raise TimeoutError(
                f"{self.service} did not finish within {timeout_sec:.0f} s; "
                f"the robot may still be speaking"
            )
        result = future.result()
        if getattr(result, "success", True) is False:
            reason = getattr(result, "message", "") or "no reason given"
            raise RuntimeError(f"{self.service} reported failure: {reason}")
