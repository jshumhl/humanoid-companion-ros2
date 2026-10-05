"""What the agent is doing right now, for other parts of the robot to follow.

The agent is always in one of four phases:

    idle       waiting for the operator (push-to-talk, before Enter) or stopped
    listening  recording, or waiting for someone to speak (always-on)
    thinking   recognizing speech and waiting for the model
    speaking   playing a reply

A face display, a status light or a logger can follow these to show the robot
is listening or thinking. The phase is reported only when it changes.

Reports go through a backend, like gestures. The default one only logs, so the
agent runs with no ROS installed. The ROS 2 backend publishes the phase name as
std_msgs/String with transient-local durability, so a node that starts later
still receives the current phase.
"""

import logging

log = logging.getLogger(__name__)

IDLE = "idle"
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"
PHASES = (IDLE, LISTENING, THINKING, SPEAKING)


class StatusBackendUnavailable(RuntimeError):
    """The selected backend cannot be used, e.g. ROS 2 is not installed."""


class StatusBackend:
    name = "base"

    def publish(self, phase):
        raise NotImplementedError

    def close(self):
        pass


class StubBackend(StatusBackend):
    """Default backend: records the phase in the debug log only."""

    name = "stub"

    def publish(self, phase):
        log.debug("status: %s", phase)


class Ros2Backend(StatusBackend):
    """Publishes the phase as std_msgs/String on a latched (transient-local) topic."""

    name = "ros2"

    def __init__(self, topic):
        try:
            import rclpy
            from rclpy.qos import DurabilityPolicy, QoSProfile
            from std_msgs.msg import String
        except ImportError as e:
            raise StatusBackendUnavailable(
                f"ROS 2 is not available ({e}). Use status.backend: stub, or run inside "
                f"a sourced ROS 2 environment."
            ) from e

        self._rclpy = rclpy
        self._string = String
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self._node = rclpy.create_node("voice_agent_status")
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._publisher = self._node.create_publisher(String, topic, qos)
        self.topic = topic
        log.info("Publishing status on %s", topic)

    def publish(self, phase):
        self._publisher.publish(self._string(data=phase))

    def close(self):
        self._node.destroy_node()
        # Other ROS 2 parts of the agent (gestures, robot TTS) may share the
        # context, so it is shut down only if this backend started it.
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


def make_backend(status_config):
    """Build the configured backend, falling back to the stub if it is unavailable."""
    if status_config.backend == "ros2":
        try:
            return Ros2Backend(status_config.ros2_topic)
        except StatusBackendUnavailable as e:
            log.warning("Status backend 'ros2' unavailable, using 'stub': %s", e)
    return StubBackend()


class StatusReporter:
    """Reports phase changes. Never raises: a status display must not stop speech."""

    def __init__(self, backend):
        self._backend = backend
        self.phase = None

    def set(self, phase):
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {', '.join(PHASES)}")
        if phase == self.phase:
            return
        self.phase = phase
        try:
            self._backend.publish(phase)
        except Exception as e:
            log.warning("Status %s could not be reported: %s", phase, e)

    def close(self):
        self.set(IDLE)
        self._backend.close()


class StatusSpeech:
    """Wraps a speech output so that playback reports `speaking`.

    The phase changes when playback actually starts, the same moment a gesture
    starts, so a display never shows speaking while the reply is still being
    synthesized. What comes after speaking is set by the conversation loop.
    """

    def __init__(self, speech, status):
        self._speech = speech
        self._status = status

    def say(self, text, on_playback_start=None, interrupt=None):
        def started():
            self._status.set(SPEAKING)
            if on_playback_start is not None:
                on_playback_start()

        return self._speech.say(text, on_playback_start=started, interrupt=interrupt)

    def __getattr__(self, name):
        return getattr(self._speech, name)


def build_reporter(status_config):
    if not status_config.enabled:
        return StatusReporter(StubBackend())
    return StatusReporter(make_backend(status_config))
