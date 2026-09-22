"""Timeouts and cancellation of in-flight calls. No network."""

import threading
import time

import pytest

from voice_agent.timeouts import CallCancelled, CallTimeout, call_with_timeout


def test_returns_the_result():
    assert call_with_timeout(lambda x: x * 2, 1.0, 21) == 42


def test_raises_the_callees_error():
    def boom():
        raise ValueError("no")

    with pytest.raises(ValueError):
        call_with_timeout(boom, 1.0)


def test_timeout():
    release = threading.Event()
    try:
        with pytest.raises(CallTimeout):
            call_with_timeout(lambda: release.wait(5), 0.1)
    finally:
        release.set()


def test_cancel_stops_waiting_promptly():
    """An interrupted person should not wait out the request they abandoned."""
    cancel = threading.Event()
    release = threading.Event()
    threading.Timer(0.05, cancel.set).start()

    started = time.monotonic()
    try:
        with pytest.raises(CallCancelled):
            call_with_timeout(lambda: release.wait(10), 10.0, cancel_event=cancel)
    finally:
        release.set()
    assert time.monotonic() - started < 1.0   # not the 10 s timeout


def test_cancel_before_the_call_finishes_discards_the_result():
    cancel = threading.Event()
    release = threading.Event()
    seen = []

    def slow():
        release.wait(5)
        seen.append("finished")
        return "答案"

    cancel.set()
    with pytest.raises(CallCancelled):
        call_with_timeout(slow, 5.0, cancel_event=cancel)
    release.set()
    # The call may still finish in its own thread, but its result is not used.
    assert seen in ([], ["finished"])


def test_a_call_that_finishes_first_is_not_cancelled():
    cancel = threading.Event()
    assert call_with_timeout(lambda: "好", 1.0, cancel_event=cancel) == "好"
