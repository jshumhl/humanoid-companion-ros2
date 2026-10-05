"""Hard time limit for blocking calls (network, camera) so the agent never hangs."""

import threading

CANCEL_POLL_SEC = 0.02  # how soon a cancel is noticed


class CallTimeout(TimeoutError):
    pass


class CallCancelled(Exception):
    """The caller stopped waiting, e.g. the person interrupted the robot."""


def call_with_timeout(fn, timeout_sec, *args, cancel_event=None, **kwargs):
    """Run fn in a daemon thread and wait at most timeout_sec.

    On timeout the thread is abandoned (it cannot be killed) and CallTimeout
    is raised. With `cancel_event`, waiting also stops as soon as the event is
    set, raising CallCancelled. The call itself keeps running in its thread
    until the network or the OS ends it, but its result is never used.
    Exceptions from fn are re-raised in the caller.
    """
    outcome = {}
    done = threading.Event()

    def target():
        try:
            outcome["value"] = fn(*args, **kwargs)
        except BaseException as e:  # re-raised in the calling thread
            outcome["error"] = e
        finally:
            done.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    if cancel_event is None:
        done.wait(timeout_sec)
    else:
        deadline = threading.TIMEOUT_MAX if timeout_sec is None else timeout_sec
        waited = 0.0
        while not done.wait(CANCEL_POLL_SEC):
            waited += CANCEL_POLL_SEC
            if cancel_event.is_set():
                raise CallCancelled(f"{_name(fn)} was cancelled")
            if waited >= deadline:
                break

    if not done.is_set():
        raise CallTimeout(f"{_name(fn)} took longer than {timeout_sec}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _name(fn):
    return getattr(fn, "__name__", "call")
