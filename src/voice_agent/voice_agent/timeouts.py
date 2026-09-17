"""Hard time limit for blocking calls (network, camera) so the agent never hangs."""

import threading


class CallTimeout(TimeoutError):
    pass


def call_with_timeout(fn, timeout_sec, *args, **kwargs):
    """Run fn in a daemon thread and wait at most timeout_sec.

    On timeout the thread is abandoned (it cannot be killed) and CallTimeout
    is raised. Exceptions from fn are re-raised in the caller.
    """
    outcome = {}

    def target():
        try:
            outcome["value"] = fn(*args, **kwargs)
        except BaseException as e:  # re-raised in the calling thread
            outcome["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        raise CallTimeout(f"{getattr(fn, '__name__', 'call')} took longer than {timeout_sec}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")
