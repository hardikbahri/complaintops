"""Optional Langfuse tracing of agent runs. Milestone 2.

Does nothing unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (and LANGFUSE_TRACING is not "off"), and a
tracing problem can never break a request: every call here swallows its own errors. One run becomes one trace: a
root span "triage" with a child span per tool call (through MCP) and a generation for the LLM call, with timings.
The run's text is sent to Langfuse's cloud when tracing is on, so a public deployment should leave the keys unset
or let visitors opt out per request (the API accepts trace=false).
"""
import os
import threading
from contextlib import contextmanager

_client = None
_lock = threading.Lock()


def configured():
    keys = os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    return bool(keys) and os.environ.get("LANGFUSE_TRACING", "on").lower() != "off"


def client():
    global _client
    with _lock:
        if _client is None:
            from langfuse import Langfuse

            _client = Langfuse()  # reads the keys and the base URL from the environment
        return _client


def flush():
    try:
        client().flush()
    except Exception:
        pass


class Handle:
    """One Langfuse observation, or nothing at all. Safe to call in every case."""

    def __init__(self, obs=None):
        self.obs = obs

    def update(self, **fields):
        if self.obs is not None:
            try:
                self.obs.update(**fields)
            except Exception:
                pass

    def end(self):
        if self.obs is not None:
            try:
                self.obs.end()
            except Exception:
                pass

    def child(self, name, as_type="span", **fields):
        if self.obs is None:
            return Handle()
        try:
            return Handle(self.obs.start_observation(name=name, as_type=as_type, **fields))
        except Exception:
            return Handle()

    @contextmanager
    def step(self, name, as_type="span", **fields):
        h = self.child(name, as_type, **fields)
        try:
            yield h
        except BaseException as e:
            h.update(level="ERROR", status_message=f"{type(e).__name__}: {e}"[:300])
            raise
        finally:
            h.end()


class Trace(Handle):
    """The root of one run."""

    trace_id = None

    @classmethod
    def start(cls, name, input=None, metadata=None, enabled=True):
        if not (enabled and configured()):
            return cls()
        try:
            obs = client().start_observation(name=name, as_type="span", input=input, metadata=metadata)
            t = cls(obs)
            t.trace_id = obs.trace_id
            return t
        except Exception:
            return cls()

    @property
    def active(self):
        return self.obs is not None

    def url(self):
        try:
            return client().get_trace_url(trace_id=self.trace_id)
        except Exception:
            return None
