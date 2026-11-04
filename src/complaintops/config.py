"""Load settings from a local .env file into the environment (real environment variables win).

Called explicitly by entry points (the API, evaluation scripts), not at import time, so tests and libraries
never pick up keys by accident. .env is git-ignored; .env.example lists the variable names.
"""
import os
from pathlib import Path


def load_env(path=".env"):
    p = Path(path)
    if not p.exists():
        return False
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #")[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)
    return True
