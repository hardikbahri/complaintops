"""Assemble the static copy of the page (default: build/static) for a free static host such as a Hugging Face static Space.

A static host has no backend, so the page detects that and replays the recorded agent runs, saying so. The bundle is just
the four page files plus a README; it refuses to finish if a key or a private trace link is in it.
    python deploy/build_static.py
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "static"
PAGE_FILES = ["index.html", "app.js", "styles.css", "showcase.json"]
FORBIDDEN = [r"gsk_[A-Za-z0-9]{20,}", r"sk-lf-[0-9a-f-]{20,}", r"pk-lf-[0-9a-f-]{20,}", r"sk-[A-Za-z0-9]{32,}",
             r"langfuse\.com/project", r"hf_[A-Za-z0-9]{30,}"]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for f in PAGE_FILES:
        shutil.copy2(ROOT / "web" / f, OUT / f)
    shutil.copy2(ROOT / "deploy" / "space_static_README.md", OUT / "README.md")
    for p in OUT.rglob("*"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in FORBIDDEN:
            if re.search(pat, text):
                sys.exit(f"refusing to finish: {p.name} matches {pat}")
    size = sum(p.stat().st_size for p in OUT.iterdir()) / 1e3
    print(f"built {OUT} ({size:.0f} KB, {len(list(OUT.iterdir()))} files)")


if __name__ == "__main__":
    main()
