"""Assemble the folder that gets pushed to a Hugging Face Space (default: build/space).

Copies only what the hosted server needs: the runtime modules (no training code, nothing that imports PyTorch), the
page, the int8 model with its tokenizer, and the exported search index. Refuses to finish if anything that looks like
an API key ends up in the folder.
    python deploy/build_space.py
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "space"

RUNTIME_MODULES = ["__init__", "api", "agent", "llm", "tools", "mcp_server", "serve_onnx", "search_index",
                   "taxonomy", "config", "tracing"]
ONNX_FILES = ["model_int8.onnx", "classes.json", "bias.json", "tokenizer/tokenizer.json"]
INDEX_FILES = ["vectors.npy", "meta.json.gz"]
SECRET_PATTERNS = [r"gsk_[A-Za-z0-9]{20,}", r"sk-lf-[0-9a-f-]{20,}", r"pk-lf-[0-9a-f-]{20,}", r"sk-[A-Za-z0-9]{32,}"]
LFS = ["*.onnx", "*.npy"]  # Spaces need Git LFS for files over 10 MB


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for m in RUNTIME_MODULES:
        copy(ROOT / "src" / "complaintops" / f"{m}.py", OUT / "src" / "complaintops" / f"{m}.py")
    for f in ["index.html", "app.js", "styles.css", "showcase.json"]:
        copy(ROOT / "web" / f, OUT / "web" / f)
    for f in ONNX_FILES:
        copy(ROOT / "data" / "onnx" / f, OUT / "data" / "onnx" / f)
    for f in INDEX_FILES:
        copy(ROOT / "data" / "search_index" / f, OUT / "data" / "search_index" / f)
    copy(ROOT / "deploy" / "Dockerfile", OUT / "Dockerfile")
    copy(ROOT / "deploy" / "requirements.txt", OUT / "requirements.txt")
    copy(ROOT / "deploy" / "space_README.md", OUT / "README.md")
    (OUT / ".gitattributes").write_text("".join(f"{p} filter=lfs diff=lfs merge=lfs -text\n" for p in LFS), encoding="utf-8")

    for p in OUT.rglob("*"):  # nothing secret may be shipped
        if p.is_file() and p.suffix in {".py", ".js", ".html", ".css", ".json", ".md", ".txt", ".yml"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            for pat in SECRET_PATTERNS:
                if re.search(pat, text):
                    sys.exit(f"refusing to finish: {p} looks like it contains a key ({pat})")
    if any(p.name == ".env" for p in OUT.rglob("*")):
        sys.exit("refusing to finish: a .env file is in the bundle")

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"built {OUT} ({total:.0f} MB, {sum(1 for p in OUT.rglob('*') if p.is_file())} files)")


if __name__ == "__main__":
    main()
