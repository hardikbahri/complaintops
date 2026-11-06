"""The web page: it is served, its numbers come from the real result files, and no secret is in the repo files."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [r"gsk_[A-Za-z0-9]{20,}", r"sk-lf-[0-9a-f-]{20,}", r"pk-lf-[0-9a-f-]{20,}",
                   r"sk-[A-Za-z0-9]{32,}", r"AKIA[0-9A-Z]{16}"]


def test_no_api_keys_in_the_page_the_readme_or_the_results():
    """Keys live in the git-ignored .env only. Scan everything that gets committed or served."""
    files = [p for d in ("web", "results", "src", "tests") for p in (ROOT / d).rglob("*")
             if p.is_file() and p.suffix in {".html", ".js", ".css", ".json", ".py", ".md", ".txt", ".csv", ".yml"}]
    files += [ROOT / "README.md", ROOT / ".env.example", ROOT / "docker-compose.yml"]
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            assert not re.search(pat, text), f"{p} contains something that looks like a key ({pat})"


def test_env_file_is_git_ignored():
    assert re.search(r"^\.env$", (ROOT / ".gitignore").read_text(), re.M)


def test_showcase_numbers_match_the_gate_reports():
    s = json.loads((ROOT / "web" / "showcase.json").read_text(encoding="utf-8"))
    g = json.loads((ROOT / "results" / "gate24_v4_vs_v3.json").read_text())
    assert s["gates"]["refinement"]["candidate"]["weighted_f1"] == g["candidate"]["weighted_f1"]
    assert s["gates"]["blocked"]["passed"] is False and s["gates"]["promoted"]["passed"] is True
    for key in ("facts", "taxonomy", "calibration", "serving", "drift", "gates", "retrieval"):
        assert s[key], key
    names = [m["name"] for m in s["taxonomy"]["validation"]]
    assert names[0].startswith("TF-IDF")
    assert len(s["lessons"]) == 2  # the experiments that did not work, with numbers from the result files
    for lesson in s["lessons"]:
        assert all(lesson[k] for k in ("title", "verdict", "tried", "result", "learned"))
    xgb_card = s["lessons"][0]["result"]
    v = s["taxonomy"]["validation"]
    assert f"{v[0]['weighted_f1']:.3f}" in xgb_card and "XGBoost" in s["lessons"][0]["title"]


def test_the_page_only_shows_the_24_served_classes():
    text = json.dumps(json.loads((ROOT / "web" / "showcase.json").read_text(encoding="utf-8")))
    text += (ROOT / "web" / "app.js").read_text(encoding="utf-8") + (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    for banned in ("26 classes", "26-class", "original 26", "26 to 19", "models_validation", "calibration_26"):
        assert banned not in text, banned


needs_model = pytest.mark.skipif(not (ROOT / "data" / "onnx" / "model_int8.onnx").exists(), reason="run export_onnx first")


@needs_model
def test_page_assets_are_served_and_api_routes_are_not_shadowed():
    from fastapi.testclient import TestClient

    from complaintops.api import app

    with TestClient(app) as c:
        page = c.get("/")
        assert page.status_code == 200 and "ComplaintOps" in page.text and 'src="app.js"' in page.text
        for asset in ("/styles.css", "/app.js", "/showcase.json"):
            assert c.get(asset).status_code == 200, asset
        assert c.get("/health").json() == {"status": "ok"}   # the static mount must not hide the API
        assert c.get("/docs").status_code == 200
        assert c.get("/does-not-exist").status_code == 404


def test_the_recorded_runs_are_public_safe_and_match_the_example_chips():
    """The static copy replays these: they must carry no private link, and each must be reachable from a chip."""
    s = json.loads((ROOT / "web" / "showcase.json").read_text(encoding="utf-8"))
    runs = s["recorded_runs"]
    assert len(runs) == 4 and "langfuse" not in json.dumps(runs).lower()
    assert all(r.get("trace_url") is None and r.get("trace_id") is None for r in runs)
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    for r in runs:
        assert r["text"] in js, f"no example chip for the recorded run {r['label']!r}"  # the chips replay by exact text
