"""The hosted bundle: it contains every module the server needs, and importing the server pulls in nothing heavy."""
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "complaintops"


def runtime_modules():
    spec = importlib.util.spec_from_file_location("build_space", ROOT / "deploy" / "build_space.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RUNTIME_MODULES


def complaintops_imports(path):
    """Every complaintops module a file imports, including imports inside functions (the lazy ones)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("complaintops"):
            parts = node.module.split(".")
            found.update([parts[1]] if len(parts) > 1 else [alias.name for alias in node.names])
        elif isinstance(node, ast.Import):
            found.update(a.name.split(".")[1] for a in node.names if a.name.startswith("complaintops."))
    return found


def test_every_module_the_server_imports_is_in_the_bundle():
    shipped = set(runtime_modules())
    # vector_store is the pgvector backend: tools imports it only when SEARCH_BACKEND is not "memory"
    optional = {"vector_store"}
    for module in sorted(shipped - {"__init__"}):
        missing = complaintops_imports(SRC / f"{module}.py") - shipped - optional
        assert not missing, f"{module}.py imports {missing}, which the bundle does not ship"


def test_importing_the_server_does_not_load_pytorch_a_database_driver_or_the_training_stack():
    code = ("import sys, os; os.environ['SEARCH_BACKEND']='memory'\n"
            "import complaintops.api, complaintops.agent, complaintops.tools, complaintops.mcp_server\n"
            "import complaintops.serve_onnx, complaintops.search_index, complaintops.tracing, complaintops.llm\n"
            "heavy = [m for m in ('torch', 'transformers', 'peft', 'sklearn', 'pandas', 'psycopg', 'sentence_transformers', "
            "'mlflow', 'langfuse', 'xgboost') if m in sys.modules]\n"
            "print('LOADED:' + ','.join(heavy))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}).stdout
    assert out.strip().endswith("LOADED:"), out  # nothing heavy: the slim container would not have it installed
