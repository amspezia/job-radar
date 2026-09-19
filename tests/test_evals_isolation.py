"""The harness must not drag production code in with it.

Importing `job_radar.retrieval.vector` pulls `job_radar.config` (four required env vars)
and `job_radar.db.base`, which builds a read-write engine at import. Three modules are
allowed to pay that price because they genuinely read production; nothing else may, and
nothing may touch `eval.run` / `eval.qrels` / `eval.label` (implementation plan §3).

Two checks, because neither is enough alone: an AST rule catches direct imports cheaply
but passes a module that imports an allow-listed one (or uses `importlib`), and a
subprocess import catches exactly that — it is the transitive truth.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("eval/evals_db", "eval/embedding")

# The only modules that may import production code.
PROD_TOUCHING = {
    "eval/evals_db/prod_reader.py",
    "eval/embedding/tiers/tier_a.py",
    "eval/embedding/parity.py",
}
# And the only production modules they may import: never `job_radar.db.base`,
# `job_radar.config`, `job_radar.fit` or `job_radar.adapters`.
ALLOWED_PROD_IMPORTS = {
    "job_radar.db.models",
    "job_radar.retrieval.vector",
    "job_radar.retrieval.filters",
    "job_radar.ingest.embed_text",
}
FORBIDDEN_EVAL_MODULES = ("eval.run", "eval.qrels", "eval.label")

PROBE = """
import importlib
import json
import sys

for name in json.loads(sys.argv[1]):
    importlib.import_module(name)

print(json.dumps(sorted(
    name for name in sys.modules
    if name.split(".")[0] in {"job_radar", "langfuse"}
    or name in {"eval.run", "eval.qrels", "eval.label"}
)))
"""


def _harness_files() -> list[Path]:
    files = sorted(path for d in SCANNED for path in (REPO_ROOT / d).rglob("*.py"))
    assert files, f"no harness modules found under {SCANNED} — the scan would pass vacuously"
    return files


def _imports(path: Path) -> tuple[set[str], set[str]]:
    """(modules imported, those plus `from X import y` spelled as `X.y`), at any depth.

    Function-local imports count: a lazy import still runs, and the subprocess check below
    is what makes laziness acceptable in the first place.
    """
    modules: set[str] = set()
    qualified: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            qualified.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules, modules | qualified


def _is_prod(module: str) -> bool:
    return module == "job_radar" or module.startswith("job_radar.")


def test_only_the_allow_listed_modules_import_production() -> None:
    violations = [
        f"{path.relative_to(REPO_ROOT)} imports {module}"
        for path in _harness_files()
        if path.relative_to(REPO_ROOT).as_posix() not in PROD_TOUCHING
        for module in sorted(_imports(path)[0])
        if _is_prod(module)
    ]

    assert violations == []


def test_the_allow_listed_modules_import_only_their_four_production_modules() -> None:
    violations = [
        f"{path.relative_to(REPO_ROOT)} imports {module}"
        for path in _harness_files()
        if path.relative_to(REPO_ROOT).as_posix() in PROD_TOUCHING
        for module in sorted(_imports(path)[0])
        if _is_prod(module) and module not in ALLOWED_PROD_IMPORTS
    ]

    assert violations == []


def test_nothing_imports_the_existing_eval_harness() -> None:
    # `eval/run.py` imports `job_radar.db.base` at module top (F13), and its qrels and
    # label logic is deliberately reimplemented in `eval/embedding/labels.py`.
    violations = [
        f"{path.relative_to(REPO_ROOT)} imports {module}"
        for path in _harness_files()
        for module in sorted(_imports(path)[1])
        if module.startswith(FORBIDDEN_EVAL_MODULES)
    ]

    assert violations == []


def test_importing_the_harness_pulls_no_production_module() -> None:
    """The transitive check: one clean interpreter, prod's environment stripped."""
    names = []
    for path in _harness_files():
        if path.relative_to(REPO_ROOT).as_posix() in PROD_TOUCHING:
            continue
        parts = path.relative_to(REPO_ROOT).with_suffix("").parts
        names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    assert "eval.evals_db.cli" in names, "the CLIs must import their prod-touching parts lazily"

    result = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(names)],
        cwd=REPO_ROOT,
        # No DATABASE_URL, OLLAMA_BASE_URL, EMBEDDING_MODEL, GENERATION_MODEL or
        # EVALS_DATABASE_URL: a module that builds settings at import fails here.
        env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"importing {names} failed:\n{result.stderr}"
    assert json.loads(result.stdout.splitlines()[-1]) == []
