"""`job-radar-eval-embedding` argument parsing, dispatch and import hygiene.

No database, no Ollama: the commands are monkeypatched, or run only against fakes.
"""

import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from eval.embedding import cli

RUN_ID = "0f3f3a8e-2a8f-4a1e-9a5b-9d4f7c3a1b22"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["import-topic", "--tier", "A", "--name", "A-real"],
            {"tier": "A", "name": "A-real", "profile_id": None},
        ),
        (["embed", "--topic", "t", "--embedder", "all"], {"topic": "t", "embedder": "all"}),
        (["embed", "--topic", "t", "--embedder", "bge-m3"], {"embedder": "bge-m3"}),
        (["run", "--topic", "t", "--embedder", "all"], {"embedder": "all", "per_text": False}),
        (["run", "--topic", "t", "--embedder", "nomic-v1.5", "--per-text"], {"per_text": True}),
        (["verify", "--topic", "t"], {"topic": "t"}),
        (
            ["judge", "--topic", "t", "--model", "gemma3:12b"],
            {
                "model": "gemma3:12b",
                "limit": None,
                "resume": None,
                "concurrency": 4,
                "fewshot_seed": 13,
                "fewshot_per_grade": 1,
            },
        ),
        (
            [
                "judge",
                "--topic",
                "t",
                "--model",
                "m",
                "--limit",
                "24",
                "--resume",
                RUN_ID,
                "--concurrency",
                "1",
                "--fewshot-per-grade",
                "2",
            ],
            {"limit": 24, "resume": UUID(RUN_ID), "concurrency": 1, "fewshot_per_grade": 2},
        ),
        (
            ["judge-calibrate", "--topic", "t", "--judge-run", RUN_ID],
            {"judge_run": UUID(RUN_ID), "min_kappa": 0.6, "held_out_half": False},
        ),
        (
            [
                "judge-calibrate",
                "--topic",
                "t",
                "--judge-run",
                RUN_ID,
                "--held-out-half",
                "--min-kappa",
                "0.5",
            ],
            {"min_kappa": 0.5, "held_out_half": True},
        ),
        (
            ["label-blind", "--topic", "t"],
            {"topic": "t", "n": 50, "bucket": "mixed", "seed": 0, "language": "en"},
        ),
        (
            [
                "label-blind",
                "--topic",
                "t",
                "--n",
                "20",
                "--bucket",
                "pooled",
                "--seed",
                "3",
                "--language",
                "any",
            ],
            {"n": 20, "bucket": "pooled", "seed": 3, "language": "any"},
        ),
        (
            ["evaluate", "--topic", "t", "--view", "all"],
            {
                "view": "all",
                "judge_run": None,
                "bootstrap": 1000,
                "allow_unverified": False,
                "language": None,
                "out_dir": "eval/results",
            },
        ),
        (
            [
                "evaluate",
                "--topic",
                "t",
                "--view",
                "silver",
                "--judge-run",
                RUN_ID,
                "--bootstrap",
                "10",
                "--allow-unverified",
                "--language",
                "en",
                "--out-dir",
                "/tmp/out",
            ],
            {
                "judge_run": UUID(RUN_ID),
                "bootstrap": 10,
                "allow_unverified": True,
                "language": "en",
                "out_dir": "/tmp/out",
            },
        ),
        (["topics"], {"command": "topics"}),
        (["labels-export", "--topic", "t"], {"out": None}),
        (["labels-export", "--topic", "t", "--out", "b.json"], {"out": "b.json"}),
        (["labels-import", "--topic", "t", "--file", "b.json"], {"file": "b.json"}),
    ],
)
def test_parses_each_subcommand(argv: list[str], expected: dict) -> None:
    args = cli.build_parser().parse_args(argv)
    assert callable(args.func)
    for key, value in expected.items():
        assert getattr(args, key) == value


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["import-topic", "--tier", "A"],
        ["import-topic", "--name", "t"],
        ["import-topic", "--tier", "B", "--name", "t"],
        ["embed", "--topic", "t"],
        ["run", "--embedder", "all"],
        ["verify"],
        ["judge", "--topic", "t"],
        ["judge-calibrate", "--topic", "t"],
        ["judge-calibrate", "--topic", "t", "--judge-run", "not-a-uuid"],
        ["judge-calibrate", "--topic", "t", "--judge-run", RUN_ID, "--all-labels"],
        ["label-blind"],
        ["label-blind", "--topic", "t", "--bucket", "strong"],
        ["label-blind", "--topic", "t", "--n", "many"],
        ["label-blind", "--topic", "t", "--language", "fr"],
        ["evaluate", "--topic", "t", "--view", "silver", "--language", "any"],
        ["evaluate", "--topic", "t"],
        ["labels-import", "--topic", "t"],
    ],
)
def test_missing_or_invalid_arguments_exit(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)


@pytest.mark.parametrize("view", ["silver", "human", "effective", "all"])
def test_view_choices(view: str) -> None:
    assert cli.build_parser().parse_args(["evaluate", "--topic", "t", "--view", view]).view == view


def test_unknown_view_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["evaluate", "--topic", "t", "--view", "gold"])


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        ("_import_topic", ["import-topic", "--tier", "A", "--name", "A-real"]),
        ("_embed", ["embed", "--topic", "A-real", "--embedder", "all"]),
        ("_run", ["run", "--topic", "A-real", "--embedder", "all", "--per-text"]),
        ("_verify", ["verify", "--topic", "A-real"]),
        ("_judge", ["judge", "--topic", "A-real", "--model", "gemma3:12b"]),
        ("_label_blind", ["label-blind", "--topic", "A-real", "--n", "5"]),
        ("_judge_calibrate", ["judge-calibrate", "--topic", "A-real", "--judge-run", RUN_ID]),
        ("_evaluate", ["evaluate", "--topic", "A-real", "--view", "human"]),
        ("_topics", ["topics"]),
        ("_labels_export", ["labels-export", "--topic", "A-real"]),
        ("_labels_import", ["labels-import", "--topic", "A-real", "--file", "b.json"]),
    ],
)
def test_main_routes_to_the_command_and_returns_its_code(
    monkeypatch: pytest.MonkeyPatch, command: str, argv: list[str]
) -> None:
    calls = []

    async def fake(args) -> int:
        calls.append(args)
        return 7

    monkeypatch.setattr(cli, command, fake)
    assert cli.main(argv) == 7
    assert len(calls) == 1


def test_select_specs_by_name_and_all() -> None:
    specs = [SimpleNamespace(name="nomic-v1.5"), SimpleNamespace(name="bge-m3")]
    assert cli._select_specs(specs, "all") == specs
    assert cli._select_specs(specs, "bge-m3") == [specs[1]]


def test_select_specs_unknown_name_lists_the_configured_ones() -> None:
    specs = [SimpleNamespace(name="nomic-v1.5"), SimpleNamespace(name="bge-m3")]
    with pytest.raises(ValueError, match="unknown embedder 'qwen3'") as excinfo:
        cli._select_specs(specs, "qwen3")
    message = str(excinfo.value)
    assert "nomic-v1.5" in message
    assert "bge-m3" in message


def test_recipes_cover_every_hyde_text_only_with_per_text() -> None:
    topic = SimpleNamespace(query_inputs={"hyde_texts": ["a", "b", "c"]})
    assert cli._recipes(topic, False) == ["hyde_mean"]
    assert cli._recipes(topic, True) == [
        "hyde_mean",
        "hyde_text:0",
        "hyde_text:1",
        "hyde_text:2",
    ]


def test_import_pulls_neither_the_prod_touching_modules_nor_job_radar() -> None:
    probe = (
        "import sys, eval.embedding.cli;"
        "print(sorted(m for m in sys.modules if m.startswith('job_radar')"
        " or m in ('eval.embedding.tiers.tier_a', 'eval.embedding.parity')))"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    assert done.stdout.strip() == "[]"


@pytest.mark.parametrize(
    ("flag", "expected"), [([], "en"), (["--language", "es"], "es"), (["--language", "any"], None)]
)
def test_label_blind_passes_the_language_and_any_means_no_filter(
    monkeypatch: pytest.MonkeyPatch, flag: list[str], expected: str | None
) -> None:
    calls: dict = {}

    @asynccontextmanager
    async def fake_session():
        yield "session"

    async def fake_label_blind(session, topic, **kwargs) -> dict:
        calls.update(kwargs)
        return {"labeled": 0, "skipped": 0, "remaining": 0}

    monkeypatch.setattr("eval.evals_db.base.evals_session", fake_session)
    monkeypatch.setattr("eval.embedding.label_blind.label_blind", fake_label_blind)

    assert cli.main(["label-blind", "--topic", "t", *flag]) == 0
    assert calls["language"] == expected


def test_evaluate_passes_the_language_to_the_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []

    @asynccontextmanager
    async def fake_session():
        yield "session"

    async def fake_evaluate(session, topic, view, **kwargs) -> dict:
        seen.append(kwargs["language"])
        return {}

    monkeypatch.setattr("eval.evals_db.base.evals_session", fake_session)
    monkeypatch.setattr("eval.embedding.evaluate.evaluate_topic", fake_evaluate)
    monkeypatch.setattr("eval.embedding.evaluate.format_report", lambda report: "")
    monkeypatch.setattr("eval.embedding.evaluate.write_report", lambda report, out: out)

    assert cli.main(["evaluate", "--topic", "t", "--view", "human", "--language", "en"]) == 0
    assert cli.main(["evaluate", "--topic", "t", "--view", "human"]) == 0
    assert seen == ["en", None]
