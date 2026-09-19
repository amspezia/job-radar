# Job Radar task runner. Run `just` to list targets.

# Show available recipes
default:
    @just --list

# Lint with ruff
lint:
    uv run ruff check .

# Auto-format (and apply lint fixes) with ruff
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Run the test suite
test:
    uv run pytest

# ---------------------------------------------------------------------------
# Eval Ollama — dedicated instance on :11435 so eval never blocks ingest
# ---------------------------------------------------------------------------

_EVAL_OLLAMA_PORT := "11435"
_EVAL_OLLAMA_URL  := "http://localhost:" + _EVAL_OLLAMA_PORT
_EVAL_OLLAMA_PID  := "/tmp/ollama-eval.pid"

# Apply performance settings to the main Ollama (port 11434) and restart it
ollama-configure:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Applying Ollama performance settings via launchctl…"
    launchctl setenv OLLAMA_NUM_PARALLEL 12
    launchctl setenv OLLAMA_FLASH_ATTENTION 1
    launchctl setenv OLLAMA_MAX_LOADED_MODELS 2
    launchctl setenv OLLAMA_CONTEXT_LENGTH 4096
    launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
    echo "Restarting Ollama…"
    pkill -x ollama 2>/dev/null || true
    sleep 1
    OLLAMA_NUM_PARALLEL=12 OLLAMA_FLASH_ATTENTION=1 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &>/tmp/ollama-main.log &
    echo "Waiting for Ollama to be ready…"
    for i in $(seq 1 20); do
        curl -sf http://localhost:11434/ >/dev/null 2>&1 && break
        sleep 1
    done
    echo "Main Ollama ready with NUM_PARALLEL=12 FLASH_ATTENTION=1 CONTEXT=4096 KV_CACHE=q8_0"

# Start the eval Ollama instance (port 11435) and pre-warm both models
eval-ollama-start:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -f {{_EVAL_OLLAMA_PID}} ] && kill -0 "$(cat {{_EVAL_OLLAMA_PID}})" 2>/dev/null; then
        echo "eval Ollama already running (pid $(cat {{_EVAL_OLLAMA_PID}}))"
        exit 0
    fi
    OLLAMA_HOST=127.0.0.1:{{_EVAL_OLLAMA_PORT}} OLLAMA_NUM_PARALLEL=12 OLLAMA_FLASH_ATTENTION=1 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &>/tmp/ollama-eval.log &
    echo $! > {{_EVAL_OLLAMA_PID}}
    echo "started eval Ollama (pid $!) on port {{_EVAL_OLLAMA_PORT}}, waiting for ready…"
    for i in $(seq 1 20); do
        curl -sf {{_EVAL_OLLAMA_URL}}/ >/dev/null 2>&1 && break
        sleep 1
    done
    echo "warming up nomic-embed-text…"
    curl -sf -X POST {{_EVAL_OLLAMA_URL}}/api/embed \
        -H "Content-Type: application/json" \
        -d '{"model":"nomic-embed-text","input":"warmup","keep_alive":-1}' >/dev/null
    echo "warming up generation model…"
    curl -sf -X POST {{_EVAL_OLLAMA_URL}}/api/generate \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$(grep GENERATION_MODEL .env | cut -d= -f2 | tr -d '\"' || echo qwen2.5:3b)\",\"prompt\":\"hi\",\"keep_alive\":-1}" >/dev/null
    echo "eval Ollama ready — both models loaded"

# Stop the eval Ollama instance
eval-ollama-stop:
    #!/usr/bin/env bash
    if [ -f {{_EVAL_OLLAMA_PID}} ]; then
        pid=$(cat {{_EVAL_OLLAMA_PID}})
        kill "$pid" 2>/dev/null && echo "stopped eval Ollama (pid $pid)" || echo "process already gone"
        rm -f {{_EVAL_OLLAMA_PID}}
    else
        echo "no eval Ollama pid file found"
    fi

# Interactive labeling session (uses eval Ollama on :11435)
eval-label *args:
    OLLAMA_BASE_URL={{_EVAL_OLLAMA_URL}} uv run job-radar-eval-label {{args}}

# Run eval metrics for all configs (uses eval Ollama on :11435)
eval-run *args:
    OLLAMA_BASE_URL={{_EVAL_OLLAMA_URL}} uv run job-radar-eval-run {{args}}

# OAT parameter sweep (uses eval Ollama on :11435)
eval-sweep *args:
    OLLAMA_BASE_URL={{_EVAL_OLLAMA_URL}} uv run job-radar-eval-sweep {{args}}

# Commit golden baseline from latest eval run (activates CI regression gate)
eval-commit-golden *args:
    uv run python scripts/commit_eval_golden.py {{args}}

# Inject the 5 synthetic personas + 100 jobs into the DB (idempotent)
eval-inject-synthetic *args:
    uv run job-radar-eval-inject-synthetic {{args}}

# Remove all synthetic persona/job/label data
eval-teardown-synthetic:
    uv run job-radar-eval-inject-synthetic --teardown

# Per-persona retrieval eval over the synthetic personas (uses eval Ollama on :11435)
eval-run-synthetic *args:
    OLLAMA_BASE_URL={{_EVAL_OLLAMA_URL}} uv run job-radar-eval-run-synthetic {{args}}

# CV-parsing quality eval against the 5 personas' ground truth (uses eval Ollama on :11435)
eval-profile-parsing:
    OLLAMA_BASE_URL={{_EVAL_OLLAMA_URL}} uv run job-radar-eval-profile-parsing

# Wipe all eval labels (irreversible — forces full re-labeling)
eval-reset-labels:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run python scripts/reset_eval_labels.py

# ---------------------------------------------------------------------------
# EVALS database — the separate database the embedding-eval harness owns
# ---------------------------------------------------------------------------

# Create EVALS_DATABASE_URL's database if absent, then migrate it to head
evals-db-init:
    uv run job-radar-evals-db init

# Upgrade the EVALS database to head
evals-db-migrate:
    uv run job-radar-evals-db migrate

# Current revision and row counts of the embedding_* tables
evals-db-status:
    uv run job-radar-evals-db status

# ---------------------------------------------------------------------------
# Embedding eval harness — topics, runs, judge and metrics
# ---------------------------------------------------------------------------

# Import a tier's pool and queries into the EVALS database (reads prod read-only)
embedding-import *args:
    uv run job-radar-eval-embedding import-topic {{args}}

# Fill the vector cache for a topic without creating runs (uses eval Ollama on :11435)
embedding-embed *args:
    OLLAMA_BASE_URL={{_EVAL_OLLAMA_URL}} uv run job-radar-eval-embedding embed {{args}}

# Embed, rank and store a full ranking per embedder (uses eval Ollama on :11435)
embedding-run *args:
    OLLAMA_BASE_URL={{_EVAL_OLLAMA_URL}} uv run job-radar-eval-embedding run {{args}}

# Run the parity checks that prove the import reproduces production's dense arm
embedding-verify *args:
    uv run job-radar-eval-embedding verify {{args}}

# Grade postings yourself, blind to every model score: the only gold labels
embedding-label *args:
    uv run job-radar-eval-embedding label-blind {{args}}

# Grade a topic's pool with the LLM judge (uses eval Ollama on :11435)
embedding-judge *args:
    OLLAMA_BASE_URL={{_EVAL_OLLAMA_URL}} uv run job-radar-eval-embedding judge {{args}}

# Score a judge run against your blind labels
embedding-judge-calibrate *args:
    uv run job-radar-eval-embedding judge-calibrate {{args}}

# Metrics, bootstrap intervals and negative controls for a topic
embedding-evaluate *args:
    uv run job-radar-eval-embedding evaluate {{args}}

# List the imported topics
embedding-topics:
    uv run job-radar-eval-embedding topics

# Back up the human label rows as JSON
embedding-labels-export *args:
    uv run job-radar-eval-embedding labels-export {{args}}

# Restore human label rows from a JSON backup
embedding-labels-import *args:
    uv run job-radar-eval-embedding labels-import {{args}}
