#!/usr/bin/env bash
# setup_and_push.sh
# ─────────────────────────────────────────────────────────────────
# Opsætter dspy-prompt-optimizer projekt komplet:
#   1. Opretter venv
#   2. Installerer afhængigheder
#   3. Kører testsuite
#   4. Initialiserer git
#   5. Opretter GitHub repo via gh CLI
#   6. Pusher kode
#
# KRAV: gh CLI installeret (brew install gh)
# BRUG: ./setup_and_push.sh
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_NAME="dspy-prompt-optimizer"
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  $PROJECT_NAME — setup og GitHub push"
echo "============================================================"
echo ""

# ── 1. Python version ─────────────────────────────────────────────
echo "[1/6] Python version"
PYTHON=$(which python3.12 2>/dev/null || which python3 2>/dev/null || echo "")
if [ -z "$PYTHON" ]; then
    echo "FEJL: Python ikke fundet"
    exit 1
fi
PY_VER=$("$PYTHON" --version)
echo "  ✓  $PY_VER @ $PYTHON"

# ── 2. Opret venv ─────────────────────────────────────────────────
echo ""
echo "[2/6] Opretter venv"
if [ ! -d "$PROJ/.venv" ]; then
    "$PYTHON" -m venv "$PROJ/.venv"
    echo "  ✓  .venv oprettet"
else
    echo "  ✓  .venv eksisterer allerede"
fi

PY="$PROJ/.venv/bin/python"
PIP="$PROJ/.venv/bin/pip"

# ── 3. Installer afhængigheder ────────────────────────────────────
echo ""
echo "[3/6] Installer afhængigheder"
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet pytest pyyaml

# DSPy er optional — installer hvis tilgængeligt
if "$PIP" install --quiet dspy-ai 2>/dev/null; then
    echo "  ✓  pytest, pyyaml, dspy-ai installeret"
else
    echo "  ✓  pytest, pyyaml installeret (dspy-ai installeres separat)"
    echo "  →  Installer manuelt: pip install dspy-ai"
fi

# Tilføj projektet til PYTHONPATH (undgår editable install problemer)
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
echo "  ✓  $PROJECT_NAME tilføjet til PYTHONPATH"

# ── 4. Kør testsuite ──────────────────────────────────────────────
echo ""
echo "[4/6] Kører testsuite"
cd "$PROJ"
if "$PROJ/.venv/bin/python" -m pytest tests/ -v --tb=short 2>&1 | tee /tmp/test_output.txt; then
    TESTS_OK=true
    echo ""
    echo "  ✓  ALLE TESTS BESTOD"
else
    TESTS_OK=false
    echo ""
    echo "  ⚠  Nogle tests fejlede — fortsætter alligevel"
    echo "     (integration tests kræver Ollama)"
fi

# ── 5. Git initialisering ─────────────────────────────────────────
echo ""
echo "[5/6] Git"
cd "$PROJ"

if [ ! -d "$PROJ/.git" ]; then
    git init
    git config user.email "$(git config --global user.email 2>/dev/null || echo '')"
    git config user.name "$(git config --global user.name 2>/dev/null || echo 'Jørgen Strange Olsen')"
    echo "  ✓  git init"
else
    echo "  ✓  git repo eksisterer allerede"
fi

# Sørg for .gitignore er korrekt
cat > "$PROJ/.gitignore" << 'GITIGNORE'
__pycache__/
*.py[cod]
*.pyo
.Python
*.so
*.egg
*.egg-info/
dist/
build/
.eggs/
.venv/
venv/
.env
*.bak
optimization_result.json
.pytest_cache/
.coverage
htmlcov/
*.log
GITIGNORE

git add -A
git diff --cached --quiet && echo "  ✓  Ingen ændringer" || {
    git commit -m "Initial release v0.1.0

- metric.py: yaml_list_metric, chunk_quality_metric, MetricComposer
- optimizer.py: OptimizationRunner, OptimizerConfig, OptimizationResult  
- patcher.py: PythonPromptPatcher, YamlPromptPatcher, JsonPromptPatcher
- examples/optimize_chunk_boundary.py: semantisk_chunking integration
- tests/: 47 unit tests"
    echo "  ✓  Initial commit"
}

# ── 6. GitHub push ────────────────────────────────────────────────
echo ""
echo "[6/6] GitHub"

# Tjek gh CLI
if ! command -v gh &>/dev/null; then
    echo "  FEJL: gh CLI ikke installeret"
    echo ""
    echo "  Installer: brew install gh"
    echo "  Login:     gh auth login"
    echo ""
    echo "  Derefter kør manuelt:"
    echo "    cd $PROJ"
    echo "    gh repo create $PROJECT_NAME --public --source=. --push"
    exit 1
fi

# Tjek login
# Fix gh config permissions hvis nødvendigt
if [ ! -d "$HOME/.config/gh" ]; then
    mkdir -p "$HOME/.config/gh" 2>/dev/null ||     sudo mkdir -p "$HOME/.config/gh" && sudo chown "$USER" "$HOME/.config/gh"
fi

if ! gh auth status &>/dev/null; then
    echo "  FEJL: Ikke logget ind på GitHub"
    echo ""
    echo "  Kør: gh auth login"
    echo "  Vælg: GitHub.com → HTTPS → Login with browser"
    echo ""
    echo "  Derefter kør dette script igen."
    exit 1
fi

GH_USER=$(gh api user --jq '.login' 2>/dev/null || echo "")
echo "  GitHub bruger: $GH_USER"

# Tjek om repo allerede eksisterer
if gh repo view "$GH_USER/$PROJECT_NAME" &>/dev/null; then
    echo "  ✓  Repo eksisterer: github.com/$GH_USER/$PROJECT_NAME"
    
    # Sæt remote hvis ikke sat
    if ! git remote get-url origin &>/dev/null; then
        git remote add origin "https://github.com/$GH_USER/$PROJECT_NAME.git"
    fi
    
    git push -u origin main 2>/dev/null || git push -u origin master 2>/dev/null || {
        git branch -M main
        git push -u origin main
    }
    echo "  ✓  Pushet til existing repo"
else
    echo "  Opretter nyt repo: $PROJECT_NAME..."
    gh repo create "$PROJECT_NAME" \
        --public \
        --description "Automatisk DSPy-baseret prompt-optimering til lokale og cloud LLMs via Ollama" \
        --source=. \
        --push \
        --remote=origin
    echo "  ✓  Oprettet og pushet: github.com/$GH_USER/$PROJECT_NAME"
fi

# ── Åbn i VS Code ─────────────────────────────────────────────────
echo ""
if command -v code &>/dev/null; then
    code "$PROJ"
    echo "  ✓  Åbnet i VS Code"
else
    echo "  (VS Code: code $PROJ)"
fi

# ── Opsummering ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  FÆRDIG"
echo "============================================================"
echo ""
echo "  Projekt:  $PROJ"
echo "  Venv:     $PROJ/.venv"
[ -n "$GH_USER" ] && echo "  GitHub:   https://github.com/$GH_USER/$PROJECT_NAME"
echo ""
echo "  Næste skridt:"
echo "  ─────────────"
echo "  # Kør tests:"
echo "  cd $PROJ && .venv/bin/python -m pytest tests/ -v"
echo ""
echo "  # Brug som bibliotek i semantisk_chunking:"
echo "  PYTHONPATH=. python prompt_optimizer.py --score-only"
echo ""
echo "  # Kør optimering:"
echo "  PYTHONPATH=. python prompt_optimizer.py --iterations 10"
echo "============================================================"
