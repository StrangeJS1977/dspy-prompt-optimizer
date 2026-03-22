cd /Users/jorgenstrangeolsen/2026_ML/dspy-prompt-optimizer && .venv/bin/python -m pytest tests/ -v

cd "/Users/jorgenstrangeolsen/2026_ML/dspy-prompt-optimizer"

# Reparer gh permissions (én gang):
sudo chown -R $(whoami) ~/.config
mkdir -p ~/.config/gh && chmod 700 ~/.config/gh

# Login:
gh auth login

# Opret repo og push:
git init
git add -A
git commit -m "Initial release v0.1.0"
gh repo create dspy-prompt-optimizer \
    --public \
    --description "Automatisk DSPy-baseret prompt-optimering til lokale og cloud LLMs" \
    --source=. \
    --push \
    --remote=origin

# Åbn i VS Code:
code .