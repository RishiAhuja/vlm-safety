#!/usr/bin/env bash
# =============================================================================
# setup.sh — bootstrap the VLM Cultural Safety project on a fresh machine.
#
# What it does:
#   1. Installs Ollama if not present (Linux curl-pipe or macOS brew)
#   2. Starts the Ollama daemon if not already running
#   3. Pulls the three VLM models and the judge LLM
#   4. Creates a Python virtual environment and installs dependencies
#   5. Copies .env.example → .env (you must fill in OPENAI_API_KEY)
#
# Usage:  bash setup.sh
# =============================================================================
set -euo pipefail

MODELS=(
    "llava:7b"          # LLaVA 7B — Western-origin VLM
    "qwen2.5vl:3b"      # Qwen 2.5 VL 3B — Eastern-origin VLM
    "minicpm-v"         # MiniCPM-V — Eastern-origin VLM
    "llama3.1:8b"       # LLaMA 3.1 8B — optional local fallback judge
)

# ── colour helpers ────────────────────────────────────────────────────────────
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }

# ── 1. Install Ollama ─────────────────────────────────────────────────────────
if command -v ollama &>/dev/null; then
    green "✓ Ollama already installed ($(ollama --version))"
else
    yellow "▸ Installing Ollama..."
    if [[ "$(uname)" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            brew install ollama
        else
            red "Homebrew not found. Install it from https://brew.sh then re-run."
            exit 1
        fi
    else
        # Linux — official install script
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    green "✓ Ollama installed"
fi

# ── 2. Start Ollama daemon ────────────────────────────────────────────────────
if ollama list &>/dev/null; then
    green "✓ Ollama daemon is running"
else
    yellow "▸ Starting Ollama daemon..."
    if [[ "$(uname)" == "Darwin" ]]; then
        open -a Ollama 2>/dev/null || OLLAMA_NUM_PARALLEL=3 ollama serve &>/dev/null &
    else
        # Linux — configure systemd override for GPU parallel slots, then start
        if systemctl list-unit-files ollama.service &>/dev/null 2>&1; then
            # Inject OLLAMA_NUM_PARALLEL into the service environment
            sudo mkdir -p /etc/systemd/system/ollama.service.d
            sudo tee /etc/systemd/system/ollama.service.d/parallel.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_NUM_PARALLEL=3"
EOF
            sudo systemctl daemon-reload
            sudo systemctl restart ollama 2>/dev/null || true
            green "✓ Ollama systemd service restarted with OLLAMA_NUM_PARALLEL=3"
        else
            OLLAMA_NUM_PARALLEL=3 ollama serve &>/dev/null &
            sleep 3
        fi
    fi
    # Wait up to 15 s for daemon to be ready
    for i in $(seq 1 15); do
        if ollama list &>/dev/null; then
            green "✓ Ollama daemon ready"
            break
        fi
        sleep 1
        if [[ $i -eq 15 ]]; then
            red "Ollama daemon did not start in time. Run 'ollama serve' manually."
            exit 1
        fi
    done
fi

# ── 3. Pull models ────────────────────────────────────────────────────────────
for model in "${MODELS[@]}"; do
    if ollama list | grep -q "${model%%:*}"; then
        green "✓ $model already pulled"
    else
        yellow "▸ Pulling $model (this may take a while)..."
        ollama pull "$model"
        green "✓ $model pulled"
    fi
done

# ── 4. Python virtual environment ─────────────────────────────────────────────
if [[ -d venv ]]; then
    green "✓ Python venv already exists"
else
    yellow "▸ Creating Python virtual environment..."
    python3 -m venv venv
    green "✓ venv created"
fi

yellow "▸ Installing Python dependencies..."
# shellcheck disable=SC1091
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
green "✓ Python dependencies installed"

# ── 5. Environment file ───────────────────────────────────────────────────────
if [[ -f .env ]]; then
    green "✓ .env already exists"
else
    cp .env.example .env
    yellow "▸ .env created from .env.example"
    yellow "  → Open .env and set your OPENAI_API_KEY before running the judge step."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
green "═══════════════════════════════════════════════════════════════"
green "  Setup complete. Run the pipeline:"
green ""
green "    source venv/bin/activate"
green "    python run_all.py           # full pipeline"
green "    python run_all.py --step infer    # inference only"
green "    python run_all.py --step judge    # judging only"
green "    python run_all.py --step analyze  # analysis only"
green "═══════════════════════════════════════════════════════════════"
