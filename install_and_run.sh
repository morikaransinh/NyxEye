#!/bin/bash

# Project Daredevil - Setup and Launch Script
# This script installs all dependencies listed in the README and then runs the demo via ./main.sh.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

log() {
    printf "\033[0;32m[INFO]\033[0m %s\n" "$1"
}

warn() {
    printf "\033[1;33m[WARN]\033[0m %s\n" "$1"
}

error() {
    printf "\033[0;31m[ERROR]\033[0m %s\n" "$1"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        error "Required command '$1' not found. Please install it and re-run this script."
        exit 1
    fi
}

log "Checking required tools..."
require_command python3

if [[ "$OSTYPE" == darwin* ]]; then
    if ! command -v brew >/dev/null 2>&1; then
        warn "Homebrew not found. OpenAL Soft installation will be skipped. Install Homebrew to enable spatial audio: https://brew.sh"
    fi
fi

ENV_DIR="$REPO_DIR/env"

if [[ ! -d "$ENV_DIR" ]]; then
    log "Creating Python virtual environment at $ENV_DIR..."
    python3 -m venv "$ENV_DIR"
else
    log "Virtual environment already exists at $ENV_DIR."
fi

log "Activating virtual environment..."
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

log "Upgrading pip..."
pip install --upgrade pip

log "Installing Python dependencies..."
pip install torch torchvision transformers opencv-python numpy ultralytics PyOpenAL pygame

if [[ "$OSTYPE" == darwin* ]]; then
    if command -v brew >/dev/null 2>&1; then
        if ! brew list --versions openal-soft >/dev/null 2>&1; then
            log "Installing OpenAL Soft via Homebrew..."
            brew install openal-soft
        else
            log "OpenAL Soft already installed."
        fi
    else
        warn "Skipping OpenAL Soft installation (Homebrew not available)."
    fi
else
    warn "Skipping OpenAL Soft installation (supported on macOS only)."
fi

if [[ ! -x "$REPO_DIR/main.sh" ]]; then
    log "Making main.sh executable..."
    chmod +x "$REPO_DIR/main.sh"
fi

log "Setup complete. Launching demo..."
./main.sh "$@"

