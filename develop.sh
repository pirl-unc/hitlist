#!/usr/bin/env bash

set -e

# Install into the virtualenv that is already active, if there is one.
#
# This script used to create and activate ./.venv unconditionally.  Run from a
# shell that already had a virtualenv active, it therefore installed into a
# *different* environment than the developer was using -- reporting success
# while `hitlist` on PATH went on resolving to whatever stale copy was in the
# active environment.  That is how a 1.45.0 tree survived several releases
# behind a 1.55.0 checkout, and it fails silently in exactly the way that is
# hardest to notice: the install works, it is just somewhere else.
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Installing into the active virtualenv: $VIRTUAL_ENV"
else
    VENV_DIR=".venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment at $VENV_DIR..."
        python -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# Check if UV is installed and available in the PATH
if command -v uv &> /dev/null; then
    echo "Using uv to install package with development dependencies..."
    uv pip install -e ".[dev]"
else
    echo "uv not found, falling back to regular pip..."
    pip install -e ".[dev]"
fi

# Say where it landed and which code the console script will run, and fail if
# the environment's hitlist distribution metadata disagrees with that code.
# The failure this guards against is not an install error -- it is an install
# that succeeds into the wrong place, or leaves stale or duplicate metadata
# behind (#553), so the install's exit status cannot confirm it.  The check never
# removes anything; it names what to move aside.
echo
python scripts/check_dev_install.py
