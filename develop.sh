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

# Say where it landed and which code the console script will run.  The failure
# this guards against is not an install error -- it is an install that
# succeeds into the wrong place, so the only useful confirmation is the
# resolved path, not an exit code.
echo
python - <<'PY'
import shutil, subprocess, sys
import hitlist

print(f"import hitlist -> {hitlist.__version__}  ({hitlist.__file__})")
cli = shutil.which("hitlist")
if cli is None:
    print("WARNING: no `hitlist` on PATH")
    sys.exit(0)
reported = subprocess.run([cli, "--version"], capture_output=True, text=True).stdout.strip()
print(f"`hitlist` on PATH -> {cli}")
print(f"                    {reported}")
if hitlist.__version__ not in reported:
    print(
        f"\nWARNING: the console script reports {reported!r} but the importable "
        f"package is {hitlist.__version__}.\n"
        "Another install is shadowing this one; `pip uninstall hitlist` until "
        "none remain, then re-run this script."
    )
PY
