"""Independent byte snapshots for immutable evidence-package generations."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def snapshot_file(source, target):
    """Copy verified source bytes to a new inode, preferring a filesystem clone.

    Clone operations are copy-on-write where the filesystem supports them.
    A full copy is the portable fallback. Neither path may hard-link a mutable
    ingestion file or a predecessor package into a published generation.
    """
    source = Path(source).resolve(strict=True)
    target = Path(target)
    if not source.is_file():
        raise ValueError(f"Evidence snapshot source is not a file: {source}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Evidence snapshot target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    command = None
    if sys.platform == "darwin":
        command = ["cp", "-c", "-p", str(source), str(target)]
    elif sys.platform.startswith("linux"):
        command = [
            "cp",
            "--reflink=auto",
            "--preserve=mode,timestamps",
            "--",
            str(source),
            str(target),
        ]
    try:
        if command is not None:
            try:
                subprocess.run(
                    command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
            except (OSError, subprocess.CalledProcessError):
                target.unlink(missing_ok=True)
                shutil.copy2(source, target)
        else:
            shutil.copy2(source, target)
        source_stat, target_stat = source.stat(), target.stat()
        if source_stat.st_dev == target_stat.st_dev and source_stat.st_ino == target_stat.st_ino:
            raise ValueError(f"Evidence snapshot shares a source inode: {source}")
    except Exception:
        target.unlink(missing_ok=True)
        raise
