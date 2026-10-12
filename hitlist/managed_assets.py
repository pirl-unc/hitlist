"""Shared filesystem bounds for explicitly acquired, checksum-pinned assets."""

import os
import stat
from pathlib import Path


def check_cache_path(root, path):
    root, path = Path(root), Path(path)
    if ".." in path.relative_to(root).parts:
        raise ValueError("Managed cache paths cannot traverse outside their root")
    for relative in reversed(path.relative_to(root).parents):
        if (root / relative).is_symlink():
            raise ValueError(f"Symlinks are not supported in the managed cache: {root / relative}")
    if root.is_symlink() or path.is_symlink():
        raise ValueError(f"Symlinks are not supported in the managed cache: {path}")


def cache_bytes(root):
    """Count regular files, including partials; fail closed on unsupported types."""
    check_cache_path(root, root)
    total = 0

    def fail(error):
        raise error

    for directory, dirs, files in os.walk(root, onerror=fail):
        for name in dirs + files:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"Unsupported file type in managed cache: {path}")
    return total
