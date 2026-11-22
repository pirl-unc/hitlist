# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public exception hierarchy for failures a user can act on (#886).

``hitlist.cli.main`` catches :class:`HitlistError` once, prints ``Error: ...``
to stderr and exits 1; any other exception is a bug and keeps its traceback.
All classes subclass ``RuntimeError`` (``DatasetError`` also ``ValueError``) so
pre-existing ``except`` callers keep working. Raise these with ``raise ... from cause``
so the original exception type stays inspectable via ``__cause__``.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "DatasetError",
    "DownloadError",
    "HitlistError",
    "InsufficientSpaceError",
    "IntegrityError",
    "OfflineError",
]


class HitlistError(RuntimeError):
    """Base class for expected, user-actionable hitlist failures."""


class DownloadError(HitlistError):
    """A network or filesystem failure while fetching a file (HTTP, timeout, disk)."""


class OfflineError(DownloadError):
    """Raised when a network download is required while offline mode is enabled (#960)."""


class InsufficientSpaceError(DownloadError):
    """Target filesystem does not have enough free space for the planned download (#884).

    Subclasses :class:`DownloadError` (and therefore :class:`HitlistError` and
    ``RuntimeError``) and exposes ``path``, ``required_bytes`` and ``available_bytes``
    so programmatic callers can act on space shortfalls without parsing messages.
    """

    def __init__(
        self,
        message: str,
        *,
        path: Path | str | None = None,
        required_bytes: int | None = None,
        available_bytes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.path = Path(path) if path is not None else None
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes


class IntegrityError(HitlistError):
    """Fetched or cached bytes failed validation (size, SHA-256, corrupt archive)."""


class DatasetError(HitlistError, ValueError):
    """A dataset name cannot be fetched as asked (unknown, manual, or multi-file bundle)."""
