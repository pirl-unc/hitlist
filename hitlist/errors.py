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
All classes subclass ``RuntimeError`` so pre-existing ``except RuntimeError``
callers keep working. Raise these with ``raise ... from cause`` so the original
exception type stays inspectable via ``__cause__``.
"""

__all__ = ["DownloadError", "HitlistError", "IntegrityError"]


class HitlistError(RuntimeError):
    """Base class for expected, user-actionable hitlist failures."""


class DownloadError(HitlistError):
    """A network or filesystem failure while fetching a file (HTTP, timeout, disk)."""


class IntegrityError(HitlistError):
    """Fetched or cached bytes failed validation (size, SHA-256, corrupt archive)."""
