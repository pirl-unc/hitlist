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

"""hitlist: curated and harmonized MHC ligand mass spectrometry data for pMHC target selection and model training.

Side effects at import time
---------------------------
Importing ``hitlist`` (or any submodule) does two things:

1. Sets ``pandas.options.future.infer_string = True``.  This is the pandas 3.0
   default, available as a future flag in 2.1+.  Cuts string-column memory
   ~5x at every layer of the build / load pipeline by switching pandas from
   ``object`` dtype (Python ``str`` references, ~50-100 bytes/cell) to
   pyarrow-backed ``StringDtype`` (~10 bytes/cell, identical layout to parquet).

   Downstream consumers that import hitlist inherit the behavior — this is
   forward-compatible (pandas 3.x will do this by default).  Code that depends
   on the legacy ``object`` representation can opt back out::

       import hitlist
       import pandas as pd
       pd.options.future.infer_string = False

   The flag is wrapped in :func:`contextlib.suppress(AttributeError)` so a
   future pandas release that removes the option (after promoting it to the
   permanent default) won't break import.

2. Removes the legacy ``~/.hitlist/index/`` cache directory if present.
   That cache was a per-source CSV-scan artifact obsoleted in v1.30.41 when
   ``get_index()`` started deriving counts from ``observations.parquet``
   directly.  The cleanup is one-shot and idempotent — if the directory
   isn't there, this is a no-op.  Wrapped in ``suppress`` so a permissions
   issue can't break ``import hitlist``.
"""

import contextlib as _contextlib
import shutil as _shutil

import pandas as _pd

with _contextlib.suppress(AttributeError):
    _pd.options.future.infer_string = True


def _cleanup_legacy_index_dir() -> None:
    """One-shot removal of the obsolete ``~/.hitlist/index/`` cache.

    Pre-v1.30.41 hitlist wrote per-source allele-count parquets to this
    directory as a CSV-scan cache.  ``get_index()`` now derives all
    counts live from ``observations.parquet``, so the cache is dead
    weight — but a pip upgrade leaves the inert files behind.  This
    cleanup removes them so they don't mislead users browsing
    ``~/.hitlist/`` looking for the source of "stale" indexes.
    """
    from .downloads import data_dir

    legacy = data_dir() / "index"
    if legacy.exists():
        _shutil.rmtree(legacy, ignore_errors=True)


with _contextlib.suppress(Exception):
    _cleanup_legacy_index_dir()

from .version import __version__  # noqa: E402  -- after side-effect setup

__all__ = ["__version__"]
