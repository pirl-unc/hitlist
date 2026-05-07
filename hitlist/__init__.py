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

Side effect at import time
--------------------------
Importing ``hitlist`` (or any submodule) sets
``pandas.options.future.infer_string = True``.  This is the pandas 3.0
default, available as a future flag in 2.1+.  It cuts string-column memory
~5x at every layer of the build / load pipeline by switching pandas from
``object`` dtype (Python ``str`` references, ~50-100 bytes/cell) to
pyarrow-backed ``StringDtype`` (~10 bytes/cell, identical layout to parquet).

Downstream consumers that import hitlist inherit the behavior — this is
forward-compatible (pandas 3.x will do this by default).  Code that depends
on the legacy ``object`` representation can opt back out:

    import hitlist
    import pandas as pd
    pd.options.future.infer_string = False

The flag is wrapped in :func:`contextlib.suppress(AttributeError)` so a
future pandas release that removes the option (after promoting it to the
permanent default) won't break import.
"""

import contextlib as _contextlib

import pandas as _pd

with _contextlib.suppress(AttributeError):
    _pd.options.future.infer_string = True

from .version import __version__

__all__ = ["__version__"]
