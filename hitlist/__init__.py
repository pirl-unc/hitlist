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

"""hitlist: curated and harmonized MHC ligand mass spectrometry data for pMHC target selection and model training."""

import pandas as _pd

# Opt every hitlist process into pandas' pyarrow-backed ``StringDtype``
# (pandas 3.0 default, available in 2.1+ as a future flag).  Cuts string-
# column memory ~5x at every layer — scanner output, parquet reads, the
# build pipeline.  Downstream consumers that import hitlist inherit the
# behavior, which is forward-compatible (pandas 3.0 will set this by
# default anyway).  Set BEFORE any DataFrame construction so the
# inference applies from process start.
_pd.options.future.infer_string = True

from .version import __version__  # noqa: E402  -- after option set

__all__ = ["__version__"]
