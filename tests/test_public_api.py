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

"""The curated top-level API: importable from `hitlist` directly + discoverable."""

import pytest

import hitlist
from hitlist import _PUBLIC_API


def test_top_level_entry_points_importable():
    from hitlist import (  # noqa: F401
        ProteomeIndex,
        VersionedDatasetRegistry,
        build_observations,
        fetch,
        generate_ms_observations_table,
        load_ms_observations,
    )

    assert callable(build_observations)
    assert callable(fetch)


def test_public_api_is_discoverable():
    assert "build_observations" in dir(hitlist)
    assert "load_ms_observations" in hitlist.__all__
    assert "VersionedDatasetRegistry" in hitlist.__all__


def test_unknown_attribute_raises_attributeerror():
    missing = "this_does_not_exist"
    with pytest.raises(AttributeError):
        getattr(hitlist, missing)


@pytest.mark.parametrize("name", sorted(_PUBLIC_API))
def test_every_public_api_entry_resolves(name):
    """Every `_PUBLIC_API` key must import from the module it names.

    The lazy `__getattr__` means a typo'd module path or a renamed
    function is not an import error at startup -- it is an AttributeError
    the first time some user touches that name, which may be never in our
    own test suite.  Resolving all of them here is the only thing that
    makes the dict self-checking, and it replaces the per-function
    `assert hitlist.x is x` assertions that were accumulating in
    `test_curation.py`, a file about curation logic rather than wiring.
    """
    import importlib

    attribute = getattr(hitlist, name)
    module = importlib.import_module(_PUBLIC_API[name], package="hitlist")
    assert attribute is getattr(module, name)
    assert name in hitlist.__all__
    assert name in dir(hitlist)
