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
